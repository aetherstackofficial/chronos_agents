"""Strategy script library (PROTOCOL_V2 section 1) — real files on disk.

``ScriptStore`` persists user strategy scripts under ``scripts_library/`` so
users can refer to them again across bridge restarts. Every script is a real
``<script_id>.py`` file plus one shared ``index.json`` holding the metadata.
IDs are 8 lowercase hex characters.

Durability model:
* The files are the source of truth for *code*; ``index.json`` is the source
  of truth for *metadata* (name, language, timestamps).
* If ``index.json`` is missing or corrupt, the index is reconstructed from the
  ``*.py`` files on disk (name falls back to the script id, timestamps to the
  file mtime) — scripts are never silently lost.

Concurrency: a re-entrant lock guards every mutation and the index flush. The
bridge runs a single asyncio loop and calls these methods via
``asyncio.to_thread``, so this level of locking is sufficient.

``scripts_library/`` is gitignored (the integrator owns ``.gitignore``).
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("chronos.analytics")

_ROOT = Path(__file__).resolve().parent.parent  # swarm_orchestrator/
_ID_RE = re.compile(r"^[0-9a-f]{8}$")
_INDEX_NAME = "index.json"


def _utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


class ScriptStore:
    """File-backed store of strategy scripts (save / update / get / list / delete).

    Return shapes (PROTOCOL_V2 section 1):

    * ``save``/``update`` -> ``{script_id, name, language, saved_at, updated_at, size, path}``
    * ``get``             -> ``{script_id, name, code, language, saved_at, updated_at, size}``
    * ``list``            -> ``[{script_id, name, language, saved_at, updated_at, size}]``
    * ``delete``          -> ``bool``

    ``save``/``update`` raise :class:`ValueError` for empty code or name;
    ``get``/``update`` raise :class:`KeyError` for an unknown ``script_id``
    (routers map these to HTTP 400 / 404).
    """

    def __init__(self, base_dir: str = "scripts_library") -> None:
        """Create the store rooted at ``base_dir``.

        A relative ``base_dir`` resolves against the ``swarm_orchestrator``
        package root (same convention as ``runner.sandbox``) so the library
        lands in the same place regardless of the process working directory.
        """
        base = Path(base_dir)
        self._dir: Path = base if base.is_absolute() else (_ROOT / base)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._index: dict[str, dict] = {}
        self._load_index()
        logger.info(
            "script store ready at %s (%d script(s))", self._dir, len(self._index)
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #
    def save(
        self,
        name: str,
        code: str,
        language: str = "python",
        script_id: Optional[str] = None,
    ) -> dict:
        """Persist ``code`` as a new script (or overwrite ``script_id`` if given).

        Writes ``<script_id>.py`` and updates ``index.json``. Returns the full
        metadata dict including the file ``path``. Raises :class:`ValueError`
        for empty code/name or a malformed explicit ``script_id``.
        """
        self._validate_code(code)
        self._validate_name(name)
        with self._lock:
            if script_id is not None:
                if not _ID_RE.match(script_id):
                    raise ValueError(
                        f"script_id must be 8 lowercase hex chars, got {script_id!r}"
                    )
                sid = script_id
            else:
                sid = self._new_id()
            now = _utc_now_iso()
            existing = self._index.get(sid)
            meta = {
                "script_id": sid,
                "name": name.strip(),
                "language": language,
                "saved_at": existing["saved_at"] if existing else now,
                "updated_at": now,
                "size": len(code.encode("utf-8")),
            }
            self._path(sid).write_text(code, encoding="utf-8")
            self._index[sid] = meta
            self._flush_index()
            logger.info("script %s saved (%r, %d bytes)", sid, meta["name"], meta["size"])
            return {**meta, "path": str(self._path(sid))}

    def update(
        self,
        script_id: str,
        name: Optional[str] = None,
        code: Optional[str] = None,
    ) -> dict:
        """Update a script's name and/or code. Returns the refreshed metadata.

        Raises :class:`KeyError` for an unknown id and :class:`ValueError`
        when the provided name/code is empty.
        """
        with self._lock:
            meta = self._index.get(script_id)
            if meta is None:
                raise KeyError(f"unknown script_id: {script_id}")
            if code is not None:
                self._validate_code(code)
                self._path(script_id).write_text(code, encoding="utf-8")
                meta["size"] = len(code.encode("utf-8"))
            if name is not None:
                self._validate_name(name)
                meta["name"] = name.strip()
            meta["updated_at"] = _utc_now_iso()
            self._flush_index()
            logger.info("script %s updated", script_id)
            return {**meta, "path": str(self._path(script_id))}

    def get(self, script_id: str) -> dict:
        """Return a script's metadata plus its ``code`` read from disk.

        Raises :class:`KeyError` when the id is unknown or the file vanished.
        """
        with self._lock:
            meta = self._index.get(script_id)
            if meta is None:
                raise KeyError(f"unknown script_id: {script_id}")
            path = self._path(script_id)
            try:
                code = path.read_text(encoding="utf-8")
            except OSError as exc:
                logger.warning("script %s file unreadable (%s); dropping", script_id, exc)
                self._index.pop(script_id, None)
                self._flush_index()
                raise KeyError(f"unknown script_id: {script_id}") from exc
            return {
                "script_id": meta["script_id"],
                "name": meta["name"],
                "code": code,
                "language": meta["language"],
                "saved_at": meta["saved_at"],
                "updated_at": meta["updated_at"],
                "size": meta["size"],
            }

    def list(self) -> list[dict]:
        """Return metadata for every script (no code), newest first."""
        with self._lock:
            metas = [dict(m) for m in self._index.values()]
        metas.sort(key=lambda m: (m.get("saved_at") or "", m["script_id"]), reverse=True)
        return metas

    def delete(self, script_id: str) -> bool:
        """Delete a script's file and index entry. ``False`` if unknown."""
        with self._lock:
            if script_id not in self._index:
                return False
            del self._index[script_id]
            try:
                self._path(script_id).unlink(missing_ok=True)
            except OSError as exc:  # index already updated; file is orphaned at worst
                logger.warning("could not remove script file %s: %s", script_id, exc)
            self._flush_index()
            logger.info("script %s deleted", script_id)
            return True

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_code(code: str) -> None:
        """Reject non-string or blank code with :class:`ValueError`."""
        if not isinstance(code, str) or not code.strip():
            raise ValueError("script code must be a non-empty string")

    @staticmethod
    def _validate_name(name: str) -> None:
        """Reject non-string or blank names with :class:`ValueError`."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("script name must be a non-empty string")

    def _path(self, script_id: str) -> Path:
        """Absolute path of the ``.py`` file backing ``script_id``."""
        return self._dir / f"{script_id}.py"

    def _new_id(self) -> str:
        """Generate a fresh unused 8-hex id."""
        while True:
            sid = uuid.uuid4().hex[:8]
            if sid not in self._index and not self._path(sid).exists():
                return sid

    def _load_index(self) -> None:
        """Load ``index.json``; reconcile with — or reconstruct from — disk."""
        raw: dict = {}
        idx_path = self._dir / _INDEX_NAME
        if idx_path.exists():
            try:
                raw = json.loads(idx_path.read_text(encoding="utf-8")).get("scripts", {})
            except (ValueError, OSError) as exc:
                logger.warning("index.json unreadable (%s); rebuilding from disk", exc)
                raw = {}
        index: dict[str, dict] = {}
        for sid, meta in raw.items():
            if isinstance(meta, dict) and _ID_RE.match(str(sid)) and self._path(sid).exists():
                index[sid] = {
                    "script_id": sid,
                    "name": str(meta.get("name") or sid),
                    "language": str(meta.get("language") or "python"),
                    "saved_at": str(meta.get("saved_at") or _utc_now_iso()),
                    "updated_at": str(meta.get("updated_at") or _utc_now_iso()),
                    "size": int(meta.get("size") or self._path(sid).stat().st_size),
                }
        # Adopt orphan files that exist on disk but are missing from the index.
        for file in sorted(self._dir.glob("*.py")):
            sid = file.stem
            if not _ID_RE.match(sid) or sid in index:
                continue
            stat = file.stat()
            ts = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
            index[sid] = {
                "script_id": sid,
                "name": sid,
                "language": "python",
                "saved_at": ts,
                "updated_at": ts,
                "size": stat.st_size,
            }
            logger.info("adopted orphan script file %s", file.name)
        self._index = index
        self._flush_index()

    def _flush_index(self) -> None:
        """Atomically rewrite ``index.json`` (temp file + ``os.replace``)."""
        payload = json.dumps({"version": 1, "scripts": self._index}, indent=2)
        fd, tmp_name = tempfile.mkstemp(
            prefix=".index-", suffix=".json.tmp", dir=str(self._dir)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
            os.replace(tmp_name, self._dir / _INDEX_NAME)
        except OSError as exc:
            logger.warning("index flush failed: %s", exc)
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
