"""Fixed, data-fitted, per-input observation scaler for the v2 RL agents.

Training and live apply exactly this code to the raw ``build_observation_v2`` vector::

    z = clip((x - center) / scale, -clip, clip)         # float32

``center`` and ``scale`` are fitted once (``python -m training.fit_scaler``) on random-action
episodes of the training split and then never change, so the policy sees the same numbers in
training, evaluation and live (SB3's running-statistics ``VecNormalize`` drifts during training and
would differ live; here it only normalises the REWARD).

Inputs that are already bounded and meaningful (one-hots, time encodings, own-state ratios, the
level strength / touches / age) are pass-through: center 0, scale 1, never fitted
(``is_passthrough``).  The JSON file carries the input names, center, scale, clip, the
``spec_hash`` of the role, the number of samples and the fit date; ``load`` refuses a file whose
names differ from ``agents.features_v2.OBS_SPEC[role]``.

Runtime dependencies: numpy (``agents.features_v2`` is imported lazily for the checks in
``save`` / ``load``).
"""
from __future__ import annotations

import json
import os
import re
import warnings
from typing import Dict, List, Optional, Sequence

import numpy as np

DEFAULT_CLIP = 10.0
_PASS_EXACT = frozenset({
    "tod_sin", "tod_cos", "dow_sin", "dow_cos", "to_close",
    "inventory", "holding", "risk_appetite",
    "mem_hist_avail", "mem_range_pos_today", "mem_range_pos_5d", "mem_year_range_pos",
})
_PASS_RE = re.compile(r"^(sector_.+|last_action_\d+|mem_lvl_(up|dn)\d_(strength|touches|age))$")


def is_passthrough(name: str) -> bool:
    """True for the inputs that are never fitted (center 0, scale 1)."""
    return name in _PASS_EXACT or _PASS_RE.match(name) is not None


def role_of_names(names: Sequence[str]) -> Optional[str]:
    """The role whose ``OBS_SPEC`` equals ``names`` (None if there is none)."""
    from agents.features_v2 import OBS_SPEC
    for role, spec in OBS_SPEC.items():
        if list(spec) == list(names):
            return role
    return None


class ObsScaler:
    """clip((x - center) / scale, -clip, clip) per input.  ``meta`` is free-form fit information
    (role, spec_hash, n_samples, fit date, per-input fit report ...) stored in the JSON."""

    def __init__(self, names: Sequence[str], center, scale, clip: float = DEFAULT_CLIP,
                 meta: Optional[dict] = None) -> None:
        self.names: List[str] = [str(n) for n in names]
        self.center = np.asarray(center, dtype=np.float64).reshape(-1)
        self.scale = np.asarray(scale, dtype=np.float64).reshape(-1)
        n = len(self.names)
        if self.center.shape != (n,) or self.scale.shape != (n,):
            raise ValueError(f"names ({n}), center {self.center.shape} and scale {self.scale.shape} "
                             "must have the same length")
        if not (np.isfinite(self.center).all() and np.isfinite(self.scale).all()):
            raise ValueError("center and scale must be finite")
        if not (self.scale > 0.0).all():
            raise ValueError("every scale must be > 0")
        if not clip > 0.0:
            raise ValueError("clip must be > 0")
        self.clip = float(clip)
        self.meta: dict = dict(meta or {})

    # ---- use
    @property
    def n_inputs(self) -> int:
        return len(self.names)

    def transform(self, vec) -> np.ndarray:
        """float32 scaled copy of ``vec`` (shape (n_inputs,) or (batch, n_inputs))."""
        x = np.asarray(vec, dtype=np.float64)
        if x.shape[-1] != self.center.shape[0]:
            raise ValueError(f"observation has {x.shape[-1]} inputs, scaler has {self.center.shape[0]}")
        z = (np.nan_to_num(x, nan=0.0, posinf=1e30, neginf=-1e30) - self.center) / self.scale
        return np.clip(z, -self.clip, self.clip).astype(np.float32)

    __call__ = transform

    # ---- (de)serialisation
    def to_dict(self) -> dict:
        d = dict(self.meta)
        d.update({"names": list(self.names), "center": [float(c) for c in self.center],
                  "scale": [float(s) for s in self.scale], "clip": self.clip})
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ObsScaler":
        meta = {k: v for k, v in d.items() if k not in ("names", "center", "scale", "clip")}
        return cls(d["names"], d["center"], d["scale"], float(d.get("clip", DEFAULT_CLIP)), meta)

    def save(self, path, role: Optional[str] = None, n_samples: Optional[int] = None,
             fit_date: Optional[str] = None, extra: Optional[dict] = None) -> None:
        """Write the JSON.  ``role`` (inferred from the names if omitted) fixes the ``spec_hash``
        stored with it; raises if the names are not ``OBS_SPEC[role]``."""
        from agents.features_v2 import OBS_SPEC, spec_hash
        role = role or self.meta.get("role") or role_of_names(self.names)
        if role not in OBS_SPEC or list(OBS_SPEC[role]) != self.names:
            raise ValueError(f"scaler names are not OBS_SPEC[{role!r}]")
        self.meta.update(extra or {})
        self.meta["role"] = role
        self.meta["spec_hash"] = spec_hash(role)
        if n_samples is not None:
            self.meta["n_samples"] = int(n_samples)
        if fit_date is not None:
            self.meta["fit_date"] = str(fit_date)
        d = os.path.dirname(os.fspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=1)

    @classmethod
    def load(cls, path, role: Optional[str] = None, strict_hash: bool = False) -> "ObsScaler":
        """Read the JSON.  Raises ``ValueError`` if the names differ from ``OBS_SPEC[role]`` (role:
        argument, else the file's, else inferred).  A stored ``spec_hash`` that no longer matches the
        current one (features_v2 / market_memory source changed) warns, or raises with
        ``strict_hash``."""
        from agents.features_v2 import OBS_SPEC, spec_hash
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        sc = cls.from_dict(d)
        role = role or sc.meta.get("role") or role_of_names(sc.names)
        if role not in OBS_SPEC:
            raise ValueError(f"{path}: input names match no known role (file role {sc.meta.get('role')!r})")
        if list(OBS_SPEC[role]) != sc.names:
            a, b = list(OBS_SPEC[role]), sc.names
            diff = [(x, y) for x, y in zip(a, b) if x != y][:3]
            raise ValueError(f"{path}: input names differ from OBS_SPEC[{role!r}] "
                             f"({len(b)} in file, {len(a)} in spec; first differences {diff})")
        stored = sc.meta.get("spec_hash")
        if stored and stored != spec_hash(role):
            msg = f"{path}: spec_hash differs from the current {role} spec (features changed since the fit?)"
            if strict_hash:
                raise ValueError(msg)
            warnings.warn(msg)
        sc.meta.setdefault("role", role)
        return sc

    @classmethod
    def identity(cls, names: Sequence[str], clip: float = DEFAULT_CLIP) -> "ObsScaler":
        """Pass-through for every input (tests / stubs)."""
        n = len(names)
        return cls(names, np.zeros(n), np.ones(n), clip)

    def __repr__(self) -> str:
        return f"ObsScaler({self.n_inputs} inputs, clip={self.clip:g})"
