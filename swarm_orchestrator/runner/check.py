"""Strategy pre-run checks, part 1: static analysis and the sample ticks.

Used by the bridge (and the sandbox manager) BEFORE a strategy is run:

* :func:`static_check` -- reads the code with ``ast`` (it never executes it) and
  reports problems with line numbers. It replaces the old ``"class UserStrategy"
  in code`` text search, which a comment or a string could satisfy.
* :func:`build_sample_ticks` -- realistic tick snapshots for the dry run
  (runner/dryrun.py), taken from the real engine so they never drift from the
  payload a live strategy receives.

The dry run itself executes untrusted code, so it runs inside the sealed
sandbox (see runner/sandbox.py), never here.

A problem is ``{"stage": str, "line": int | None, "message": str}``.
"""
from __future__ import annotations

import ast
from typing import List, Optional

# Marker the dry run prints before its JSON report, so output from the user's
# own print() calls cannot be mistaken for the result.
DRYRUN_SENTINEL = "@@DRYRUN@@"

# One strategy file may not be bigger than this. It also keeps the code small
# enough to hand to a sandbox container as one environment variable.
MAX_CODE_BYTES = 64 * 1024

# Imports that usually do not work inside the sandbox (no network except the
# engine, no host files). A warning, never an error: the code may only mention
# them, and the real run decides.
# Third-party packages that are NOT in the sandbox image (it has only the
# standard library, pyzmq and the SDK). Importing one fails at run time, so say
# so now; the dry run would catch it too, but a development machine with these
# installed would let the check pass and the sealed run fail.
_NOT_INSTALLED = frozenset({
    "numpy", "pandas", "scipy", "sklearn", "torch", "tensorflow", "matplotlib",
    "statsmodels", "talib", "ta", "yfinance", "gymnasium", "stable_baselines3",
})

_SANDBOX_HOSTILE = frozenset({
    "os", "sys", "subprocess", "socket", "shutil", "ctypes",
    "requests", "urllib", "http", "httpx", "ftplib", "smtplib",
})


def _problem(stage: str, message: str, line: Optional[int] = None) -> dict:
    return {"stage": stage, "line": line, "message": message}


def _base_name(node: ast.expr) -> Optional[str]:
    """The last name in a base-class expression: ``Strategy`` or ``sdk.Strategy``."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _defined_names(tree: ast.Module) -> set:
    """Top-level names a module defines or imports (for the missing-import check)."""
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


def _derives_from_strategy(cls: ast.ClassDef, classes: dict, seen: Optional[set] = None) -> bool:
    """True if ``cls`` reaches ``Strategy`` through its bases (same-file bases followed)."""
    seen = seen or set()
    if cls.name in seen:
        return False
    seen.add(cls.name)
    for base in cls.bases:
        name = _base_name(base)
        if name == "Strategy":
            return True
        if name in classes and _derives_from_strategy(classes[name], classes, seen):
            return True
    return False


def _has_method(cls: ast.ClassDef, classes: dict, method: str, seen: Optional[set] = None) -> bool:
    """True if ``cls`` or a same-file base defines ``method``."""
    seen = seen or set()
    if cls.name in seen:
        return False
    seen.add(cls.name)
    for node in cls.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method:
            return True
    return any(_base_name(b) in classes and _has_method(classes[_base_name(b)], classes, method, seen)
               for b in cls.bases)


def static_check(code: str) -> dict:
    """Check strategy source without running it.

    Returns ``{"errors": [problem, ...], "warnings": [problem, ...]}``. An empty
    ``errors`` list means the file is worth dry-running.
    """
    errors: List[dict] = []
    warnings: List[dict] = []

    if not isinstance(code, str) or not code.strip():
        return {"errors": [_problem("static", "the strategy is empty")], "warnings": []}
    if len(code.encode("utf-8")) > MAX_CODE_BYTES:
        return {"errors": [_problem(
            "static", f"the strategy is larger than {MAX_CODE_BYTES // 1024} KB")], "warnings": []}

    try:
        tree = ast.parse(code, filename="<strategy>")
    except SyntaxError as exc:
        return {"errors": [_problem("syntax", f"{exc.msg}", exc.lineno)], "warnings": []}
    except ValueError as exc:                       # e.g. null bytes in the source
        return {"errors": [_problem("syntax", str(exc))], "warnings": []}

    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    cls = classes.get("UserStrategy")
    if cls is None:
        errors.append(_problem(
            "static", "the file must define `class UserStrategy(Strategy)` at the top level "
                      "(a comment or string that mentions it does not count)"))
        return {"errors": errors, "warnings": warnings}

    if not _derives_from_strategy(cls, classes):
        errors.append(_problem(
            "static", "UserStrategy must subclass Strategy: `class UserStrategy(Strategy)`",
            cls.lineno))
    elif "Strategy" in {_base_name(b) for b in cls.bases} and \
            "Strategy" not in _defined_names(tree) and \
            not any(isinstance(b, ast.Attribute) for b in cls.bases):
        errors.append(_problem(
            "static", "`Strategy` is not imported; add `from runner.sdk import Strategy` "
                      "at the top of the file", cls.lineno))

    if not _has_method(cls, classes, "on_tick"):
        errors.append(_problem(
            "static", "UserStrategy defines no `on_tick(self, state)`, so it can never place an order",
            cls.lineno))

    for node in ast.walk(tree):
        mods = []
        if isinstance(node, ast.Import):
            mods = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods = [node.module.split(".")[0]]
        for mod in mods:
            if mod in _NOT_INSTALLED:
                warnings.append(_problem(
                    "static", f"`{mod}` is not installed in the sandbox; only the Python standard "
                              "library is available", node.lineno))
            elif mod in _SANDBOX_HOSTILE:
                warnings.append(_problem(
                    "static", f"`{mod}` is mostly unavailable inside the sandbox "
                              "(no network except the engine, no host files)", node.lineno))

    return {"errors": errors, "warnings": warnings}


# --------------------------------------------------------------------------- #
# Sample ticks for the dry run                                                 #
# --------------------------------------------------------------------------- #
_PRICE_FIELDS = ("last_price", "best_bid", "best_ask", "mid_price", "micro_price")


def _shifted(tick: dict, delta: float, **overrides) -> dict:
    """Copy of ``tick`` with every price moved by ``delta`` and fields overridden."""
    out = {k: (list(v) if isinstance(v, list) else dict(v) if isinstance(v, dict) else v)
           for k, v in tick.items()}
    for key in _PRICE_FIELDS:
        if isinstance(out.get(key), (int, float)):
            out[key] = round(out[key] + delta, 2)
    if isinstance(out.get("bar"), dict):
        for key in ("o", "h", "l", "c"):
            if isinstance(out["bar"].get(key), (int, float)):
                out["bar"][key] = round(out["bar"][key] + delta, 2)
    for side in ("lob_bids", "lob_asks"):
        out[side] = [[round(p + delta, 2), q] for p, q in out.get(side, [])]
    out.update(overrides)
    return out


def build_sample_ticks() -> List[dict]:
    """Three tick snapshots -- calm, dip, spike -- from the real engine.

    The dip and spike move price 1% from VWAP with an extreme RSI and order
    flow, so a typical RSI/VWAP strategy takes its buy and sell branches during
    the dry run instead of returning ``[]`` three times.
    """
    try:
        from engine.matching_engine import MatchingEngine
        engine = MatchingEngine()
        engine.init_sim("TCS", "TECH", 190.0, seed=1)
        payload = None
        for _ in range(3):
            for topic, data in engine.physics_tick():
                if topic == "tick":
                    payload = data
        base = payload
    except Exception:                      # engine package missing (pure-SDK host)
        base = None
    if base is None:                       # minimal fallback with the documented fields
        base = {"seq": 3, "mode": "LIVE", "symbol": "TCS", "sector": "TECH",
                "last_price": 190.0, "best_bid": 189.95, "best_ask": 190.05,
                "mid_price": 190.0, "micro_price": 190.0, "spread": 0.1,
                "vwap": 190.0, "rsi": 50.0, "ofi": 0.0, "volatility": 0.001,
                "volume_ma": 1000.0, "step_volume": 800, "tick_size": 0.05,
                "unix_time": 1767258900, "day_count": 1, "market_minute": 3,
                "regime": "RANGING", "paused": False,
                "bar": {"minute": 3, "o": 190.0, "h": 190.1, "l": 189.9, "c": 190.0, "v": 800},
                "lob_bids": [[189.95, 500]], "lob_asks": [[190.05, 500]], "leaderboard": []}
    vwap = float(base.get("vwap") or base["last_price"])
    last = float(base["last_price"])
    return [
        dict(base, scenario="calm"),
        dict(_shifted(base, round(vwap * 0.99 - last, 2), rsi=25.0, ofi=-0.6), scenario="dip"),
        dict(_shifted(base, round(vwap * 1.01 - last, 2), rsi=75.0, ofi=0.6), scenario="spike"),
    ]
