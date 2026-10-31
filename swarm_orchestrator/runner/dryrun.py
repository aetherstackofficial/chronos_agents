"""Strategy pre-run checks, part 2: the dry run.

Loads a strategy file, builds ``UserStrategy`` and calls its hooks the way the
SDK does -- ``on_start`` once, ``on_tick`` with a few sample ticks, ``on_fill``
with a fake fill, ``on_stop`` -- then reports every problem with the exact line.

The live loop (runner/sdk.py) is deliberately forgiving: an exception in
``on_tick`` is logged and the tick skipped, and an invalid order is dropped with
a log line. That keeps a running strategy alive, but it also means a typo shows
up as a strategy that silently does nothing. The dry run is where those same
problems become visible errors *before* the run starts.

**This file executes untrusted code.** Never call :func:`run_dry` in the bridge
process. runner/sandbox.py runs it inside the sealed sandbox (a container with
no network, no capabilities and a memory limit) and enforces the wall-clock
timeout from outside, so an infinite loop cannot hang anything.

Usage (inside the sandbox)::

    python -m runner.dryrun STRATEGY.py TICKS.json

prints one line ``@@DRYRUN@@{json}`` so output from the user's own ``print``
calls cannot be mistaken for the result.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import time
import traceback
from typing import Any, List, Optional

from core.market_context import MarketContext
from runner.check import DRYRUN_SENTINEL as SENTINEL
from runner.sdk import MAX_ORDERS_PER_TICK, Strategy, check_order
SLOW_CALL_S = 0.5            # ticks arrive once a second; a hook this slow is a problem
MAX_CAPTURED_OUTPUT = 2000   # characters of the strategy's own prints to return


def _where(exc: BaseException, path: str) -> tuple:
    """(line number, source text) of the innermost frame inside the user's file."""
    line, text = None, None
    for frame in traceback.extract_tb(exc.__traceback__):
        if frame.filename == path:
            line, text = frame.lineno, (frame.line or "").strip() or None
    return line, text


def _error(stage: str, exc: BaseException, path: str, scenario: Optional[str] = None) -> dict:
    line, text = _where(exc, path)
    out = {"stage": stage, "line": line, "message": f"{type(exc).__name__}: {exc}".strip()}
    if text:
        out["code"] = text
    if scenario:
        out["scenario"] = scenario
    return out


def _merge(problems: List[dict]) -> List[dict]:
    """Collapse the same problem found in several sample ticks into one entry.

    A typo in ``on_tick`` fails on all three ticks; the user should see it once,
    with the ticks listed, not three times.
    """
    merged: dict = {}
    for item in problems:
        key = (item["stage"], item.get("line"), item["message"])
        if key not in merged:
            merged[key] = dict(item, scenarios=[])
        if item.get("scenario"):
            merged[key]["scenarios"].append(item["scenario"])
    out = []
    for item in merged.values():
        scenarios = item.pop("scenarios")
        item.pop("scenario", None)
        if scenarios:
            noun = "ticks" if len(scenarios) > 1 else "tick"
            item["message"] += f"  (sample {noun}: {', '.join(scenarios)})"
        out.append(item)
    return out


def _fake_fill(order: dict, price: float) -> dict:
    """An engine FILLED ack for ``order`` (shape of PROTOCOL 4.2)."""
    signed = order["qty"] if order["action"] == "BUY" else -order["qty"]
    return {"msg": "ACK", "status": "FILLED", "order_id": 1, "seq": 1,
            "executed_qty": order["qty"], "average_price": price, "resting_qty": 0,
            "cash": round(1_000_000 - signed * price, 2), "pos": signed,
            "client_order_id": "dry-run-1"}


def run_dry(path: str, ticks: List[dict]) -> dict:
    """Run the strategy file at ``path`` against ``ticks``; return the report.

    Report: ``{"ok", "errors", "warnings", "orders", "ticks_tested", "output"}``.
    Each error is ``{"stage", "line", "message", "code"?}``.
    """
    errors: List[dict] = []
    warnings: List[dict] = []
    orders_seen: List[dict] = []
    sink = io.StringIO()

    def finish() -> dict:
        merged_errors, merged_warnings = _merge(errors), _merge(warnings)
        return {"ok": not merged_errors, "errors": merged_errors, "warnings": merged_warnings,
                "orders": orders_seen[:5], "ticks_tested": len(ticks),
                "output": sink.getvalue()[:MAX_CAPTURED_OUTPUT]}

    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        # ---- import: runs the file's top-level code ------------------------
        try:
            spec = importlib.util.spec_from_file_location("chronos_user_strategy", path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
        except BaseException as exc:      # noqa: BLE001 - SystemExit from sys.exit() too
            errors.append(_error("import", exc, path))
            return finish()

        cls = getattr(module, "UserStrategy", None)
        if not (isinstance(cls, type) and issubclass(cls, Strategy)):
            errors.append({"stage": "import", "line": None,
                           "message": "UserStrategy must subclass Strategy "
                                      "(from runner.sdk import Strategy)"})
            return finish()

        # ---- build ----------------------------------------------------------
        try:
            strategy = cls()
        except BaseException as exc:      # noqa: BLE001
            errors.append(_error("init", exc, path))
            return finish()

        def call(stage: str, fn, *args, scenario: Optional[str] = None) -> Any:
            """Run one hook; record an error (and return None) if it raises or is slow."""
            started = time.perf_counter()
            try:
                result = fn(*args)
            except BaseException as exc:  # noqa: BLE001
                errors.append(_error(stage, exc, path, scenario))
                return None
            took = time.perf_counter() - started
            if took > SLOW_CALL_S:
                item = {"stage": stage, "line": None,
                        "message": f"{stage} took {took:.1f}s; ticks arrive every second, "
                                   "so keep it fast"}
                if scenario:
                    item["scenario"] = scenario
                warnings.append(item)
            return result

        context = strategy.context = MarketContext()      # as the SDK runner attaches it

        call("on_start", strategy.on_start,
             {"agent_id": "STRAT_dryrun", "name": "dry-run", "host": "dry-run",
              "order_port": 0, "data_port": 0})

        # ---- on_tick with each sample tick ----------------------------------
        first_valid: Optional[dict] = None
        for tick in ticks:
            scenario = tick.get("scenario", "tick")
            context.update(tick)
            raw = call("on_tick", strategy.on_tick, dict(tick), scenario=scenario)
            if raw is None or raw == []:
                continue
            if not isinstance(raw, (list, tuple)):
                errors.append({"stage": "on_tick", "line": None, "scenario": scenario,
                               "message": "on_tick must return a list of order dicts, "
                                          f"got {type(raw).__name__}"})
                continue
            valid = 0
            for i, order in enumerate(raw, start=1):
                clean, reason = check_order(order, float(tick.get("tick_size") or 0.05))
                if clean is None:
                    errors.append({"stage": "on_tick", "line": None, "scenario": scenario,
                                   "message": f"order {i} {order!r} is invalid: {reason}"})
                    continue
                valid += 1
                orders_seen.append(clean)
                first_valid = first_valid or dict(clean, _price=tick.get("last_price", 100.0))
            if valid > MAX_ORDERS_PER_TICK:
                warnings.append({"stage": "on_tick", "line": None, "scenario": scenario,
                                 "message": f"returned {valid} orders; only the first "
                                            f"{MAX_ORDERS_PER_TICK} are sent each tick"})

        # ---- on_fill / on_stop ------------------------------------------------
        if first_valid is not None:
            price = first_valid.pop("_price")
            call("on_fill", strategy.on_fill, _fake_fill(first_valid, float(price)))
        call("on_stop", strategy.on_stop)

    return finish()


def main(argv: Optional[list] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print("usage: python -m runner.dryrun STRATEGY.py TICKS.json", file=sys.stderr)
        return 2
    path, ticks_path = argv
    with open(ticks_path, encoding="utf-8") as fh:
        ticks = json.load(fh)
    report = run_dry(path, ticks)
    sys.__stdout__.write(SENTINEL + json.dumps(report) + "\n")
    sys.__stdout__.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
