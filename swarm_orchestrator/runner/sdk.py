"""Chronos strategy SDK — the user-facing runtime (PROTOCOL.md section 10).

Users subclass :class:`Strategy` and hand it to :func:`run_strategy`, which wires
the strategy to the engine over ZeroMQ:

* **SUB** socket on the market-data plane (topics ``tick`` and ``event``) — every
  1 Hz ``tick`` snapshot (PROTOCOL section 5.1) is passed verbatim to
  ``Strategy.on_tick``.
* **DEALER** socket on the order-entry plane — orders returned from ``on_tick``
  are validated, stamped with ``msg``/``agent_id`` and sent; the engine's ack
  (section 4.2) is awaited and, when it reports an execution (``FILLED`` /
  ``PARTIAL``), forwarded to ``Strategy.on_fill``.

The SDK is deliberately **synchronous** (plain pyzmq, no asyncio): the simplest
possible user runtime.  It is a *standalone thin client* — it never imports from
``core/``, ``engine/`` or ``agents/`` and only needs ``pyzmq`` (plus, optionally,
``python-dotenv``) at runtime.

Agent identity
--------------
``run_strategy`` defaults ``agent_id`` to ``"STRAT_" + 8 hex chars``.  The
``STRAT_`` prefix matters: per PROTOCOL section 9, the *first fill* involving an
agent whose id starts with ``STRAT_`` (or ``USER_``) flips a replaying engine
from ``REPLAY`` to ``REACTIVE`` mode permanently — i.e. running a strategy is
what hands price formation over from the historical tape to the live book.

CLI
---
``python -m runner.sdk path/to/strategy.py --name my_strat [--agent-id ID]``
runs a file that defines ``class UserStrategy(Strategy)``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import signal
import sys
import time
import uuid
from types import FrameType
from typing import Any, Optional, Type

import zmq

try:  # optional convenience — present in the full stack, not required for the SDK
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - pure-SDK environments
    pass

log = logging.getLogger("runner.sdk")
if not log.handlers:  # dedicated handler so the [SDK] tag survives co-imports
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s - [SDK] - %(message)s"))
    log.addHandler(_handler)
    log.setLevel(logging.INFO)
    log.propagate = False

# --------------------------------------------------------------------------- #
# Constants (venue + runtime policy)                                           #
# --------------------------------------------------------------------------- #
TICK_SIZE: float = 0.05  # PROTOCOL section 3 — all LIMIT prices snap to this grid
MAX_ORDERS_PER_TICK: int = 5  # SDK throttle (PROTOCOL section 10)
ACK_TIMEOUT_S: float = 5.0  # per-order wait for the engine ack
IDLE_TIMEOUT_S: float = 30.0  # exit cleanly if no tick arrives for this long
VALID_ACTIONS = frozenset({"BUY", "SELL"})
VALID_TYPES = frozenset({"MARKET", "LIMIT"})


# --------------------------------------------------------------------------- #
# Strategy base class                                                          #
# --------------------------------------------------------------------------- #
class Strategy:
    """Base class for user strategies. Override the hooks you need.

    Lifecycle (all hooks are optional overrides):

    * ``on_start(config)`` — once, before the first tick. ``config`` carries the
      resolved runtime settings (``agent_id``, ``name``, host/ports).
    * ``on_tick(state)``   — once per engine physics tick. ``state`` is the
      PROTOCOL section 5.1 payload verbatim. Return a list of order dicts
      (section 4.1 minus ``msg``/``agent_id`` — the SDK injects those).
      Shorthand ``{"action": "BUY", "qty": 100}`` is accepted as a MARKET order.
    * ``on_fill(ack)``     — for every ack whose status is FILLED or PARTIAL.
      ``ack`` is the section 4.2 payload (``ack["pos"]``/``ack["cash"]`` are the
      authoritative post-trade ledger).
    * ``on_stop()``        — once, on shutdown (after CANCEL_ALL was sent).
    """

    def on_start(self, config: dict) -> None:
        """Called once before the tick loop starts."""

    def on_tick(self, state: dict) -> list[dict]:
        """Called once per market tick; return a list of orders (may be empty)."""
        return []

    def on_fill(self, ack: dict) -> None:
        """Called for every FILLED/PARTIAL ack addressed to this strategy."""

    def on_stop(self) -> None:
        """Called once on graceful shutdown."""


# --------------------------------------------------------------------------- #
# Order validation                                                             #
# --------------------------------------------------------------------------- #
def _round_to_tick(price: float) -> float:
    """Snap ``price`` to the venue tick grid (0.05)."""
    return round(round(price / TICK_SIZE) * TICK_SIZE, 2)


def validate_order(raw: Any) -> Optional[dict]:
    """Normalise and validate one user-returned order.

    Returns the cleaned order dict (still without ``msg``/``agent_id``) or
    ``None`` if the order is invalid.  Invalid orders are logged and dropped —
    a bad order must never crash the strategy loop.
    """
    if not isinstance(raw, dict):
        log.warning("Dropping order: expected dict, got %r", type(raw).__name__)
        return None

    action = raw.get("action")
    if action not in VALID_ACTIONS:
        log.warning("Dropping order: invalid action %r (need BUY/SELL)", action)
        return None

    # Shorthand {"action","qty"} means a MARKET order.
    order_type = raw.get("type", "MARKET")
    if order_type not in VALID_TYPES:
        log.warning("Dropping order: invalid type %r (need MARKET/LIMIT)", order_type)
        return None

    qty = raw.get("qty")
    if isinstance(qty, bool) or not isinstance(qty, int) or qty <= 0:
        log.warning("Dropping order: qty must be int > 0, got %r", qty)
        return None

    order: dict = {"action": action, "type": order_type, "qty": qty}

    if order_type == "LIMIT":
        price = raw.get("price")
        if not isinstance(price, (int, float)) or isinstance(price, bool) or price <= 0:
            log.warning("Dropping LIMIT order: price must be float > 0, got %r", price)
            return None
        order["price"] = _round_to_tick(float(price))

    tif = raw.get("tif_ticks")
    if tif is not None:
        if isinstance(tif, bool) or not isinstance(tif, int) or tif < 0:
            log.warning("Dropping order: tif_ticks must be int >= 0, got %r", tif)
            return None
        order["tif_ticks"] = tif

    client_order_id = raw.get("client_order_id")
    if client_order_id is not None:
        order["client_order_id"] = str(client_order_id)

    meta = raw.get("meta")
    if isinstance(meta, dict):
        order["meta"] = meta

    return order


# --------------------------------------------------------------------------- #
# Runtime                                                                      #
# --------------------------------------------------------------------------- #
class _StrategyRunner:
    """Owns the sockets and the blocking tick loop for one strategy instance."""

    def __init__(
        self,
        strategy: Strategy,
        name: str,
        agent_id: str,
        host: str,
        order_port: int,
        data_port: int,
    ) -> None:
        self.strategy = strategy
        self.name = name
        self.agent_id = agent_id
        self.host = host
        self.order_port = order_port
        self.data_port = data_port
        self._stop_requested = False
        self._order_counter = 0

        self.ctx = zmq.Context.instance()
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.connect(f"tcp://{host}:{data_port}")
        self.sub.setsockopt(zmq.SUBSCRIBE, b"tick")
        self.sub.setsockopt(zmq.SUBSCRIBE, b"event")

        self.dealer = self.ctx.socket(zmq.DEALER)
        self.dealer.setsockopt(zmq.IDENTITY, agent_id.encode())
        self.dealer.setsockopt(zmq.LINGER, 1000)
        self.dealer.connect(f"tcp://{host}:{order_port}")

    # -- signal handling ---------------------------------------------------- #
    def request_stop(self, signum: int, frame: Optional[FrameType]) -> None:
        """Signal handler: flag the loop to exit at the next opportunity."""
        log.info("Received signal %s — shutting down gracefully", signum)
        self._stop_requested = True

    # -- order plumbing ----------------------------------------------------- #
    def _send_order(self, order: dict) -> Optional[dict]:
        """Send one order and block (<= ACK_TIMEOUT_S) for its ack."""
        self._order_counter += 1
        order.setdefault("client_order_id", f"{self.agent_id}-{self._order_counter}")
        wire = dict(order, msg="ORDER", agent_id=self.agent_id)
        self.dealer.send_json(wire)
        return self._await_ack(order["client_order_id"])

    def _await_ack(self, client_order_id: Optional[str]) -> Optional[dict]:
        """Wait for the ack matching ``client_order_id`` (or the next ack)."""
        deadline = time.monotonic() + ACK_TIMEOUT_S
        while time.monotonic() < deadline:
            remaining_ms = max(1, int((deadline - time.monotonic()) * 1000))
            if not self.dealer.poll(remaining_ms, zmq.POLLIN):
                break
            try:
                frames = self.dealer.recv_multipart()
                ack = json.loads(frames[-1])
            except (ValueError, IndexError, zmq.ZMQError) as exc:
                log.warning("Malformed ack frame: %s", exc)
                continue
            # Correlate on client_order_id when the engine echoes it; a stale
            # ack from an earlier timed-out order is simply skipped.
            if client_order_id and ack.get("client_order_id") not in (None, client_order_id):
                log.debug("Skipping stale ack for %s", ack.get("client_order_id"))
                continue
            return ack
        log.warning("No ack within %.0fs for order %s", ACK_TIMEOUT_S, client_order_id)
        return None

    def _cancel_all(self) -> None:
        """Best-effort CANCEL_ALL on shutdown (pulls our resting quotes)."""
        try:
            self.dealer.send_json({"msg": "CANCEL_ALL", "agent_id": self.agent_id})
            if self.dealer.poll(2000, zmq.POLLIN):
                self.dealer.recv_multipart()  # drain the ack; content irrelevant here
            log.info("CANCEL_ALL sent for %s", self.agent_id)
        except zmq.ZMQError as exc:
            log.warning("CANCEL_ALL failed: %s", exc)

    # -- per-tick processing ------------------------------------------------ #
    def _handle_tick(self, state: dict) -> None:
        """Run ``on_tick``, validate/throttle the returned orders, send them."""
        try:
            raw_orders = self.strategy.on_tick(state)
        except Exception:
            log.exception("on_tick raised — tick skipped (strategy kept alive)")
            return
        if not raw_orders:
            return
        if not isinstance(raw_orders, (list, tuple)):
            log.warning("on_tick must return a list of dicts, got %r — dropped",
                        type(raw_orders).__name__)
            return

        valid = [o for o in (validate_order(r) for r in raw_orders) if o is not None]
        if len(valid) > MAX_ORDERS_PER_TICK:
            log.warning("Throttling: %d orders returned, sending first %d",
                        len(valid), MAX_ORDERS_PER_TICK)
            valid = valid[:MAX_ORDERS_PER_TICK]

        for order in valid:
            if self._stop_requested:
                return
            ack = self._send_order(order)
            if ack is None:
                continue
            status = ack.get("status")
            log.info("Ack %s: %s x%s -> %s (exec %s @ %s)",
                     ack.get("order_id"), order["action"], order["qty"], status,
                     ack.get("executed_qty"), ack.get("average_price"))
            if status in ("FILLED", "PARTIAL"):
                try:
                    self.strategy.on_fill(ack)
                except Exception:
                    log.exception("on_fill raised — ignored (strategy kept alive)")

    # -- main loop ----------------------------------------------------------- #
    def run(self) -> None:
        """Blocking tick loop; returns after graceful shutdown."""
        config = {
            "agent_id": self.agent_id,
            "name": self.name,
            "host": self.host,
            "order_port": self.order_port,
            "data_port": self.data_port,
        }
        try:
            self.strategy.on_start(config)
        except Exception:
            log.exception("on_start raised — continuing into the tick loop")

        log.info("Strategy %r running as %s (data tcp://%s:%s, orders tcp://%s:%s)",
                 self.name, self.agent_id, self.host, self.data_port,
                 self.host, self.order_port)

        poller = zmq.Poller()
        poller.register(self.sub, zmq.POLLIN)
        last_tick_time = time.monotonic()

        try:
            while not self._stop_requested:
                if time.monotonic() - last_tick_time > IDLE_TIMEOUT_S:
                    log.warning(
                        "No tick received for %.0fs — engine appears to be down; "
                        "exiting cleanly", IDLE_TIMEOUT_S)
                    break
                events = dict(poller.poll(1000))
                if self.sub not in events:
                    continue
                try:
                    topic, payload = self.sub.recv_multipart()
                    data = json.loads(payload)
                except (ValueError, zmq.ZMQError) as exc:
                    log.warning("Malformed market-data frame: %s", exc)
                    continue
                if topic == b"tick":
                    last_tick_time = time.monotonic()
                    self._handle_tick(data)
                elif topic == b"event" and data.get("kind") == "mode_change":
                    log.info("Engine mode change -> %s (trigger: %s)",
                             data.get("mode"), data.get("trigger_agent"))
        except KeyboardInterrupt:
            log.info("KeyboardInterrupt — shutting down gracefully")
        finally:
            self._shutdown()

    def _shutdown(self) -> None:
        """CANCEL_ALL, ``on_stop`` hook, socket close."""
        self._cancel_all()
        try:
            self.strategy.on_stop()
        except Exception:
            log.exception("on_stop raised — ignored")
        self.sub.close(0)
        self.dealer.close()
        log.info("Strategy %s stopped", self.agent_id)


def run_strategy(
    strategy_cls: Type[Strategy],
    name: Optional[str] = None,
    agent_id: Optional[str] = None,
    host: Optional[str] = None,
    order_port: Optional[int] = None,
    data_port: Optional[int] = None,
) -> None:
    """Instantiate ``strategy_cls`` and run it against the engine. **Blocking.**

    Args:
        strategy_cls: a :class:`Strategy` subclass (the class, not an instance).
        name: human-readable strategy name (defaults to the class name).
        agent_id: engine identity. Defaults to ``"STRAT_" + 8 hex`` — the
            ``STRAT_`` prefix is what flips a replaying engine into REACTIVE
            mode on this strategy's first fill (PROTOCOL section 9).
        host / order_port / data_port: engine endpoint overrides; fall back to
            ``ZMQ_HOST`` / ``ZMQ_ORDER_PORT`` / ``ZMQ_DATA_PORT`` env vars, then
            to ``127.0.0.1:5555/5556``.

    Returns when the strategy shuts down (SIGINT/SIGTERM, KeyboardInterrupt, or
    ``IDLE_TIMEOUT_S`` seconds without a tick).
    """
    resolved_host = host or os.environ.get("ZMQ_HOST", "127.0.0.1")
    resolved_order = int(order_port or os.environ.get("ZMQ_ORDER_PORT", "5555"))
    resolved_data = int(data_port or os.environ.get("ZMQ_DATA_PORT", "5556"))
    resolved_agent = agent_id or f"STRAT_{uuid.uuid4().hex[:8]}"
    resolved_name = name or strategy_cls.__name__

    runner = _StrategyRunner(
        strategy=strategy_cls(),
        name=resolved_name,
        agent_id=resolved_agent,
        host=resolved_host,
        order_port=resolved_order,
        data_port=resolved_data,
    )

    # Graceful shutdown on SIGINT/SIGTERM (both settable on Windows; SIGTERM
    # delivery is best-effort there — KeyboardInterrupt covers Ctrl+C).
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, runner.request_stop)
        except (ValueError, OSError):  # not on the main thread, or unsupported
            pass

    runner.run()


# --------------------------------------------------------------------------- #
# CLI: python -m runner.sdk path/to/strategy.py --name X [--agent-id ID]       #
# --------------------------------------------------------------------------- #
def _load_user_strategy(path: str) -> Type[Strategy]:
    """Import ``path`` as a module and return its ``UserStrategy`` class."""
    file_path = os.path.abspath(path)
    if not os.path.isfile(file_path):
        raise SystemExit(f"[SDK] strategy file not found: {file_path}")
    spec = importlib.util.spec_from_file_location("chronos_user_strategy", file_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"[SDK] cannot import strategy file: {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    cls = getattr(module, "UserStrategy", None)
    if cls is None:
        raise SystemExit(
            "[SDK] strategy file must define `class UserStrategy(Strategy)`")

    # When executed via `python -m runner.sdk`, this file runs as `__main__`
    # while the user's file imports `runner.sdk` — two distinct module objects.
    # Check subclassing against the canonically imported Strategy.
    import runner.sdk as _canonical

    if not (isinstance(cls, type) and issubclass(cls, (_canonical.Strategy, Strategy))):
        raise SystemExit("[SDK] UserStrategy must subclass runner.sdk.Strategy")
    return cls


def _main(argv: Optional[list[str]] = None) -> None:
    """CLI entry point (see module docstring)."""
    parser = argparse.ArgumentParser(
        prog="python -m runner.sdk",
        description="Run a Chronos strategy file that defines class UserStrategy(Strategy).",
    )
    parser.add_argument("strategy_file", help="path to the user strategy .py file")
    parser.add_argument("--name", default=None, help="strategy display name")
    parser.add_argument("--agent-id", default=None,
                        help="engine identity (default: STRAT_<8 hex>)")
    parser.add_argument("--host", default=None, help="engine host (default: $ZMQ_HOST)")
    parser.add_argument("--order-port", type=int, default=None,
                        help="order-entry port (default: $ZMQ_ORDER_PORT)")
    parser.add_argument("--data-port", type=int, default=None,
                        help="market-data port (default: $ZMQ_DATA_PORT)")
    args = parser.parse_args(argv)

    strategy_cls = _load_user_strategy(args.strategy_file)
    name = args.name or os.path.splitext(os.path.basename(args.strategy_file))[0]

    # Route through the canonical module so Strategy identity stays consistent.
    import runner.sdk as _canonical

    _canonical.run_strategy(
        strategy_cls,
        name=name,
        agent_id=args.agent_id,
        host=args.host,
        order_port=args.order_port,
        data_port=args.data_port,
    )


if __name__ == "__main__":
    _main()
