"""Parameterized swarm worker: run any slice of the agent swarm as one process.

Usage:
    python worker.py --role mm --count 15
    python worker.py --role whale --count 4
    python worker.py --role retail --count 80
    python worker.py --role all                # full default swarm

Replaces the Phase-1 run_mms.py / run_whales.py / run_retail.py trio (flaw R3).
"""

import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - [SWARM] - %(message)s")

# --- CRITICAL WINDOWS FIX: IMPORT AI/TORCH FIRST -------------------------
# torch's DLLs must load before pyzmq's on Windows, or the process crashes
# with a WinError 127 DLL-ordering failure (Phase-1 finding). Keep this
# import ABOVE every zmq/asyncio networking import.
from agents.live_agents import (  # noqa: E402
    LiveMarketMaker,
    LiveRetailAgent,
    LiveWhaleAgent,
)
# Heuristic (non-torch) predatory agent — safe to import after the torch block.
from agents.red_team import RedTeamAgent  # noqa: E402

# --- IMPORT NETWORKING SECOND ---------------------------------------------
import argparse  # noqa: E402
import asyncio  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

from dotenv import load_dotenv  # noqa: E402

from core.engine_client import EngineClient  # noqa: E402
from core.loop_manager import SwarmOrchestrator  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_COUNTS = {"mm": 15, "whale": 4, "retail": 80, "redteam": 3}

# 'redteam' is opt-in only (launched from the UI Lab / --role redteam); it is
# deliberately excluded from --role all.
_BUILDERS = {
    "mm": lambda i: LiveMarketMaker(f"MM_{i}"),
    "whale": lambda i: LiveWhaleAgent(f"WHALE_{i}"),
    "retail": lambda i: LiveRetailAgent(f"RETAIL_{i}"),
    "redteam": lambda i: RedTeamAgent(f"REDTEAM_{i}"),
}


def build_agents(role: str, count: "int | None") -> list:
    """Instantiate the agents for one role, or the full swarm for 'all'.

    For --role all the per-role defaults are always used (a single --count
    cannot meaningfully apply to three roles at once).
    """
    if role == "all":
        if count is not None:
            logger.warning("--count is ignored with --role all "
                           "(per-role defaults apply).")
        agents: list = []
        for sub_role in ("mm", "whale", "retail"):
            agents.extend(_BUILDERS[sub_role](i)
                          for i in range(DEFAULT_COUNTS[sub_role]))
        return agents
    n = count if count is not None else DEFAULT_COUNTS[role]
    return [_BUILDERS[role](i) for i in range(n)]


def parse_args(argv: "list[str] | None" = None) -> argparse.Namespace:
    """Parse worker CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Chronos swarm worker (push-driven, two-plane).")
    parser.add_argument("--role",
                        choices=["mm", "whale", "retail", "redteam", "all"],
                        default="all", help="which agent population to run")
    parser.add_argument("--count", type=int, default=None,
                        help="number of agents (defaults: mm 15, whale 4, "
                             "retail 80; ignored for --role all)")
    parser.add_argument("--host", default=None,
                        help="engine host (default: ZMQ_HOST or 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None,
                        help="engine ROUTER order port (default: "
                             "ZMQ_ORDER_PORT or 5555)")
    parser.add_argument("--data-port", type=int, default=None,
                        help="engine PUB data port (default: "
                             "ZMQ_DATA_PORT or 5556)")
    return parser.parse_args(argv)


async def run_worker(args: argparse.Namespace) -> None:
    """Build the agents and drive one SwarmOrchestrator until cancelled."""
    agents = build_agents(args.role, args.count)
    identity_prefix = f"SWARM_{args.role.upper()}_{os.getpid()}"
    client = EngineClient(identity_prefix, host=args.host,
                          order_port=args.port, data_port=args.data_port)
    orchestrator = SwarmOrchestrator(client)
    orchestrator.load_agents(agents)
    logger.info("Worker '%s' starting with %d agents (role=%s).",
                identity_prefix, len(agents), args.role)
    try:
        await orchestrator.run()
    except asyncio.CancelledError:
        pass
    finally:
        logger.info("Worker shutting down...")
        await orchestrator.shutdown()


def main(argv: "list[str] | None" = None) -> None:
    """Entrypoint: env config, Windows event-loop policy, asyncio.run."""
    load_dotenv()
    if sys.platform == "win32":
        # Required before any zmq.asyncio loop is created (PROTOCOL.md §11).
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(run_worker(parse_args(argv)))
    except KeyboardInterrupt:
        logger.info("Interrupted — worker exited.")


if __name__ == "__main__":
    main()
