import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s[SWARM] - %(message)s")
log = logging.getLogger(__name__)

from agents.live_agents import LiveMarketMaker, LiveRetailAgent, LiveWhaleAgent
from agents.red_team import RedTeamAgent
import argparse, asyncio, os, sys
from dotenv import load_dotenv
from core.engine_client import EngineClient
from core.loop_manager import SwarmOrchestrator

DEFAULT_COUNTS = {"mm": 15, "whale": 4, "retail": 80, "redteam": 3}
_BUILDERS = {"mm": lambda i: LiveMarketMaker(f"MM_{i}"), "whale": lambda i: LiveWhaleAgent(f"WHALE_{i}"), "retail": lambda i: LiveRetailAgent(f"RETAIL_{i}"), "redteam": lambda i: RedTeamAgent(f"REDTEAM_{i}")}

def build_agents(role: str, count: int | None) -> list:
    if role == "all":
        if count is not None: log.warning("--count is ignored with --role all (per-role defaults apply).")
        return [agent for sub in ("mm", "whale", "retail") for agent in (_BUILDERS[sub](i) for i in range(DEFAULT_COUNTS[sub]))]
    return [_BUILDERS[role](i) for i in range(count if count is not None else DEFAULT_COUNTS[role])]

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chronos swarm worker (push-driven, two-plane).")
    p.add_argument("--role", choices=["mm", "whale", "retail", "redteam", "all"], default="all")
    p.add_argument("--count", type=int, default=None)
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--data-port", type=int, default=None)
    return p.parse_args(argv)

async def run_worker(args: argparse.Namespace) -> None:
    client = EngineClient(prefix := f"SWARM_{args.role.upper()}_{os.getpid()}", host=args.host, order_port=args.port, data_port=args.data_port)
    (orch := SwarmOrchestrator(client)).load_agents(agents := build_agents(args.role, args.count))
    log.info(f"Worker '{prefix}' starting with {len(agents)} agents (role={args.role}).")
    try: await orch.run()
    except asyncio.CancelledError: pass
    finally:
        log.info("Worker shutting down...")
        await orch.shutdown()

def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    if sys.platform == "win32": asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try: asyncio.run(run_worker(parse_args(argv)))
    except KeyboardInterrupt: log.info("Interrupted — worker exited.")

if __name__ == "__main__": main()