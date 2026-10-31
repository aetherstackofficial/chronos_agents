import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s[SWARM] - %(message)s")
log = logging.getLogger(__name__)

from agents.live_agents import LiveMarketMaker, LiveRetailAgent, LiveWhaleAgent
from agents.red_team import RedTeamAgent
import argparse, asyncio, os, random, sys
import numpy as np
from dotenv import load_dotenv
from core.engine_client import EngineClient
from core.loop_manager import SwarmOrchestrator

DEFAULT_COUNTS = {"mm": 15, "whale": 4, "retail": 80, "redteam": 3}
# Only retail reads the run seed so far; the other roles ignore it.
_BUILDERS = {"mm": lambda i, seed: LiveMarketMaker(f"MM_{i}"), "whale": lambda i, seed: LiveWhaleAgent(f"WHALE_{i}"), "retail": lambda i, seed: LiveRetailAgent(f"RETAIL_{i}", seed=seed), "redteam": lambda i, seed: RedTeamAgent(f"REDTEAM_{i}")}

def build_agents(role: str, count: int | None, seed: int | None = None) -> list:
    if role == "all":
        if count is not None: log.warning("--count is ignored with --role all (per-role defaults apply).")
        return [agent for sub in ("mm", "whale", "retail") for agent in (_BUILDERS[sub](i, seed) for i in range(DEFAULT_COUNTS[sub]))]
    return [_BUILDERS[role](i, seed) for i in range(count if count is not None else DEFAULT_COUNTS[role])]

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chronos swarm worker (push-driven, two-plane).")
    p.add_argument("--role", choices=["mm", "whale", "retail", "redteam", "all"], default="all")
    p.add_argument("--count", type=int, default=None)
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--data-port", type=int, default=None)
    p.add_argument("--seed", type=int, default=None, help="Run seed for the whole swarm (personalities, wake-ups, random draws). Omit for a fresh random one, which is logged so the run can be repeated.")
    return p.parse_args(argv)

async def run_worker(args: argparse.Namespace) -> None:
    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2 ** 31)
    # Seed AFTER the imports above: loading the PPO models touches global RNG state.
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    log.info(f"Swarm seed = {seed} (pass --seed {seed} to repeat this run).")
    client = EngineClient(prefix := f"SWARM_{args.role.upper()}_{os.getpid()}", host=args.host, order_port=args.port, data_port=args.data_port)
    (orch := SwarmOrchestrator(client, seed=seed)).load_agents(agents := build_agents(args.role, args.count, seed))
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