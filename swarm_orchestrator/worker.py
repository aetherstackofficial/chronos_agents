import logging
import random
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s[SWARM] - %(message)s")
log = logging.getLogger(__name__)

# The models (torch) must load before zmq: on Windows the other order fails (WinError 1114).
import agents.live_agents  # noqa: F401
import argparse, asyncio, os, sys
from dotenv import load_dotenv
from agents.swarm_factory import DEFAULT_COUNTS, TickClock, build_swarm
from core.engine_client import EngineClient
from core.loop_manager import SwarmOrchestrator

ALL_ROLES = ("mm", "whale", "retail")   # --role all; red team runs as its own worker

def build_agents(role: str, count: int | None, seed: int | None = None, clock: TickClock | None = None) -> list:
    clock = clock or TickClock()
    if role == "all":
        if count is not None: log.warning("--count is ignored with --role all (per-role defaults apply).")
        return build_swarm({r: DEFAULT_COUNTS[r] for r in ALL_ROLES}, seed, clock)
    return build_swarm({role: count if count is not None else DEFAULT_COUNTS[role]}, seed, clock)

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chronos swarm worker (push-driven, two-plane).")
    p.add_argument("--role", choices=["mm", "whale", "retail", "redteam", "all"], default="all")
    p.add_argument("--count", type=int, default=None)
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--data-port", type=int, default=None)
    p.add_argument("--seed", type=int, default=int(s) if (s := os.getenv("CHRONOS_SEED")) else None,
                   help="seeds the orchestrator and every agent (env CHRONOS_SEED); omit for an unseeded swarm")
    return p.parse_args(argv)

async def run_worker(args: argparse.Namespace) -> None:
    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2 ** 31)
    # Seed AFTER the imports above: loading the PPO models touches global RNG state.
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    log.info(f"Swarm seed = {seed} (pass --seed {seed} to repeat this run).")
    client = EngineClient(prefix := f"SWARM_{args.role.upper()}_{os.getpid()}", host=args.host, order_port=args.port, data_port=args.data_port)
    clock = TickClock()
    (orch := SwarmOrchestrator(client, seed=args.seed, clock=clock)).load_agents(agents := build_agents(args.role, args.count, args.seed, clock))
    log.info(f"Worker '{prefix}' starting with {len(agents)} agents (role={args.role}, seed={args.seed}).")
    if args.seed is None: log.warning("No --seed: the swarm is unseeded, so runs are not reproducible.")
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
