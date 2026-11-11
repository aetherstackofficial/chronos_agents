"""Realism suite (validation/runner.py) with the v2 models in the swarm.

    python -m tools.run_realism_v2 --mode smoke [--workers 5] [--tag v2]
    CHRONOS_V2_WHALES=whale_swing,whale_position python -m tools.run_realism_v2 --mode full

Identical to ``python -m validation.runner`` except that ``agents.live_agents_v2.install()`` runs at
import time, in this process and in every spawned worker (the worker imports this module as
``__mp_main__``), so whales and market makers are the v2 agents. The report card is written by
validation.runner as usual; this script then copies it to ``report_card_<mode>_<instrument>_<tag>_<stamp>.json``
with the v2 model names recorded, so old-swarm and v2 cards sit side by side.
"""
import glob
import json
import os
import sys
import time

from agents import live_agents_v2

live_agents_v2.install()

if __name__ == "__main__":
    from validation import runner
    tag = "v2"
    argv = sys.argv[1:]
    if "--tag" in argv:
        i = argv.index("--tag")
        tag = argv[i + 1]
        del argv[i:i + 2]
    t0 = time.time()
    runner.main(argv)
    cards = [p for p in glob.glob(os.path.join(runner.REPORT_DIR, "report_card_*.json")) if os.path.getmtime(p) >= t0]
    for p in cards:
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
        d["swarm_models"] = {"whales": os.environ.get("CHRONOS_V2_WHALES", "whale_swing").split(","),
                             "mm": "mm", "roles": os.environ.get("CHRONOS_V2_ROLES", "whale,mm").split(","),
                             "dir": "agents/ml_models/v2"}
        # every option of agents/live_agents_v2.py and agents/latent_liquidity.py used in this run
        d["swarm_env"] = {k: v for k, v in sorted(os.environ.items()) if k.startswith("CHRONOS_")}
        base = os.path.basename(p)[len("report_card_"):]
        mode, inst, stamp = base.split("_", 2)
        out = os.path.join(runner.REPORT_DIR, f"report_card_{mode}_{inst}_{tag}_{stamp}")
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(d, fh, indent=1)
        os.remove(p)
        print("v2 report:", out)
