"""Live progress of the training runs, for watching in your own terminal.

    cd swarm_orchestrator
    python -m training.progress            (refreshes every 10 s, Ctrl+C to stop)
    python -m training.progress --once     (print once and exit)
    python -m training.progress --round1   (the old Round 1 schedule instead of the final runs)

The final runs (training/runs/final_logs/) are shown when they exist.

Reads only the log files (training/runs/round1_logs/), so it never disturbs
the training. Shows every run of the schedule: done / running / waiting, how
many steps are done, speed, time left, and the latest eval scores.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "runs", "round1_logs")
STATUS = os.path.join(LOG_DIR, "status.txt")

# Same order and steps as training/run_round1.sh
SCHEDULE = [(m, a, s) for m in ("whale_swing", "whale_position", "mm")
            for a, s in (("td3", 1_000_000), ("sac", 1_000_000), ("ppo", 5_000_000))]
# Steps per second for runs that have not started yet (smoke-run speeds)
GUESS_SPS = {"td3": 180, "sac": 130, "ppo": 800}

RE_START = re.compile(r"^(\S+ \S+) start (\S+) \((\d+) steps\)")
RE_END = re.compile(r"^(\S+ \S+) end (\S+) exit=(-?\d+) hours=([\d.]+)")
RE_STEPS = re.compile(r"total_timesteps\s*\|\s*(\d+)")
RE_CLIP = re.compile(r"\[clip @ (\d+)\] worst share ([\d.]+)%")
RE_EVAL = re.compile(r"Eval num_timesteps=(\d+), episode_reward=(-?[\d.]+) \+/- ([\d.]+)")
RE_FINISHED = re.compile(r"finished: (\d+) steps in ([\d.]+) s \(([\d.]+) steps/s")


def _ts(s: str) -> dt.datetime:
    return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def _tail(path: str, nbytes: int = 400_000) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - nbytes))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def _evals(path: str):
    """Eval scores of a run: from <run>/eval/evaluations.npz (exact numbers),
    else from the eval lines of the log (rounded to 2 decimals there)."""
    name = os.path.splitext(os.path.basename(path))[0]
    npz = os.path.join(HERE, "runs", name, "eval", "evaluations.npz")
    try:
        import numpy as np
        with np.load(npz) as d:
            return [(int(t), float(r.mean()), float(r.std())) for t, r in zip(d["timesteps"], d["results"])]
    except Exception:
        pass
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return [(int(a), float(b), float(c)) for a, b, c in RE_EVAL.findall(f.read())]
    except OSError:
        return []


def _hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    return f"{h}h {rem // 60:02d}m"


def _bar(frac: float, width: int = 28) -> str:
    frac = min(1.0, max(0.0, frac))
    n = int(round(frac * width))
    return "[" + "#" * n + "-" * (width - n) + "]"


def snapshot() -> str:
    now = dt.datetime.now()
    starts, ends = {}, {}
    try:
        with open(STATUS, encoding="utf-8") as f:
            for line in f:
                m = RE_START.match(line)
                if m:
                    starts[m.group(2)] = (_ts(m.group(1)), int(m.group(3)))
                m = RE_END.match(line)
                if m:
                    ends[m.group(2)] = (_ts(m.group(1)), int(m.group(3)), float(m.group(4)))
    except OSError:
        return f"No status file yet: {STATUS}"

    out = [f"Chronos Round 1 training  -  {now:%Y-%m-%d %H:%M:%S}", ""]
    out.append(f"{'#':>2}  {'run':<28} {'state':<9} {'progress':<36} {'speed':>9}  {'time':>8}  last eval reward")
    remaining = 0.0
    current = None
    for i, (model, algo, steps) in enumerate(SCHEDULE, 1):
        name = f"r1_{model}_{algo}_s1"
        log = os.path.join(LOG_DIR, f"{name}.log")
        evals = _evals(log) if name in starts else []
        last_eval = f"{evals[-1][1]:+.4f} (@{evals[-1][0] // 1000}k)" if evals else "-"
        if name in ends:
            t_end, rc, hours = ends[name]
            text = _tail(log, 4000)
            fin = RE_FINISHED.search(text)
            sps = float(fin.group(3)) if fin else steps / max(hours * 3600, 1)
            state = "DONE" if rc == 0 else f"FAILED({rc})"
            out.append(f"{i:>2}  {name:<28} {state:<9} {_bar(1.0)} 100%  {sps:7.0f}/s  {hours:6.2f}h  {last_eval}")
        elif name in starts:
            t0, total = starts[name]
            text = _tail(log)
            done = 0
            for rx in (RE_STEPS, RE_CLIP):
                found = rx.findall(text)
                if found:
                    v = found[-1][0] if isinstance(found[-1], tuple) else found[-1]
                    done = max(done, int(v))
            elapsed = (now - t0).total_seconds()
            sps = done / elapsed if elapsed > 0 and done else 0.0
            left = (total - done) / sps if sps else total / GUESS_SPS[algo]
            remaining += left
            frac = done / total
            out.append(f"{i:>2}  {name:<28} {'RUNNING':<9} {_bar(frac)} {frac * 100:3.0f}%  {sps:7.0f}/s  {_hms(elapsed):>8}  {last_eval}")
            current = (name, done, total, elapsed, left, text)
        else:
            remaining += steps / GUESS_SPS[algo]
            out.append(f"{i:>2}  {name:<28} {'waiting':<9} {_bar(0.0)}   0%  {'':>9}  {'':>8}  -")

    out.append("")
    if current:
        name, done, total, elapsed, left, text = current
        out.append(f"Now running : {name}")
        out.append(f"  steps     : {done:,} / {total:,}")
        out.append(f"  running   : {_hms(elapsed)}   this run ends in about {_hms(left)}"
                   f"  (~{(now + dt.timedelta(seconds=left)):%H:%M})")
        clips = RE_CLIP.findall(text)
        if clips:
            out.append(f"  inputs at clip (last 1000 steps): worst {float(clips[-1][1]):.1f}%  (small is good)")
        evals = _evals(os.path.join(LOG_DIR, f"{name}.log"))
        if evals:
            out.append("  eval rewards so far (higher is better): "
                       + ", ".join(f"{s // 1000}k: {r:+.4f}" for s, r, _ in evals))
        else:
            out.append("  eval rewards so far: none yet (first eval at 250k steps)")
    if any(n == "ROUND1_DONE" for n in open(STATUS, encoding="utf-8").read().split()):
        out.append("ROUND 1 FINISHED.")
    else:
        out.append(f"Whole Round 1 left : about {_hms(remaining)}  (finish ~{(now + dt.timedelta(seconds=remaining)):%a %H:%M})")
    try:
        import psutil
        vm = psutil.virtual_memory()
        out.append(f"RAM free           : {vm.available / 2**30:.1f} GB of {vm.total / 2**30:.1f} GB"
                   f"   CPU busy: {psutil.cpu_percent(interval=0.5):.0f}%")
    except Exception:
        pass
    return "\n".join(out)


FINAL_DIR = os.path.join(HERE, "runs", "final_logs")
# name, total env steps (decisions), minutes per decision
FINAL_RUNS = [("final_mm_ppo_s1", 4_000_000, 1),
              ("final_whale_swing_ppo_bc_s1", 600_000, 15),
              ("final_whale_position_ppo_bc_s1", 130_000, 75)]


def final_snapshot() -> str:
    now = dt.datetime.now()
    status = ""
    try:
        with open(os.path.join(FINAL_DIR, "status.txt"), encoding="utf-8") as f:
            status = f.read()
    except OSError:
        pass
    out = [f"Chronos final training  -  {now:%Y-%m-%d %H:%M:%S}", "",
           f"{'run':<32} {'state':<8} {'progress':<36} {'speed':>10}  {'running':>8}  {'left':>8}  last eval reward"]
    for name, total, k in FINAL_RUNS:
        log = os.path.join(FINAL_DIR, f"{name}.log")
        if not os.path.exists(log):
            bc = os.path.join(FINAL_DIR, f"bc_{name.replace('final_', '').replace('_ppo_bc_s1', '')}.log")
            state = "BC" if os.path.exists(bc) else "waiting"
            out.append(f"{name:<32} {state:<8} {_bar(0.0)}   0%")
            continue
        m = re.search(re.escape(name) + r" exit=(-?\d+)", status)
        text = _tail(log)
        done = 0
        for rx in (RE_STEPS, RE_CLIP):
            found = rx.findall(text)
            if found:
                v = found[-1][0] if isinstance(found[-1], tuple) else found[-1]
                done = max(done, int(v))
        elapsed = (now - dt.datetime.fromtimestamp(os.path.getctime(log))).total_seconds()
        if m:
            state = "DONE" if m.group(1) == "0" else f"FAIL({m.group(1)})"
            done = total if m.group(1) == "0" else done
        else:
            state = "RUNNING"
        sps = done / elapsed if elapsed > 0 and done else 0.0
        left = (total - done) / sps if sps and not m else 0.0
        evals = _evals(log)
        last = f"{evals[-1][1]:+.4f} (@{evals[-1][0] // 1000}k)" if evals else "-"
        frac = done / total
        out.append(f"{name:<32} {state:<8} {_bar(frac)} {frac * 100:3.0f}%  {sps * k:6.0f} min/s  "
                   f"{_hms(elapsed):>8}  {_hms(left):>8}  {last}")
    out.append("")
    out.append("speed = market minutes simulated per second (a whale decision spans 15 or 75 minutes)")
    try:
        import psutil
        vm = psutil.virtual_memory()
        out.append(f"RAM free: {vm.available / 2**30:.1f} GB of {vm.total / 2**30:.1f} GB"
                   f"   CPU busy: {psutil.cpu_percent(interval=0.5):.0f}%")
    except Exception:
        pass
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--once", action="store_true", help="print once and exit")
    ap.add_argument("--every", type=float, default=10.0, help="refresh seconds (default 10)")
    ap.add_argument("--round1", action="store_true", help="show the Round 1 schedule")
    args = ap.parse_args()
    global snapshot
    if not args.round1 and os.path.isdir(FINAL_DIR):
        snapshot = final_snapshot
    if args.once:
        print(snapshot())
        return
    try:
        while True:
            text = snapshot()
            os.system("cls" if os.name == "nt" else "clear")
            print(text)
            print("\n(refreshes every %gs, Ctrl+C to stop - training keeps running)" % args.every)
            time.sleep(args.every)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
