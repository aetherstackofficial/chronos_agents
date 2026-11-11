"""Export a trained run for live use.

    python -m training.export --run training/runs/<dir> --name whale_swing [--best] [--overwrite]

Copies ``model.zip``, ``obs_scaler.json`` (the fixed observation scaler, applied identically live
via ``agents.obs_scaler.ObsScaler``; its sha256 is recorded in ``spec.json``), ``vecnormalize.pkl``
(reward statistics only) and the resolved config into ``agents/ml_models/v2/<name>/`` and writes
``spec.json`` there: model name, algo, seed, the
input names in order (``OBS_SPEC``), ``spec_hash``, the outputs in order, frame stack, git commit,
library versions, dataset manifest hash and the evaluation results found in the run.
``--best`` exports the EvalCallback's best model + its normaliser instead of the final ones.

Only ``agents/ml_models/v2/<name>/`` is ever written (the old models next to it are never
touched); an existing non-empty target is refused without ``--overwrite``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional

from training import train as T
from training.evaluate import _jsonable

EXPORT_ROOT = T.ROOT / "agents" / "ml_models" / "v2"      # tests monkeypatch this
OUTPUT_MEANING = {
    "whale": ["target position in [-1, 1] x capacity (entry/exit band in action_decode)",
              "participation: linear [-1, 1] -> [participation_min, participation_max]",
              "aggressiveness: linear [-1, 1] -> [0, 1] (1 = through the opposite touch)"],
    "mm": ["bid offset: linear [-1, 1] -> integer ticks [mm_offset_min, mm_offset_max] behind the best bid",
           "ask offset: same, in front of the best ask",
           "bid size: linear [-1, 1] -> [0, max_quote_size] shares",
           "ask size: same"],
}


def _read_json(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def collect_eval_results(run_dir: Path) -> Dict[str, dict]:
    out = {}
    for p in sorted(run_dir.glob("eval_*.json")):
        d = _read_json(p)
        if d is None:
            continue
        out[p.stem] = {"split": d.get("split"), "episodes": d.get("episodes"), "cost_mult": d.get("cost_mult"),
                       "stocks": d.get("stocks"), "metrics": d.get("metrics"),
                       "checks": [{k: c.get(k) for k in ("name", "passed", "value")} for c in d.get("checks", [])],
                       "checks_failed": d.get("checks_failed")}
    return out


def export_run(run_dir: Path, name: str, best: bool = False, overwrite: bool = False,
               export_root: Optional[Path] = None) -> Path:
    import importlib
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in (".", ".."):
        raise ValueError(f"bad export name {name!r}")
    root = Path(export_root) if export_root is not None else EXPORT_ROOT
    target = (root / name).resolve()
    if root.resolve() not in target.parents:
        raise ValueError(f"{target} is outside {root}")
    run_dir = Path(run_dir).resolve()
    cfg = T.load_yaml(run_dir / "config.yaml")
    meta = _read_json(run_dir / "run_meta.json") or {}
    model_src = run_dir / ("best/best_model.zip" if best else "model.zip")
    vn_src = run_dir / ("best/best_vecnormalize.pkl" if best else "vecnormalize.pkl")
    scaler_src = run_dir / T.SCALER_NAME
    for p in (model_src, vn_src, scaler_src):
        if not p.exists():
            raise FileNotFoundError(p)
    if target.exists() and any(target.iterdir()) and not overwrite:
        raise FileExistsError(f"{target} exists; pass --overwrite to replace its files")
    target.mkdir(parents=True, exist_ok=True)

    from agents.features_v2 import OBS_SPEC, spec_hash
    role = T.role_of(cfg["model"])
    now_hash = spec_hash(role)
    shutil.copy2(model_src, target / "model.zip")
    shutil.copy2(vn_src, target / "vecnormalize.pkl")
    shutil.copy2(scaler_src, target / T.SCALER_NAME)
    from agents.obs_scaler import ObsScaler
    scaler = ObsScaler.load(target / T.SCALER_NAME, role)
    shutil.copy2(run_dir / "config.yaml", target / "config.yaml")
    spec = {
        "name": name, "model": cfg["model"], "role": role, "algo": cfg["algo"], "seed": cfg.get("seed", meta.get("seed")),
        "source_run": str(run_dir.name), "weights": "best" if best else "final",
        "obs_names": list(OBS_SPEC[role]), "n_inputs": len(OBS_SPEC[role]),
        "spec_hash": meta.get("spec_hash", now_hash), "spec_hash_now": now_hash,
        "spec_hash_matches_training": meta.get("spec_hash", now_hash) == now_hash,
        "outputs": list(T.ACTION_NAMES[role]), "output_meaning": OUTPUT_MEANING[role],
        "frame_stack": int(cfg.get("frame_stack", 1)), "clip_obs": scaler.clip,
        "obs_scaler": {"file": T.SCALER_NAME, "sha256": T.sha256_file(target / T.SCALER_NAME),
                       "spec_hash": scaler.meta.get("spec_hash"), "n_samples": scaler.meta.get("n_samples"),
                       "fit_date": scaler.meta.get("fit_date"), "clip": scaler.clip,
                       "obs_normalisation": "fixed per-input scaler; vecnormalize.pkl holds reward statistics only"},
        "gamma": cfg.get("gamma"), "net_arch": cfg.get("net_arch"),
        "git_commit": meta.get("git_commit"), "libraries": meta.get("libraries"),
        "dataset_manifest_sha256": meta.get("dataset_manifest_sha256"),
        "trained_timesteps": meta.get("trained_timesteps"), "train_start": meta.get("start_time"),
        "train_end": meta.get("end_time"),
        "exported_at": dt.datetime.now().isoformat(timespec="seconds"),
        "eval": collect_eval_results(run_dir),
    }
    (target / "spec.json").write_text(json.dumps(_jsonable(spec), indent=2), encoding="utf-8")
    return target


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="training.export", description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--name", required=True, help="folder name under agents/ml_models/v2/")
    ap.add_argument("--best", action="store_true", help="export best/best_model.zip + its normaliser")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    if str(T.ROOT) not in sys.path:
        sys.path.insert(0, str(T.ROOT))
    target = export_run(Path(a.run), a.name, best=a.best, overwrite=a.overwrite)
    print(f"exported to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
