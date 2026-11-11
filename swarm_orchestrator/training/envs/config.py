"""Load training/configs/env.yaml (with optional nested overrides)."""
from __future__ import annotations

import copy
import os
from typing import Optional

import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_ENV_YAML = os.path.join(_ROOT, "training", "configs", "env.yaml")
THRESHOLDS_YAML = os.path.join(_ROOT, "validation", "config", "thresholds.yaml")


def _merge(base: dict, over: dict) -> dict:
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def thresholds_intraday_rt_bps() -> float:
    with open(THRESHOLDS_YAML, encoding="utf-8") as fh:
        return float(yaml.safe_load(fh)["a8_no_free_lunch"]["cost_bps_zero_brokerage"])


def load_env_config(path: Optional[str] = None, overrides: Optional[dict] = None) -> dict:
    """The env config as a plain dict. ``costs.intraday_rt_bps: null`` is
    resolved from validation/config/thresholds.yaml."""
    with open(path or DEFAULT_ENV_YAML, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if overrides:
        _merge(cfg, overrides)
    if cfg["costs"].get("intraday_rt_bps") is None:
        cfg["costs"]["intraday_rt_bps"] = thresholds_intraday_rt_bps()
    return cfg


def resolve_config(cfg) -> dict:
    """None -> default file; str -> that file; dict -> used as given (merged onto the default)."""
    if cfg is None:
        return load_env_config()
    if isinstance(cfg, str):
        return load_env_config(cfg)
    return load_env_config(overrides=cfg)
