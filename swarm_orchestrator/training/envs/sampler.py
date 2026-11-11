"""Episode sampler for ReplayBookEnv: one stock, one date range, one set of
randomised conditions per episode. Only ONE stock's parquet is loaded per draw
(``load_stock`` keeps just the last one)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from training.data.episode_bars import DEFAULT_PARQUET_DIR, SESSION_BARS, bars_for_sessions, load_stock
from training.data.sectors import canonical_sector
from training.data.splits import HELD_OUT_STOCKS, SPLIT_NAMES, is_held_out, valid_episode_starts
from training.envs.config import resolve_config

MODEL_NAMES = ("whale_swing", "whale_position", "mm")


@dataclass
class Episode:
    symbol: str
    sector: str
    split: str
    start_session: int           # first traded session
    length: int                  # traded sessions
    memory_sessions: int         # M: sessions pre-rolled into the memory (before the warm-up)
    memory_first_session: int    # session_id of the oldest memory session (== warmup if M = 0)
    warmup_session: Optional[int]  # session right before the start (None if there is none)
    bars: List[dict] = field(repr=False, default_factory=list)  # warm-up + episode, engine replay format
    adv20: float = 0.0
    capacity: int = 0            # max |position| in shares
    max_quote_size: int = 0      # mm only
    capital: float = 0.0
    start_price: float = 0.0
    risk_tolerance: float = 0.08
    cost_mult: float = 1.0
    order_delay: int = 0
    obs_noise: float = 0.01
    lp_size_mult: float = 1.0
    engine_seed: int = 0

    @property
    def warmup_bars(self) -> int:
        return SESSION_BARS if self.warmup_session is not None else 0


def available_symbols(parquet_dir: str = DEFAULT_PARQUET_DIR) -> List[str]:
    return sorted(f[:-8] for f in os.listdir(parquet_dir) if f.endswith(".parquet"))


def preroll_arrays(df, first_session: int, n_sessions: int):
    """(day_index, minute, o, h, l, c, v) python lists for sessions
    first_session .. first_session+n_sessions-1, ``day_index`` = session_id."""
    sid = df["session_id"].to_numpy()
    lo = int(np.searchsorted(sid, first_session, side="left"))
    hi = int(np.searchsorted(sid, first_session + n_sessions, side="left"))
    p = df.iloc[lo:hi]
    return (p["session_id"].tolist(), p["minute"].tolist(), p["open"].tolist(), p["high"].tolist(),
            p["low"].tolist(), p["close"].tolist(), p["volume"].tolist())


def _loguniform(rng, lo: float, hi: float) -> float:
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


class EpisodeSampler:
    def __init__(self, model_name: str, split: str, cfg=None, seed: int = 0,
                 parquet_dir: str = DEFAULT_PARQUET_DIR) -> None:
        if model_name not in MODEL_NAMES:
            raise ValueError(f"model_name must be one of {MODEL_NAMES}")
        if split not in SPLIT_NAMES:
            raise ValueError(f"split must be one of {SPLIT_NAMES}")
        self.model_name = model_name
        self.split = split
        self.cfg = resolve_config(cfg)
        self.parquet_dir = parquet_dir
        self.rng = np.random.default_rng(seed)      # used only when draw() gets no generator
        syms = available_symbols(parquet_dir)
        if split == "train":
            syms = [s for s in syms if not is_held_out(s)]
        if not syms:
            raise ValueError("no stocks available for this split")
        self.symbols = syms
        self._starts_cache: dict = {}

    # ------------------------------------------------------------------ #
    def _starts(self, df, symbol: str, length: int) -> np.ndarray:
        key = (symbol, length)
        s = self._starts_cache.get(key)
        if s is None:
            s = valid_episode_starts(df, self.split, length)
            s = s[s >= 1]                      # needs a warm-up session and some ADV history
            self._starts_cache[key] = s
        return s

    def _draw_memory(self, rng) -> int:
        mix = self.cfg["sampler"]["memory_mix"]
        p = np.array([m[0] for m in mix], dtype=float)
        k = int(rng.choice(len(mix), p=p / p.sum()))
        lo, hi = int(mix[k][1]), int(mix[k][2])
        return int(rng.integers(lo, hi + 1))

    def draw(self, np_random: Optional[np.random.Generator] = None) -> Episode:
        rng = np_random if np_random is not None else self.rng
        sc = self.cfg["sampler"]
        lo, hi = sc["length_sessions"][self.model_name]
        while True:
            symbol = self.symbols[int(rng.integers(len(self.symbols)))]
            df = load_stock(symbol, self.parquet_dir)
            length = int(rng.integers(int(lo), int(hi) + 1))
            starts = self._starts(df, symbol, length)
            while starts.size == 0 and length > 1:      # split too short for this length: shrink
                length -= 1
                starts = self._starts(df, symbol, length)
            if starts.size > 0:
                break
            # a stock listed after this split ended (e.g. JIOFIN for 'train') has no episode here:
            # drop it from this sampler and draw again
            if len(self.symbols) <= 1:
                raise ValueError(f"{symbol}: no valid episode start in split {self.split!r}")
            self.symbols = [s for s in self.symbols if s != symbol]
        start = int(starts[int(rng.integers(starts.size))])

        warm = start - 1
        M = min(self._draw_memory(rng), warm)       # sessions 0..warm-1 exist before the warm-up
        mem_first = warm - M

        bars = bars_for_sessions(df, warm, 1 + length)
        # ADV of the (up to) adv_sessions sessions before the start
        sid = df["session_id"].to_numpy()
        n_adv = int(sc["adv_sessions"])
        a = int(np.searchsorted(sid, max(start - n_adv, 0), side="left"))
        b = int(np.searchsorted(sid, start, side="left"))
        n_have = start - max(start - n_adv, 0)
        adv = float(df["volume"].to_numpy()[a:b].sum()) / max(n_have, 1)
        price0 = float(bars[SESSION_BARS]["o"])      # open of the first traded session

        episode = Episode(
            symbol=symbol, sector=canonical_sector(df["sector"].iloc[0]),
            split=self.split, start_session=start, length=length, memory_sessions=M,
            memory_first_session=mem_first, warmup_session=warm, bars=bars, adv20=adv,
            start_price=price0)

        if self.model_name == "mm":
            frac = _loguniform(rng, *sc["mm_capacity_adv_frac"])
            episode.capacity = max(int(adv * frac), 5)
            episode.max_quote_size = max(episode.capacity // int(sc["mm_quote_divisor"]), 1)
            episode.capital = float(sc["mm_capital_multiple"]) * episode.capacity * price0
        else:
            frac = _loguniform(rng, *sc["whale_capacity_adv_frac"])
            episode.capacity = max(int(adv * frac), 1)
            episode.capital = float(episode.capacity) * price0
        rlo, rhi = sc["risk_tolerance_range"]
        episode.risk_tolerance = float(rng.uniform(rlo, rhi)) if self.model_name != "mm" else 0.08
        clo, chi = self.cfg["costs"]["cost_mult_range"]
        episode.cost_mult = float(rng.uniform(clo, chi))
        episode.order_delay = int(rng.choice(sc["order_delay_choices"]))
        episode.obs_noise = float(self.cfg["env"]["obs_noise"])
        llo, lhi = sc["lp_size_range"]
        episode.lp_size_mult = float(rng.uniform(llo, lhi))
        episode.engine_seed = int(rng.integers(1, 2 ** 31 - 1))
        return episode
