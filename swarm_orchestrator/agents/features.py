"""Single source of truth for live observation vectors.

Both the training environments (`agents/base_environment.py::_get_obs`) and the
live agents build the SAME 6-feature layout. Live agents MUST go through
`build_observation` — never hand-roll features — so training and live inference
can never drift apart.

Every input field is a REAL engine value from the `tick` snapshot
(PROTOCOL.md §5.1). There are no defaults masking missing data: a missing tick
field raises KeyError by design. The only guards are the two divide-by-zero
guards the training env itself uses (vwap <= 0, volume_ma <= 0).
"""

import numpy as np


def build_observation(tick: dict, inventory: int, max_capacity: float) -> np.ndarray:
    """Build the (1, 6) float32 observation the PPO/heuristic brains expect.

    Layout (identical to UniversalMarketEnv._get_obs):
        [vwap_dist, volatility, volume_ratio, rsi_norm, ofi, pos_norm]

    Args:
        tick: full engine tick snapshot (PROTOCOL.md §5.1).
        inventory: the agent's current signed position (shares).
        max_capacity: the agent's position-normalisation denominator (> 0).

    Returns:
        np.ndarray of shape (1, 6), dtype float32, NaN/inf scrubbed.
    """
    last_price = float(tick["last_price"])
    vwap = float(tick["vwap"])
    volume_ma = float(tick["volume_ma"])

    vwap_dist = (last_price - vwap) / vwap if vwap > 0 else 0.0
    volatility = float(tick["volatility"])
    volume_ratio = float(tick["step_volume"]) / volume_ma if volume_ma > 0 else 1.0
    rsi_norm = float(tick["rsi"]) / 100.0
    ofi = float(tick["ofi"])
    pos_norm = float(inventory) / float(max_capacity)

    obs = np.array(
        [[vwap_dist, volatility, volume_ratio, rsi_norm, ofi, pos_norm]],
        dtype=np.float32,
    )
    return np.nan_to_num(obs)
