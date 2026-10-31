# Writing a Chronos strategy: one-page guide

A strategy is a Python class with one important method, `on_tick`. Every second the engine gives it the market state and you return the orders you want. It trades as a normal participant: it moves the price, and the AI agents react.

## Template

```python
from runner.sdk import Strategy

class UserStrategy(Strategy):               # the class MUST be called UserStrategy
    def on_start(self, config):              # once, before the first tick
        self.pos = 0

    def on_tick(self, state):                # once per second; return a LIST of orders
        if state["rsi"] < 30 and self.pos < 500:
            return [{"action": "BUY", "qty": 25}]            # market buy
        if state["rsi"] > 70 and self.pos > 0:
            return [{"action": "SELL", "type": "LIMIT",      # limit sell
                     "price": state["best_ask"] or state["last_price"],
                     "qty": self.pos, "tif_ticks": 20}]
        return []                                            # do nothing this tick

    def on_fill(self, ack):                  # after each FILLED / PARTIAL order
        self.pos = ack["pos"]                # the engine's number is the truth

    def on_stop(self):                       # once, when you press Stop
        pass
```

Start from `runner/examples/momentum.py` (trend following) or `mean_reversion.py` for longer examples.

## What `state` contains (the tick snapshot)

| Field | Meaning |
|---|---|
| `last_price`, `best_bid`, `best_ask`, `mid_price`, `micro_price`, `spread` | prices (best_bid/ask are `None` if that side is empty) |
| `bar` | the minute's candle: `{minute, o, h, l, c, v}` |
| `vwap`, `rsi`, `ofi`, `volatility`, `volume_ma`, `step_volume` | indicators from real trades (RSI is 50 until 15 bars exist; OFI is -1 to +1) |
| `lob_bids`, `lob_asks` | top 10 levels, `[[price, qty], ...]`, best first |
| `mode` | `REPLAY` (history sets the price) or `REACTIVE` (the AI agents do) |
| `tick_size`, `symbol`, `sector`, `regime`, `unix_time`, `day_count`, `market_minute`, `leaderboard` | context |

## Market history: `self.context`

`state` is only the current tick. For what the market has already done, use `self.context`, which the SDK updates just before every `on_tick`. It reports history only; it never forecasts, and nothing waits on it (your orders are never blocked).

```python
def on_tick(self, state):
    ctx = self.context
    if not (ctx and ctx.ready):            # needs ~120 bars; until then, trade as normal
        return []
    prev = ctx.prev_day                    # None on the first day
    if prev and state["last_price"] > prev["h"]:
        return [{"action": "BUY", "qty": 10}]   # broke yesterday's high
    return []
```

| Member | Meaning |
|---|---|
| `ready`, `bars_seen` | `ready` is True once 120 bars have been seen |
| `day_open`, `day_high`, `day_low`, `day_close` | today so far (`None` before the first bar) |
| `prev_day` | yesterday as `{o, h, l, c, v}`, or `None` on the first day |
| `days` | the last 5 completed days, newest last (today not included) |
| `swing_highs()`, `swing_lows()` | confirmed pivot prices, newest last. A bar is a pivot if it beats the 5 bars before and the 5 bars after it, so it shows up 5 bars late (it never uses the future) |
| `ret(minutes)` | return over the last `minutes` bars, `None` if there are not enough |
| `vol_ratio()` | today's realised volatility vs the recent daily norm, `None` if unknown |

In a replay, one bar is one tick (a 15-minute file gives one 15-minute bar per tick), and the "day" changes every 375 ticks.

## Returning orders

- Each order is a dict. `{"action": "BUY", "qty": 25}` is a **market** order. Add `"type": "LIMIT"` and a `"price"` for a limit order.
- `action` is `BUY` or `SELL`. `qty` must be a whole number above 0. Prices are snapped to `tick_size`.
- Optional: `"tif_ticks"` (how many ticks a resting limit order lives; default 60, `0` = until the day ends) and `"client_order_id"` (your own label).
- **At most 5 orders per tick** are sent; extras are dropped. A bad order is dropped with a log line, so use **Check** to see why.
- A large order can be split: return a slice each tick and keep your own count in `self`. The strategy keeps running until you stop it.

## The hand-over from history to the AI agents

Your strategy trades as `STRAT_xxxxxxxx`. In a **replay**, history sets the price until your **first fill**. Then the engine switches to `REACTIVE` for good: history stops and the AI agents set the price, so you see what your trading would have changed. A resting order that has not traded does not trigger it.

## Rules of the sandbox

Strategies run in a sealed container: **no internet** (only the engine), **no files** outside `/tmp`, **512 MB** memory, **1 CPU**, 64 processes. Only the **Python standard library** is available (no numpy or pandas). Keep `on_tick` fast: ticks arrive every second.

## Check before you run

On the **Strategies** page press **Check**. It reads the code (the class, the base class, `on_tick`), then runs `on_start`, `on_tick` on three sample ticks (calm, a dip, a spike), `on_fill` and `on_stop` in the sandbox, and shows every problem with its **line number**. **Run** repeats the same check, so a broken strategy never starts.

From a terminal (development only): `python -m runner.sdk my_strategy.py --name my_strategy`.
