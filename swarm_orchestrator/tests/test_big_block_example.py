"""runner/examples/big_block.py (the Strategies page's default): fed ticks by hand, no engine."""
import importlib.util
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "runner" / "examples" / "big_block.py"


def strategy():
    spec = importlib.util.spec_from_file_location("big_block_example", PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    s = mod.UserStrategy()
    s.on_start({"agent_id": "STRAT_test"})
    return s


def tick(run, minute, px=99.0, vwap=100.0):
    return {"run_id": run, "unix_time": 1_760_000_000 + 60 * minute, "last_price": px, "vwap": vwap, "paused": False}


def test_a_thin_book_ends_the_block_after_max_slices():
    s = strategy()
    sent = []
    for m in range(80):
        orders = s.on_tick(tick("A", m))
        sent += orders
        for o in orders:          # every slice fills only 25 shares, as in a 1-share-per-level book
            s.on_fill({"status": "PARTIAL", "executed_qty": 25, "average_price": 99.0,
                       "pos": s.pos + (25 if o["action"] == "BUY" else -25)})
    buys = [o for o in sent if o["action"] == "BUY"]
    assert len(buys) == s.MAX_SLICES                     # not one 10,000 order every minute for ever
    assert s.phase in ("HOLD", "EXIT", "REST", "WAIT")


def test_a_new_run_starts_over():
    s = strategy()
    for m in range(12):
        s.on_tick(tick("A", m))
    assert s.phase == "ENTER"
    s.pos = 500
    orders = [o for m in range(12, 12 + s.WARMUP - 1) for o in s.on_tick(tick("B", m))]
    assert orders == [] and s.phase == "WAIT" and s.pos == 0   # no trade in the new run's first minutes
