"""MatchingEngine integration tests — the No-Dummy invariants (PROTOCOL §8, §12.1).

These are the tests that guard the core Phase-2 promise: price is produced ONLY by
real trades, the ledger is conserved, and runs are deterministic under a fixed seed.
"""
import pytest

from engine.matching_engine import MatchingEngine


def make_engine(seed=42):
    e = MatchingEngine()
    e.init_sim("TCS", "TECH", 190.0, seed=seed)
    return e


def order(agent, action, otype, qty, price=None):
    msg = {"msg": "ORDER", "agent_id": agent, "action": action, "type": otype, "qty": qty}
    if price is not None:
        msg["price"] = price
    return msg


# --- No-Dummy core -------------------------------------------------------------

def test_price_changes_only_from_real_trades():
    """§8.1: last_price is the last TRADE price and nothing else moves it."""
    e = make_engine()
    start = e.last_price
    # Physics ticks with an empty book must NOT drift the price (no random.gauss).
    for _ in range(20):
        e.physics_tick()
    # SIM_LP + ghost flow may trade, so price MAY move — but only via fills.
    # Assert last_price always equals a real executed trade price (never a random drift).
    if e.trades:
        assert any(e.last_price == pytest.approx(t["price"]) for t in e.trades)
    else:
        assert e.last_price == pytest.approx(start)


def test_market_buy_prints_at_book_price():
    e = make_engine()
    e.handle_message(order("MM_1", "SELL", "LIMIT", 100, 190.10))
    ack, pubs = e.handle_message(order("WHALE_1", "BUY", "MARKET", 60))
    assert ack["status"] in ("FILLED", "PARTIAL")
    assert ack["executed_qty"] == 60
    assert ack["average_price"] == pytest.approx(190.10)
    assert e.last_price == pytest.approx(190.10)  # the trade set the price
    trade_pubs = [p for t, p in pubs if t == "trade"]
    assert trade_pubs and trade_pubs[0]["price"] == pytest.approx(190.10)


def test_ledger_conservation_invariant():
    """§8.9: total cash and total position are invariant across any fill.

    Measured across a SECOND trade so both agents are already seeded — otherwise the
    whale's lazy first-sight seeding (its +5M capital) legitimately changes the sum.
    """
    e = make_engine()
    e.handle_message(order("MM_1", "SELL", "LIMIT", 500, 190.10))
    e.handle_message(order("WHALE_1", "BUY", "MARKET", 100))  # seeds WHALE_1
    cash_before = sum(a["cash"] for a in e.agent_ledger.values())
    pos_before = sum(a["pos"] for a in e.agent_ledger.values())
    e.handle_message(order("WHALE_1", "BUY", "MARKET", 200))  # both already exist
    cash_after = sum(a["cash"] for a in e.agent_ledger.values())
    pos_after = sum(a["pos"] for a in e.agent_ledger.values())
    assert cash_after == pytest.approx(cash_before)
    assert pos_after == pos_before


def test_ack_reports_authoritative_ledger():
    e = make_engine()
    e.handle_message(order("MM_1", "SELL", "LIMIT", 100, 190.10))
    ack, _ = e.handle_message(order("WHALE_1", "BUY", "MARKET", 60))
    assert ack["pos"] == 60
    assert ack["cash"] == pytest.approx(5_000_000.0 - 60 * 190.10)


def test_reject_bad_order():
    e = make_engine()
    ack, _ = e.handle_message(order("MM_1", "BUY", "LIMIT", 0, 190.0))
    assert ack["status"] == "REJECTED"
    ack, _ = e.handle_message({"msg": "ORDER", "agent_id": "MM_1", "action": "BUY",
                               "type": "LIMIT", "qty": 10})  # missing price
    assert ack["status"] == "REJECTED"


def test_unknown_message_rejected():
    e = make_engine()
    ack, _ = e.handle_message({"msg": "NONSENSE"})
    assert ack["status"] == "REJECTED"


# --- Determinism ---------------------------------------------------------------

def test_same_seed_same_run():
    """§8.7: identical seed + identical inputs ⇒ identical price path."""
    def run(seed):
        e = MatchingEngine()
        e.init_sim("TCS", "TECH", 190.0, seed=seed)
        prices = []
        for _ in range(60):
            pubs = e.physics_tick()
            tick = [p for t, p in pubs if t == "tick"][0]
            prices.append(tick["last_price"])
        return prices

    assert run(123) == run(123)


def test_different_seed_diverges():
    def run(seed):
        e = MatchingEngine()
        e.init_sim("TCS", "TECH", 190.0, seed=seed)
        for _ in range(60):
            e.physics_tick()
        return e.last_price
    # Overwhelmingly likely to differ; if identical the RNG isn't wired to price flow.
    assert run(1) != run(999)


# --- Tick payload contract (§5.1) ---------------------------------------------

REQUIRED_TICK_FIELDS = {
    "seq", "run_id", "mode", "paused", "unix_time", "day_count", "market_minute",
    "symbol", "sector", "last_price", "best_bid", "best_ask", "mid_price",
    "micro_price", "spread", "bar", "vwap", "rsi", "ofi", "volatility",
    "volume_ma", "step_volume", "regime", "lob_bids", "lob_asks", "leaderboard",
}


def test_tick_payload_has_all_required_fields():
    e = make_engine()
    e.handle_message(order("MM_1", "SELL", "LIMIT", 100, 190.10))
    e.handle_message(order("WHALE_1", "BUY", "MARKET", 60))
    pubs = e.physics_tick()
    tick = [p for t, p in pubs if t == "tick"][0]
    missing = REQUIRED_TICK_FIELDS - set(tick)
    assert not missing, f"tick payload missing fields: {missing}"
    assert set(tick["bar"]) >= {"minute", "o", "h", "l", "c", "v"}


def test_clock_advances_one_minute_per_tick():
    e = make_engine()
    m0 = e.market_minute
    e.physics_tick()
    assert e.market_minute == m0 + 1


def test_sequence_numbers_monotonic():
    e = make_engine()
    seqs = []
    for _ in range(5):
        pubs = e.physics_tick()
        for _t, p in pubs:
            if "seq" in p:
                seqs.append(p["seq"])
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs)  # strictly increasing, no dupes


def test_pause_freezes_the_clock():
    e = make_engine()
    e.handle_message({"msg": "PAUSE"})
    m0 = e.market_minute
    pubs = e.physics_tick()
    tick = [p for t, p in pubs if t == "tick"][0]
    assert tick["paused"] is True
    assert e.market_minute == m0  # frozen


def test_ping():
    e = make_engine()
    ack, _ = e.handle_message({"msg": "PING"})
    assert ack["status"] == "PONG"
