"""Order-book unit tests — price-time priority, self-trade prevention, depletion.

Tests the transport-free OrderBook (PROTOCOL §4 matching semantics, §8 no-dummy core).
"""
import pytest

from engine.book import OrderBook

TICK = 0.05


def _limit(order_id, agent_id, side, price, qty, ts=0):
    return {"order_id": order_id, "agent_id": agent_id, "side": side,
            "price": price, "qty": qty, "ts_seq": ts}


def test_add_limit_rests_when_no_cross():
    book = OrderBook()
    fills = book.add_limit(_limit(1, "MM_1", "SELL", 190.10, 100))
    assert fills == []
    assert book.best_ask() == pytest.approx(190.10)
    assert book.best_bid() is None


def test_market_buy_fills_at_resting_ask():
    book = OrderBook()
    book.add_limit(_limit(1, "MM_1", "SELL", 190.10, 100))
    fills = book.match_market("WHALE_1", "BUY", 60)
    assert sum(f["qty"] for f in fills) == 60
    assert all(f["price"] == pytest.approx(190.10) for f in fills)
    assert fills[0]["maker_id"] == "MM_1"
    assert fills[0]["taker_id"] == "WHALE_1"


def test_price_time_priority_best_price_first():
    book = OrderBook()
    book.add_limit(_limit(1, "MM_1", "SELL", 190.20, 100, ts=1))
    book.add_limit(_limit(2, "MM_2", "SELL", 190.10, 100, ts=2))  # better price, later time
    fills = book.match_market("WHALE_1", "BUY", 50)
    assert fills[0]["price"] == pytest.approx(190.10)  # best price wins over time


def test_time_priority_within_a_level_is_fifo():
    book = OrderBook()
    book.add_limit(_limit(1, "MM_1", "SELL", 190.10, 100, ts=1))
    book.add_limit(_limit(2, "MM_2", "SELL", 190.10, 100, ts=2))
    fills = book.match_market("WHALE_1", "BUY", 100)
    assert fills[0]["maker_id"] == "MM_1"  # earliest at the level fills first


def test_self_trade_prevention_skips_own_resting_order():
    book = OrderBook()
    book.add_limit(_limit(1, "WHALE_1", "SELL", 190.10, 100, ts=1))  # own order
    book.add_limit(_limit(2, "MM_2", "SELL", 190.15, 100, ts=2))
    fills = book.match_market("WHALE_1", "BUY", 50)
    # must skip its own 190.10 and hit MM_2 at 190.15 instead
    assert all(f["maker_id"] != "WHALE_1" for f in fills)
    assert fills[0]["maker_id"] == "MM_2"


def test_market_walks_multiple_levels_and_depletes_depth():
    book = OrderBook()
    book.add_limit(_limit(1, "MM_1", "SELL", 190.10, 40))
    book.add_limit(_limit(2, "MM_2", "SELL", 190.15, 40))
    fills = book.match_market("WHALE_1", "BUY", 70)
    assert sum(f["qty"] for f in fills) == 70
    prices = sorted({f["price"] for f in fills})
    assert prices == pytest.approx([190.10, 190.15])
    # 190.10 fully consumed, 190.15 partially → best ask now 190.15 with 10 left
    bids, asks = book.depth(10)
    assert asks[0][0] == pytest.approx(190.15)
    assert asks[0][1] == 10


def test_marketable_limit_executes_then_rests_remainder():
    book = OrderBook()
    book.add_limit(_limit(1, "MM_1", "SELL", 190.10, 40))
    # aggressive BUY limit at 190.20 for 100: 40 fills @190.10, 60 rests @190.20
    fills = book.add_limit(_limit(2, "WHALE_1", "BUY", 190.20, 100))
    assert sum(f["qty"] for f in fills) == 40
    assert book.best_bid() == pytest.approx(190.20)


def test_cancel_removes_resting_order():
    book = OrderBook()
    book.add_limit(_limit(1, "MM_1", "SELL", 190.10, 100))
    assert book.cancel(1) is True
    assert book.best_ask() is None
    assert book.cancel(1) is False  # already gone


def test_cancel_all_by_agent():
    book = OrderBook()
    book.add_limit(_limit(1, "MM_1", "SELL", 190.10, 100))
    book.add_limit(_limit(2, "MM_1", "SELL", 190.15, 100))
    book.add_limit(_limit(3, "MM_2", "SELL", 190.20, 100))
    removed = book.cancel_all("MM_1")
    assert removed == 2
    assert book.best_ask() == pytest.approx(190.20)


def test_limit_prices_round_to_tick():
    book = OrderBook()
    book.add_limit(_limit(1, "MM_1", "SELL", 190.123, 100))  # -> 190.10 or 190.15
    ask = book.best_ask()
    assert abs(round(ask / TICK) * TICK - ask) < 1e-9
