"""Strategy pre-run checks: static analysis, the dry run, and the guide's template.

These run everything in-process (no Docker): runner.check reads code with ast,
and runner.dryrun.run_dry is a plain function. The sealed-sandbox wiring is in
test_sandbox_policy.py.
"""
import os
import re

import pytest

from runner.check import MAX_CODE_BYTES, build_sample_ticks, static_check
from runner.dryrun import run_dry
from runner.sdk import check_order

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUIDE = os.path.join(os.path.dirname(ROOT), "docs", "STRATEGY_GUIDE.md")

HEADER = "from runner.sdk import Strategy\n"


def _dry(tmp_path, code):
    path = tmp_path / "strategy.py"
    path.write_text(code, encoding="utf-8")
    return run_dry(str(path), build_sample_ticks())


# --- static_check ------------------------------------------------------------
@pytest.mark.parametrize("code,needle,line", [
    ("", "empty", None),
    ("# class UserStrategy(Strategy)\nx = 1\n", "top level", None),          # comment is not a class
    ("s = 'class UserStrategy(Strategy)'\n", "top level", None),            # nor is a string
    ("class UserStrategy(Strategy):\n    def on_tick(self, s):\n        return []\n",
     "not imported", 1),
    ("class UserStrategy(object):\n    def on_tick(self, s):\n        return []\n",
     "must subclass Strategy", 1),
    (HEADER + "class UserStrategy(Strategy):\n    pass\n", "no `on_tick", 2),
    (HEADER + "class UserStrategy(Strategy):\n    def on_tick(self, s)\n        return []\n",
     "expected ':'", 3),
])
def test_static_errors(code, needle, line):
    errors = static_check(code)["errors"]
    assert errors and needle in errors[0]["message"]
    assert errors[0]["line"] == line


def test_static_accepts_inherited_on_tick_and_attribute_base():
    code = (HEADER + "class Base(Strategy):\n    def on_tick(self, s):\n        return []\n"
            "class UserStrategy(Base):\n    pass\n")
    assert static_check(code)["errors"] == []
    attr = "import runner.sdk as sdk\nclass UserStrategy(sdk.Strategy):\n    def on_tick(self, s):\n        return []\n"
    assert static_check(attr)["errors"] == []


def test_static_rejects_oversized_code():
    errors = static_check("x = 1\n" * (MAX_CODE_BYTES // 3))["errors"]
    assert errors and "larger than" in errors[0]["message"]


def test_static_warns_about_modules_missing_from_the_sandbox():
    code = HEADER + "import numpy as np\nimport socket\nclass UserStrategy(Strategy):\n    def on_tick(self, s):\n        return []\n"
    result = static_check(code)
    assert result["errors"] == []
    messages = " ".join(w["message"] for w in result["warnings"])
    assert "numpy" in messages and "not installed" in messages and "socket" in messages


# --- order rules (shared with the live SDK) ------------------------------------
def test_check_order_gives_reasons():
    assert check_order({"action": "BUY", "qty": 25})[0] == {"action": "BUY", "type": "MARKET", "qty": 25}
    assert "qty" in check_order({"action": "BUY", "qty": 0})[1]
    assert "action" in check_order({"side": "BUY", "qty": 1})[1]
    assert "price" in check_order({"action": "SELL", "type": "LIMIT", "qty": 1})[1]
    assert "dict" in check_order("BUY")[1]


# --- sample ticks -----------------------------------------------------------------
def test_sample_ticks_cover_calm_dip_and_spike():
    calm, dip, spike = build_sample_ticks()
    assert (calm["scenario"], dip["scenario"], spike["scenario"]) == ("calm", "dip", "spike")
    assert dip["last_price"] < calm["last_price"] < spike["last_price"]
    assert dip["rsi"] < 30 < 70 < spike["rsi"]
    assert dip["best_bid"] < calm["best_bid"]          # the book moved with the price


def test_sample_tick_has_every_field_the_guide_documents():
    tick = build_sample_ticks()[0]
    guide = open(GUIDE, encoding="utf-8").read()
    for field in ("last_price", "best_bid", "best_ask", "mid_price", "micro_price", "spread", "bar",
                  "vwap", "rsi", "ofi", "volatility", "volume_ma", "step_volume", "lob_bids",
                  "lob_asks", "mode", "tick_size", "symbol", "sector", "regime", "unix_time",
                  "day_count", "market_minute", "leaderboard"):
        assert field in tick, f"engine tick lost {field!r}"
        assert f"`{field}`" in guide, f"guide does not mention {field!r}"


# --- dry run --------------------------------------------------------------------------
@pytest.mark.parametrize("example", ["momentum", "mean_reversion"])
def test_example_strategies_pass(tmp_path, example):
    code = open(os.path.join(ROOT, "runner", "examples", f"{example}.py"), encoding="utf-8").read()
    assert static_check(code)["errors"] == []
    report = _dry(tmp_path, code)
    assert report["ok"], report["errors"]


def test_guide_template_passes_and_trades(tmp_path):
    guide = open(GUIDE, encoding="utf-8").read()
    template = re.search(r"```python\n(.*?)```", guide, re.S).group(1)
    assert static_check(template)["errors"] == []
    report = _dry(tmp_path, template)
    assert report["ok"], report["errors"]
    assert report["orders"], "the template should place an order on the dip tick"


def test_runtime_error_reports_line_and_code_once(tmp_path):
    code = HEADER + "class UserStrategy(Strategy):\n    def on_tick(self, state):\n        px = state['lastprice']\n        return []\n"
    report = _dry(tmp_path, code)
    assert not report["ok"] and len(report["errors"]) == 1          # three ticks, one entry
    error = report["errors"][0]
    assert error["stage"] == "on_tick" and error["line"] == 4
    assert "KeyError" in error["message"] and "calm, dip, spike" in error["message"]
    assert error["code"] == "px = state['lastprice']"


def test_invalid_orders_and_wrong_return_type_are_errors(tmp_path):
    bad_order = HEADER + "class UserStrategy(Strategy):\n    def on_tick(self, s):\n        return [{'action': 'BUY', 'qty': 0}]\n"
    messages = [e["message"] for e in _dry(tmp_path, bad_order)["errors"]]
    assert any("order 1" in m and "qty" in m for m in messages)
    not_list = HEADER + "class UserStrategy(Strategy):\n    def on_tick(self, s):\n        return {'action': 'BUY', 'qty': 5}\n"
    assert "must return a list" in _dry(tmp_path, not_list)["errors"][0]["message"]


def test_import_failure_and_init_failure(tmp_path):
    missing = "import notamodule\n" + HEADER + "class UserStrategy(Strategy):\n    def on_tick(self, s):\n        return []\n"
    error = _dry(tmp_path, missing)["errors"][0]
    assert error["stage"] == "import" and error["line"] == 1 and "notamodule" in error["message"]
    broken_init = HEADER + "class UserStrategy(Strategy):\n    def __init__(self):\n        raise ValueError('boom')\n    def on_tick(self, s):\n        return []\n"
    assert _dry(tmp_path, broken_init)["errors"][0]["stage"] == "init"


def test_hooks_are_exercised_and_prints_are_captured(tmp_path):
    code = (HEADER + "class UserStrategy(Strategy):\n"
            "    def on_start(self, config):\n        self.filled = 0\n"
            "    def on_tick(self, state):\n"
            "        print('tick', state['scenario'])\n"
            "        return [{'action': 'BUY', 'qty': 3}] if state['rsi'] < 30 else []\n"
            "    def on_fill(self, ack):\n        self.filled = ack['pos']\n"
            "        raise RuntimeError('on_fill blew up')\n")
    report = _dry(tmp_path, code)
    assert [e["stage"] for e in report["errors"]] == ["on_fill"]     # reached on_fill after the dip order
    assert "tick dip" in report["output"]                            # their print() is returned, not lost


def test_too_many_orders_is_a_warning_not_an_error(tmp_path):
    code = HEADER + "class UserStrategy(Strategy):\n    def on_tick(self, s):\n        return [{'action': 'BUY', 'qty': 1}] * 8\n"
    report = _dry(tmp_path, code)
    assert report["ok"] and any("only the first 5" in w["message"] for w in report["warnings"])
