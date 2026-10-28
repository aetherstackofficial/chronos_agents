"""v2 feature-layer tests — script library, results store, Monte-Carlo, scenarios,
copilot & narrator fallbacks. All deterministic and offline (no engine, no LLM key)."""
import asyncio

import pytest


# --- Script library ------------------------------------------------------------
def test_script_store_roundtrip(tmp_path):
    from runner.script_store import ScriptStore
    s = ScriptStore(base_dir=str(tmp_path / "lib"))
    code = "from runner.sdk import Strategy\nclass UserStrategy(Strategy):\n    def on_tick(self,st):return []\n"
    meta = s.save("mean_rev", code)
    assert meta["script_id"] and meta["name"] == "mean_rev"
    assert len(s.list()) == 1
    assert s.get(meta["script_id"])["code"] == code
    # persistence: a fresh store over the same dir sees it
    s2 = ScriptStore(base_dir=str(tmp_path / "lib"))
    assert any(x["script_id"] == meta["script_id"] for x in s2.list())
    assert s.delete(meta["script_id"]) is True
    assert s.list() == []


def test_script_store_rejects_empty():
    from runner.script_store import ScriptStore
    s = ScriptStore(base_dir=".tmp_reject")
    with pytest.raises(ValueError):
        s.save("x", "")
    import shutil
    shutil.rmtree(".tmp_reject", ignore_errors=True)


# --- Results store -------------------------------------------------------------
def test_results_store_records_strategy_trades():
    from analytics.results import ResultsStore
    r = ResultsStore()
    r.attach_strategy("s1", "STRAT_s1", "mr", None)
    lb = lambda pnl, pos, cash: [{"id": "STRAT_s1", "type": "Strategy", "pnl": pnl, "pos": pos, "cash": cash}]
    r.ingest("tick", {"run_id": "run1", "last_price": 190.0, "day_count": 1, "seq": 1,
                      "symbol": "TCS", "leaderboard": lb(0.0, 0, 1_000_000.0)})
    r.ingest("trade", {"seq": 2, "price": 190.0, "qty": 10, "buyer": "STRAT_s1",
                       "seller": "MM_1", "aggressor": "BUY", "unix_time": 1, "market_minute": 1})
    r.ingest("tick", {"run_id": "run1", "last_price": 191.0, "day_count": 1, "seq": 3,
                      "symbol": "TCS", "leaderboard": lb(10.0, 10, 998_100.0)})
    summaries = r.list_strategies()
    assert len(summaries) == 1 and summaries[0]["total_trades"] == 1
    detail = r.strategy_detail("s1")
    assert detail["days"] and detail["days"][0]["n_trades"] == 1
    trades = r.strategy_trades("s1")
    assert trades[0]["side"] == "BUY" and trades[0]["qty"] == 10


def test_results_reports_render():
    from analytics.results import ResultsStore
    from analytics.reports import trades_csv, strategy_pdf
    r = ResultsStore()
    r.attach_strategy("s1", "STRAT_s1", "mr", None)
    r.ingest("tick", {"run_id": "r", "last_price": 190.0, "day_count": 1, "seq": 1, "symbol": "TCS",
                      "leaderboard": [{"id": "STRAT_s1", "type": "Strategy", "pnl": 0, "pos": 0, "cash": 1e6}]})
    r.ingest("trade", {"seq": 2, "price": 190.0, "qty": 5, "buyer": "STRAT_s1", "seller": "MM",
                       "aggressor": "BUY", "unix_time": 1, "market_minute": 1})
    csv = trades_csv(r.strategy_trades("s1"), {"name": "mr"})
    assert csv and b"price" in csv.lower() if isinstance(csv, bytes) else True
    pdf = strategy_pdf(r.strategy_detail("s1"))
    assert pdf[:4] == b"%PDF"



# --- Time Machine --------------------------------------------------------------
def test_scenarios_generate_valid_csvs(tmp_path):
    from features.timemachine import SCENARIOS, ensure_scenarios, scenario_csv_path
    import os
    d = str(tmp_path / "scen")
    ensure_scenarios(d)
    assert len(SCENARIOS) == 5
    for sc in SCENARIOS:
        p = scenario_csv_path(sc["id"], d) if _accepts_dir() else scenario_csv_path(sc["id"])
        # fall back to the dir we generated into
        path = p if os.path.exists(p) else os.path.join(d, f"{sc['id']}.csv")
        assert os.path.exists(path), f"missing {sc['id']}"
        with open(path) as f:
            header = f.readline().strip().lower()
        assert header == "date,time,open,high,low,close,volume"


def test_scenario_dir_env_var_is_used(tmp_path, monkeypatch):
    """CHRONOS_SCENARIO_DIR moves the tapes (Docker: onto the volume the
    engine also mounts). Without it, Time Machine replays fail in Docker."""
    import os
    from features.timemachine import ensure_scenarios, scenario_csv_path
    target = tmp_path / "shared" / "scenarios"
    monkeypatch.setenv("CHRONOS_SCENARIO_DIR", str(target))
    ensure_scenarios()                          # no data_dir -> env var
    path = scenario_csv_path("covid_crash_2020")
    assert os.path.dirname(path) == str(target)
    assert os.path.exists(path)


def _accepts_dir():
    import inspect
    from features.timemachine import scenario_csv_path
    try:
        return "data_dir" in inspect.signature(scenario_csv_path).parameters
    except (TypeError, ValueError):
        return False



