# Realism validation suite

Implements "Market Simulator Realism Validation Spec" v1.0 against the current
simulator, with thresholds calibrated on real NSE 1-minute data.

```
pip install -r validation/requirements.txt
python -m validation.fetch_data                      # Kaggle NSE files -> data/real/nse_kaggle/
python -m validation.calibrate --n-days 268          # real-data bands (full mode)
python -m validation.calibrate --n-days 54 --instruments RELIANCE   # smoke bands
python -m validation.runner --mode smoke             # ~5-10 min, never a realism claim
python -m validation.runner --mode full              # ~1 h on 12 cores
```

Reports land in `validation/reports/` (`report_card_*.json`, spec §13 schema).

| File | Role |
|---|---|
| `sim_driver.py` | Headless seeded driver: live engine + orchestrator + agents, in-process client, swarm built exactly as a seeded `worker.py` builds it (`agents/swarm_factory.py`: tick clock, per-agent seeded RNGs), snapshot/restore |
| `data.py` | Real-data cleaning, bar series, returns, deseasonalisation, Parkinson sigma |
| `stats_utils.py` | Block bootstrap, ACF, Ljung-Box, power-law fit, bands |
| `checks/` | T1-T8 and A1-A9 |
| `calibrate.py` | §15 bands: rolling windows, 70/30 holdout |
| `runner.py`, `report.py` | Seeds in parallel, §2.6 aggregation, report card |
| `config/thresholds.yaml` | Spec defaults plus cost model, with sources |

Real-vs-sim comparisons use trade-price closes and the Parkinson range
estimator on both sides, because the reference data has no quotes and no
sub-bar data. Spec-default criteria for T1-T4 are judged on the simulator's
mid-quote, as the spec defines them.
