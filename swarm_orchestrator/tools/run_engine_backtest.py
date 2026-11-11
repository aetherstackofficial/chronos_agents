"""Run the real engine server with the dashboard's market-mode switches.

    python -m tools.run_engine_backtest

Same as ``python -m engine.server`` (ZeroMQ order/data ports, 1 Hz physics ticks), except that
``agents.backtest_mode.install_switchable()`` runs first, in this process. The dashboard ("Your Data" page,
Market mode card) can then choose, per run:

* the mode: "free" (the engine's own behaviour: after the user's first fill the agents make the price) or
  "backtest" (the real tape keeps printing, shifted by the participants' footprint; agents/backtest_mode.py);
* what happens when the replayed data runs out: "continue" (the agents keep trading) or "pause".

Defaults: the mode is "free" unless CHRONOS_MODE=backtest; the tape-end policy is CHRONOS_TAPE_END, and
this launcher sets it to "continue" when it is not set. It also installs tools/sim_speed.py (the chart's
fast-forward button: N times faster until the user's next trade). engine/server.py is not modified.
Importing this module does not start the server.
"""
import os


def main() -> None:
    os.environ.setdefault("CHRONOS_TAPE_END", "continue")
    from agents import backtest_mode
    backtest_mode.install_switchable()
    from engine import server
    from tools import real_tape, sim_speed
    real_tape.install()                # the chart's eye button: real candles behind the chart (tools/real_tape.py)
    sim_speed.install(server)          # the chart's fast-forward button (tools/sim_speed.py)
    server.main()


if __name__ == "__main__":
    main()
