"""Entry point that runs INSIDE a sandbox container (never in the bridge).

The strategy code arrives as environment variables, so no host file is mounted:

* ``STRATEGY_B64``                  -- the strategy source, base64
* ``TICKS_B64``                     -- sample ticks as JSON, base64 (dry run only)
* ``STRATEGY_NAME``/``STRATEGY_AGENT_ID`` -- identity for a real run

Modes::

    python -m runner.sandbox_entry run      # trade for real (docker_backend.launch)
    python -m runner.sandbox_entry dryrun   # check hooks once, print a report

The code is written to ``/tmp`` (the only writable place, mounted noexec) and the
environment variable is removed before the user's code starts, so the strategy
cannot read its own source back out of the process environment.
"""
from __future__ import annotations

import base64
import os
import sys
import tempfile


def _write(env_var: str, filename: str) -> str:
    raw = os.environ.pop(env_var, "")
    path = os.path.join(tempfile.gettempdir(), filename)
    with open(path, "wb") as fh:
        fh.write(base64.b64decode(raw))
    return path


def main(argv: list) -> int:
    mode = argv[0] if argv else ""
    if mode == "run":
        path = _write("STRATEGY_B64", "strategy.py")
        from runner.sdk import _main
        _main([path, "--name", os.environ.get("STRATEGY_NAME", "strategy"),
               "--agent-id", os.environ["STRATEGY_AGENT_ID"]])
        return 0
    if mode == "dryrun":
        path = _write("STRATEGY_B64", "strategy.py")
        ticks = _write("TICKS_B64", "ticks.json")
        from runner.dryrun import main as dry_main
        return dry_main([path, ticks])
    print("usage: python -m runner.sandbox_entry run|dryrun", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
