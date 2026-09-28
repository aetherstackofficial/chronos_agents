"""Boot the full default Chronos swarm: 15 MM + 4 Whale + 80 Retail.

Thin convenience wrapper — equivalent to `python worker.py --role all`.
worker.py already enforces the Windows import order (torch before zmq) and
sets WindowsSelectorEventLoopPolicy in its entrypoint.
"""

from worker import main

if __name__ == "__main__":
    main(["--role", "all"])
