"""Pytest configuration — make the swarm_orchestrator package root importable."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# On Windows, loading pyzmq's DLLs before torch makes torch's c10.dll fail to
# initialise (WinError 1114). worker.py imports the models before zmq for the
# same reason; give the test session that order too.
try:
    import torch  # noqa: F401
except ImportError:
    pass
