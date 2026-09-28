"""Chronos strategy runner (Pillar 2).

Public surface:
    - ``Strategy``       : base class users subclass to write a trading strategy.
    - ``run_strategy``   : blocking entry-point that wires a ``Strategy`` to the engine.
    - ``SandboxManager`` : launches / supervises strategy processes (used by the bridge).

The runner is a *thin standalone client*: it talks to the engine purely over ZeroMQ
(the wire protocol in ``PROTOCOL.md`` sections 4/5) and never imports from ``core/``,
``engine/`` or ``agents/``. It is intentionally shippable on its own.

``SandboxManager`` is imported lazily (PEP 562) so that a pure-SDK client — e.g. a
strategy container that only has ``pyzmq`` installed — can ``import runner`` without
needing ``psutil`` or any other supervision dependency.
"""

from typing import Any

from .sdk import Strategy, run_strategy

__all__ = ["Strategy", "run_strategy", "SandboxManager"]

__version__ = "2.0.0"


def __getattr__(name: str) -> Any:
    """Lazily resolve ``SandboxManager`` so pure-SDK clients skip psutil."""
    if name == "SandboxManager":
        from .sandbox import SandboxManager

        return SandboxManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
