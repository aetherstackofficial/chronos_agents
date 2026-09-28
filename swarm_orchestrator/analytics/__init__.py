"""Chronos analytics package (PROTOCOL_V2 sections 1-2).

Public surface:
    - ``ResultsStore``            : session-scoped strategy results store
      (``analytics.results``), fed by the bridge pump via ``ingest``.
    - ``analytics.reports``       : CSV / PDF report rendering functions.
    - ``analytics.routes_scripts``: ``router`` for the script-library API.
    - ``analytics.routes_results``: ``router`` for the results/reports API.

Routers are intentionally NOT imported here so that consumers that only need
``ResultsStore`` (e.g. the bridge pump) do not pull in FastAPI or reportlab;
the integrator imports the route modules explicitly when wiring ``bridge.py``.
"""

from typing import Any

__all__ = ["ResultsStore"]

__version__ = "2.0.0"


def __getattr__(name: str) -> Any:
    """Lazily resolve ``ResultsStore`` (PEP 562) to keep imports light."""
    if name == "ResultsStore":
        from .results import ResultsStore

        return ResultsStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
