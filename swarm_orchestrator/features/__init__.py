"""Chronos v2 feature layer.

Self-contained logic modules plus one shared FastAPI router:

- :mod:`features.timemachine` - crisis-replay scenario presets + generated tapes.
- :mod:`features.routes_features` - ``router = APIRouter()`` wired by the integrator.
"""
