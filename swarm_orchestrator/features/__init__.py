"""Chronos v2 feature layer (PROTOCOL_V2.md §3-§6).

Self-contained logic modules plus one shared FastAPI router:

- :mod:`features.montecarlo`  - real in-process Monte-Carlo counterfactual fan.
- :mod:`features.copilot`     - LLM strategy copilot (Gemini + deterministic template).
- :mod:`features.timemachine` - crisis-replay scenario presets + generated tapes.
- :mod:`features.narrator`    - explainable market narration (Gemini + heuristic).
- :mod:`features.routes_features` - ``router = APIRouter()`` wired by the integrator.
"""
