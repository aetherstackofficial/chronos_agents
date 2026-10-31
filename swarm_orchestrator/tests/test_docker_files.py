"""Images that load runner/ must also ship what runner/sdk.py imports (core/market_context.py)."""
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("name", ["Dockerfile.runner", "Dockerfile.sandbox"])
def test_image_that_copies_the_sdk_also_copies_market_context(name):
    text = (ROOT / name).read_text(encoding="utf-8")
    assert "swarm_orchestrator/runner" in text
    assert "swarm_orchestrator/core/market_context.py" in text, (
        f"{name} copies runner/ (whose __init__ imports the SDK) but not core/market_context.py: "
        "the container would crash on start with ModuleNotFoundError")
