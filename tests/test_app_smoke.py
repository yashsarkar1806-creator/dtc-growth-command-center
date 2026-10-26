"""Headless smoke test of the Streamlit UI (no browser, no model downloads)."""
from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")


def test_app_runs_without_exceptions(monkeypatch):
    monkeypatch.setenv("USE_HF_MODELS", "0")
    at = st_testing.AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=120).run()
    assert not at.exception, at.exception
    assert at.title[0].value == "DTC Growth Command Center"
    assert len(at.tabs) == 5
    # Change the budget slider and re-run: optimiser should still succeed
    at.slider[0].set_value(at.slider[0].max).run()
    assert not at.exception
    assert any("Optimised revenue" in m.label for m in at.metric)
