"""`TECHNICAL_SPEC.md` section 11 requires tracing force-disabled at the
entry point, not merely documented. Importing `causalops` at all is that
entry point (`src/causalops/__init__.py`) -- this proves it actually
happens, not just that `.env.example` recommends it (`test_env_example.py`
covers the file; this covers the code).
"""

import importlib
import os

import pytest

import causalops


def test_importing_causalops_forces_both_tracing_variables_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit ambient "true" must not survive -- tracing is prohibited
    outright, not merely off by default, so this is not `setdefault`."""
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")

    importlib.reload(causalops)

    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert os.environ["LANGCHAIN_TRACING_V2"] == "false"


def test_importing_causalops_sets_tracing_off_even_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)

    importlib.reload(causalops)

    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert os.environ["LANGCHAIN_TRACING_V2"] == "false"
