"""`.env.example` documents the environment variables CausalOps reads.

There is no `.env` loader in this project (see the file's own header), so
this is a reference a developer copies values out of, not something
CausalOps reads directly. Tracing enforcement itself lives in code, not in
this file: `test_tracing_disabled.py` proves `causalops.__init__` forces
both LangSmith variables off regardless of what is set here or in the
shell. This file only has to document that recommendation accurately.
"""

from pathlib import Path

ENV_EXAMPLE = Path(__file__).resolve().parents[2] / ".env.example"


def parsed_env_example() -> dict[str, str]:
    pairs: dict[str, str] = {}
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        name, _, value = stripped.partition("=")
        pairs[name] = value
    return pairs


def test_env_example_documents_the_anthropic_api_key() -> None:
    assert "ANTHROPIC_API_KEY" in parsed_env_example()


def test_both_langsmith_tracing_variables_default_off() -> None:
    pairs = parsed_env_example()

    for name in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"):
        assert pairs[name] == "false", f"{name} must default to false"


def test_no_langsmith_credential_is_documented() -> None:
    """A key here is the one thing that could make tracing actually send data."""
    pairs = parsed_env_example()

    assert "LANGSMITH_API_KEY" not in pairs
    assert "LANGCHAIN_API_KEY" not in pairs
