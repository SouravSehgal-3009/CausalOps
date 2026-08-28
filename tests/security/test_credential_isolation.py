"""This module's half of the "Provider and secret leakage" row in README.md's
threat-model summary: environment-only API key plus redaction; verify
it never reaches CLI text, config, artifacts, logs, reports, receipts, or
errors.

`LiveClaudeModel` (`live_model.py`) never reads `ANTHROPIC_API_KEY` itself --
it constructs a bare `ChatAnthropic(model_name=..., ...)` with no
`anthropic_api_key=` argument, and the installed SDK resolves the key from
the environment internally, inside `langchain_anthropic`/`anthropic`, code
this project does not own and cannot leak from by omission the way it could
if `causalops` itself held the key in a local variable, a log line, or an
error message.

An import-scan test (`tests/security/test_tool_boundary.py`'s own pattern)
cannot prove this the way it proves the tool-dispatch boundary, because
`os` is a standard-library module with no `causalops`-owned surface to scan
for. What actually matters is narrower and directly checkable: this module
never imports `os` (so it cannot call `os.environ`/`os.getenv` even by
accident in a later edit) and never writes the variable's name as a string
literal (so it is never the target of a hand-rolled read, a default-value
lookup, or an f-string that could end up in a report or an error message).
"""

import ast
from pathlib import Path

LIVE_MODEL_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "causalops" / "live_model.py"
)
LIVE_MODEL_SOURCE = LIVE_MODEL_PATH.read_text(encoding="utf-8")
LIVE_MODEL_AST = ast.parse(LIVE_MODEL_SOURCE)


def test_the_live_adapter_never_imports_os() -> None:
    for node in ast.walk(LIVE_MODEL_AST):
        if isinstance(node, ast.Import):
            assert "os" not in {alias.name for alias in node.names}
        if isinstance(node, ast.ImportFrom):
            assert node.module != "os"


def test_the_live_adapter_never_names_the_api_key_variable_in_code() -> None:
    """Scans the AST, not the raw source text, so explanatory comments
    (like this module's own, and `live_model.py`'s own docstring
    naming `ANTHROPIC_API_KEY` to explain why it is safe) do not trip this
    -- comments carry no runtime meaning and are not part of the AST. A
    docstring *is* a string constant in the AST, so this also proves the
    variable name never appears as an actual code-level string literal,
    only in prose. If this ever legitimately needs to change (e.g. an
    explicit `anthropic_api_key=` for a future multi-key scenario), the
    the "Provider and secret leakage" threat-model row needs a real
    redaction test to replace this one, not a deleted assertion.
    """
    for node in ast.walk(LIVE_MODEL_AST):
        if isinstance(node, ast.Constant) and node.value == "ANTHROPIC_API_KEY":
            raise AssertionError(
                f"{LIVE_MODEL_PATH}:{node.lineno} names the API key variable "
                "as a code-level string constant, not just in prose"
            )
