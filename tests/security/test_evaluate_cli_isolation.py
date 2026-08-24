"""`causalops-evaluate` must be a genuinely separate console script from
`causalops` -- `CLAUDE.md`: "`causalops evaluate` as a separate console
script never imported by `causalops.cli`." A lazy or deferred import would
satisfy that sentence's letter while defeating its purpose (this project's
own recorded history: a prior round only caught this by noticing a test
that should have failed under a reverted mutation didn't), so this file
proves it two ways: a static AST scan of the whole file (which finds an
import at any nesting depth, not just module-level statements) and a plain
substring check that also closes a dynamic `importlib.import_module(...)`
loophole the AST scan alone cannot see.
"""

from pathlib import Path

from import_scan import PACKAGE, dynamically_imported_modules, imported_modules

REPOSITORY = Path(__file__).resolve().parents[2]
CLI_SOURCE = REPOSITORY / "src" / PACKAGE / "cli.py"
EVALUATE_CLI_MODULE = f"{PACKAGE}.evaluate_cli"


def test_cli_never_imports_evaluate_cli_at_any_nesting_depth() -> None:
    """`imported_modules` walks the full AST (`ast.walk`, not just
    top-level statements), so this catches an `import`/`from ... import`
    placed inside a function body, an `if`, or a `try` -- a real lazy
    import -- not only one written at the top of the file."""
    names = imported_modules(CLI_SOURCE)

    assert EVALUATE_CLI_MODULE not in names
    assert not any(name.startswith(f"{EVALUATE_CLI_MODULE}.") for name in names)


def test_cli_source_never_names_evaluate_cli_at_all() -> None:
    """Belt and suspenders for a loophole the AST scan above cannot see: a
    dynamic `importlib.import_module("causalops.evaluate_cli")` or
    `__import__(...)` call is not an `ast.Import`/`ast.ImportFrom` node, so
    it would pass the test above while still coupling the two modules. The
    substring `"evaluate_cli"` has no legitimate reason to appear anywhere
    in `cli.py`'s source at all, dynamic import or otherwise."""
    source = CLI_SOURCE.read_text(encoding="utf-8")

    assert "evaluate_cli" not in source


def test_evaluate_cli_never_imports_cli_at_any_nesting_depth() -> None:
    """The reverse direction: `causalops.evaluate_cli` importing
    `causalops.cli` would just as surely defeat the isolation both modules
    are meant to keep, even though `CLAUDE.md` only states the one
    direction explicitly."""
    evaluate_cli_source = REPOSITORY / "src" / PACKAGE / "evaluate_cli.py"
    names = imported_modules(evaluate_cli_source)

    assert f"{PACKAGE}.cli" not in names
    assert not any(name.startswith(f"{PACKAGE}.cli.") for name in names)


def test_evaluate_cli_never_dynamically_imports_cli() -> None:
    """The reverse direction's own closure for the dynamic-import loophole
    the forward direction closes with a blunt substring check
    (`test_cli_source_never_names_evaluate_cli_at_all`). A plain substring
    check cannot be reused here: `evaluate_cli.py`'s own module docstring
    legitimately names `causalops.cli` in prose explaining this exact
    isolation rule (see its own top-of-file docstring), so a substring
    check would false-positive on the file doing the right thing.
    `dynamically_imported_modules` is scoped to `ast.Call` nodes matching
    `importlib.import_module(...)`/`__import__(...)` with a string-literal
    argument, so prose, comments, and docstrings can never trigger it --
    only an actual dynamic-import call can. This closes a currently-
    theoretical gap (nothing in this project dynamically imports anything
    today) the AST-walk in the test above cannot see on its own, since a
    function call is an `ast.Call` node, not the `ast.Import`/
    `ast.ImportFrom` nodes that scan is built on."""
    evaluate_cli_source = REPOSITORY / "src" / PACKAGE / "evaluate_cli.py"
    names = dynamically_imported_modules(evaluate_cli_source)

    assert f"{PACKAGE}.cli" not in names
    assert not any(name.startswith(f"{PACKAGE}.cli.") for name in names)
