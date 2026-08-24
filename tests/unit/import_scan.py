"""The import scanner the ground-truth isolation tests are built on.

Kept out of `tests/security/test_ground_truth_isolation.py` so it is plain
module code rather than a file `pytest` also collects for test functions:
`pyproject.toml` sets no `python_files` override, so a `test_` function
defined in this file would never run.
"""

import ast
from pathlib import Path

PACKAGE = "causalops"


def imported_modules(source: Path) -> set[str]:
    """Every module name a file imports, with relative imports resolved.

    `import causalops.evaluation`, `from causalops.evaluation import X`,
    `from . import evaluation`, and `from .evaluation import X` all have to resolve
    to the same dotted name, or this test can pass while the rule is broken.
    """
    names: set[str] = set()
    for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                base = f"{PACKAGE}.{base}" if base else PACKAGE
            if base:
                names.add(base)
            names.update(
                f"{base}.{alias.name}" if base else alias.name for alias in node.names
            )
    return names


def dynamically_imported_modules(source: Path) -> set[str]:
    """Every module name passed as a string-literal argument to
    `importlib.import_module(...)` or `__import__(...)` -- the one import
    shape `imported_modules` above cannot see, since a function call is an
    `ast.Call` node, not the `ast.Import`/`ast.ImportFrom` nodes that
    function walks for. Scoped strictly to the argument of a matching
    `ast.Call`: a module name that merely appears in a comment, docstring,
    or other string literal never reaches this function's return value, so
    prose that explains an isolation rule by naming the module it forbids
    (exactly the shape this project's own isolation modules' docstrings
    use) cannot trigger a false positive here the way a plain substring
    scan of the whole file would.
    """
    names: set[str] = set()
    for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        is_import_module_call = (
            isinstance(target, ast.Attribute)
            and target.attr == "import_module"
            and isinstance(target.value, ast.Name)
            and target.value.id == "importlib"
        ) or (isinstance(target, ast.Name) and target.id == "import_module")
        is_dunder_import_call = (
            isinstance(target, ast.Name) and target.id == "__import__"
        )
        if not (is_import_module_call or is_dunder_import_call):
            continue
        if (
            node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            names.add(node.args[0].value)
    return names
