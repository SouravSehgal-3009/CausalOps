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
    tree = ast.parse(source.read_text(encoding="utf-8"))
    importlib_aliases = {"importlib"}
    import_module_aliases = {"import_module"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            importlib_aliases.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name == "importlib"
            )
        elif isinstance(node, ast.ImportFrom) and node.module == "importlib":
            import_module_aliases.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name == "import_module"
            )

    def literal_string(expression: ast.expr) -> str | None:
        if isinstance(expression, ast.Constant) and isinstance(expression.value, str):
            return expression.value
        if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Add):
            left = literal_string(expression.left)
            right = literal_string(expression.right)
            return None if left is None or right is None else left + right
        return None

    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        is_import_module_call = (
            isinstance(target, ast.Attribute)
            and target.attr == "import_module"
            and isinstance(target.value, ast.Name)
            and target.value.id in importlib_aliases
        ) or (isinstance(target, ast.Name) and target.id in import_module_aliases)
        is_dunder_import_call = (
            isinstance(target, ast.Name) and target.id == "__import__"
        )
        if not (is_import_module_call or is_dunder_import_call):
            continue
        if node.args:
            name = literal_string(node.args[0])
            if name is not None:
                names.add(name)
    return names
