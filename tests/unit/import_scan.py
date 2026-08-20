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
