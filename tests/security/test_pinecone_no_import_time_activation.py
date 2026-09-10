"""The `pinecone` dependency (`pinecone_runbooks.py`,
`retrieval_experiment.py`'s `RAG_EXPERIMENT_ENABLED` gate) must never
activate merely because `causalops.cli` or `causalops.evaluate_cli` was
imported -- `live_setup._build_tool_registry` only imports
`causalops.pinecone_runbooks` (and, transitively, the `pinecone` package
itself) inside the branch taken when the experiment is explicitly enabled.

A stronger, simpler claim than `test_no_tracing.py` can make about
LangSmith: LangSmith is unavoidably pulled in transitively by
`langchain-core` regardless of whether tracing is used, so that file
proves no *client construction* or *network send* happens. `pinecone` has
no such transitive entanglement -- nothing else in this codebase imports
it -- so the property provable here is stronger: the module is not even
present in `sys.modules` after import. Run in a fresh subprocess, the same
reason `test_importing_causalops_cli_never_sends_a_tracing_request` does:
a property about *what happens during an import* cannot be proven by
checking `sys.modules` after the import already ran in a process that may
have imported it for unrelated reasons (e.g. pytest collecting this file
itself, which does import `pinecone.exceptions` at module scope for the
fake-client tests).
"""

import subprocess
import sys


def _assert_pinecone_absent_after_import(module: str) -> None:
    script = (
        f"import {module}\n"
        "import sys\n"
        "leaked = sorted(m for m in sys.modules if 'pinecone' in m)\n"
        "assert 'pinecone' not in sys.modules, leaked\n"
        "print('ok')\n"
    )
    finished = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert finished.returncode == 0, finished.stderr
    assert finished.stdout.strip() == "ok"


def test_importing_causalops_cli_never_loads_pinecone() -> None:
    _assert_pinecone_absent_after_import("causalops.cli")


def test_importing_causalops_evaluate_cli_never_loads_pinecone() -> None:
    _assert_pinecone_absent_after_import("causalops.evaluate_cli")


def test_importing_causalops_live_setup_never_loads_pinecone() -> None:
    """The narrowest useful case: even the module that conditionally
    imports `pinecone_runbooks` does not load `pinecone` merely by being
    imported itself -- only `_build_tool_registry` actually taking the
    enabled branch does."""
    _assert_pinecone_absent_after_import("causalops.live_setup")
