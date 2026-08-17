"""Ground truth must stay on the evaluator side of the line (section 3)."""

import ast
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from causalops.domain import (
    Evidence,
    EvidenceKind,
    GatewaySymptom,
    IncidentScope,
    InitialAlertPacket,
    InvestigationReport,
    RootCauseCode,
)
from causalops.evidence import content_hash
from causalops.prompts import render_context

PACKAGE = "causalops"
SOURCE_DIR = Path(__file__).resolve().parents[2] / "src" / PACKAGE
EVALUATOR_MODULE = f"{PACKAGE}.evaluation"

WINDOW_START = datetime(2026, 8, 16, 10, 0, tzinfo=UTC)
WINDOW_END = datetime(2026, 8, 16, 10, 30, tzinfo=UTC)


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


def sample_evidence() -> Evidence:
    payload = {"p95_ms": 900}
    return Evidence(
        evidence_id="evidence-1",
        incident_id="incident-1",
        kind=EvidenceKind.METRIC,
        source="query_metric",
        observed_at=WINDOW_START,
        summary="gateway p95 latency rose to 900 ms",
        payload=payload,
        content_hash=content_hash(payload),
    )


def sample_context() -> str:
    scope = IncidentScope(
        incident_id="incident-1",
        services=("gateway", "orders", "inventory"),
        started_at=WINDOW_START,
        ended_at=WINDOW_END,
        endpoint="/api/orders",
    )
    packet = InitialAlertPacket(
        incident_id=scope.incident_id,
        window_start=WINDOW_START,
        window_end=WINDOW_END,
        endpoint=scope.endpoint,
        symptom=GatewaySymptom.ELEVATED_LATENCY,
        services=scope.services,
        alerted_at=WINDOW_START,
        alert_source_version="alert-1",
        symptom_evidence_id="evidence-1",
        topology_evidence_id="evidence-2",
    )
    return render_context(packet, scope, [sample_evidence()], [], 4, 2)


def test_no_investigator_module_imports_the_evaluator() -> None:
    offenders = {
        source.name
        for source in sorted(SOURCE_DIR.rglob("*.py"))
        if source.name != "evaluation.py"
        and EVALUATOR_MODULE in imported_modules(source)
    }

    assert offenders == set()


def test_the_import_scan_recognises_every_form_it_has_to_catch(tmp_path: Path) -> None:
    """Guards the guard: a scan that misses a form would pass while leaking."""
    forms = (
        "import causalops.evaluation",
        "from causalops.evaluation import ExpectedOutcome",
        "from . import evaluation",
        "from .evaluation import ExpectedOutcome",
    )

    for index, form in enumerate(forms):
        source = tmp_path / f"sample_{index}.py"
        source.write_text(form, encoding="utf-8")

        assert EVALUATOR_MODULE in imported_modules(source), form


def test_importing_the_investigator_never_loads_the_evaluator() -> None:
    """Catches what a syntax scan cannot, such as a dynamic import."""
    script = (
        "import sys\n"
        "import causalops.cli, causalops.workflow, causalops.run_records\n"
        f"print({EVALUATOR_MODULE!r} in sys.modules)\n"
    )

    finished = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )

    assert finished.stdout.strip() == "False"


def test_the_evaluator_depends_on_the_investigator_and_not_the_other_way() -> None:
    assert f"{PACKAGE}.domain" in imported_modules(SOURCE_DIR / "evaluation.py")


def test_the_model_context_names_no_cause_seed_or_scenario() -> None:
    context = sample_context()

    assert "gateway p95 latency rose" in context
    for code in RootCauseCode:
        assert code.value not in context
    for evaluator_word in ("seed", "scenario", "family", "expected", "predicate"):
        assert evaluator_word not in context.lower()


def test_the_context_fences_telemetry_as_untrusted() -> None:
    context = sample_context()

    assert "<untrusted-telemetry>" in context
    assert "</untrusted-telemetry>" in context


def test_a_report_has_nowhere_to_carry_an_expected_outcome() -> None:
    suspicious = [
        name
        for name in InvestigationReport.model_fields
        if "expect" in name or "predicate" in name or "seed" in name
    ]

    assert suspicious == []
