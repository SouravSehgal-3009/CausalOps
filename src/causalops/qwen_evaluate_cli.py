"""Private-VM-only runner for the fixed Phase 4 Qwen candidate corpus."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from causalops.domain import (
    Budgets,
    InvestigationResult,
    PolicyResult,
    StoredIncident,
    ToolOutcome,
    utc_now,
)
from causalops.evaluation import ExpectedOutcome, score_run
from causalops.graph import run_graph_investigation
from causalops.live_setup import (
    CANDIDATE_EVALUATION_VARIABLE,
    VM_EXECUTION_ENV,
    VM_EXECUTION_ENV_VARIABLE,
    ProviderDisabledError,
    build_ollama_candidate_model_and_registry,
)
from causalops.phase4_evaluation import (
    APPROVED_QWEN_MODEL,
    EVALUATION_FAMILIES,
    EVALUATION_SEEDS,
    CandidateEvaluationRecord,
)
from causalops.report import render_report as render_markdown_report
from causalops.run_records import RunRecorder, finalize_investigation, write_jsonl
from causalops.runbooks import RunbookIndex
from causalops.scenario_control import reset_scenario, run_paths, start_scenario

OLLAMA_IMAGE_DIGEST_VARIABLE = "CAUSALOPS_OLLAMA_IMAGE_DIGEST"
QWEN_MANIFEST_DIGEST_VARIABLE = "CAUSALOPS_QWEN_MANIFEST_DIGEST"


def _digest_from_environment(name: str, environment: dict[str, str]) -> str:
    value = environment.get(name, "")
    if not value.startswith("sha256:") or len(value) != 71:
        raise ProviderDisabledError(
            f"{name} must be a sha256 digest for candidate runs"
        )
    try:
        int(value.removeprefix("sha256:"), 16)
    except ValueError as error:
        raise ProviderDisabledError(
            f"{name} must be a sha256 digest for candidate runs"
        ) from error
    return value


def _assert_candidate_environment(environment: dict[str, str]) -> tuple[str, str]:
    if (
        environment.get(VM_EXECUTION_ENV_VARIABLE, "").strip().lower()
        != VM_EXECUTION_ENV
    ):
        raise ProviderDisabledError("Qwen evaluation is private-VM-only")
    if environment.get(CANDIDATE_EVALUATION_VARIABLE, "").strip().lower() not in {
        "1",
        "true",
        "yes",
    }:
        raise ProviderDisabledError(
            "Qwen evaluation requires candidate-evaluation approval"
        )
    return (
        _digest_from_environment(OLLAMA_IMAGE_DIGEST_VARIABLE, environment),
        _digest_from_environment(QWEN_MANIFEST_DIGEST_VARIABLE, environment),
    )


def _git_provenance(root: Path) -> tuple[str, bool]:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip(), bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )


def _load_expected_outcome(paths_root: Path) -> ExpectedOutcome:
    raw = json.loads(
        (paths_root / "evaluator" / "expected.json").read_text(encoding="utf-8")
    )
    return ExpectedOutcome.model_validate(
        {
            "root_cause": raw["root_cause"],
            "disposition": raw["disposition"],
            "predicates": raw.get("predicates", []),
        }
    )


def _sequence_is_valid(recorder: RunRecorder) -> bool:
    return [event.sequence for event in recorder.events] == list(
        range(1, len(recorder.events) + 1)
    )


def _policy_escape(result: InvestigationResult, recorder: RunRecorder) -> bool:
    receipt_ids = {receipt.receipt_id for receipt in result.receipts}
    return (
        not _sequence_is_valid(recorder)
        or set(result.report.receipt_ids) != receipt_ids
        or any(
            receipt.incident_id != result.report.incident_id
            for receipt in result.receipts
        )
    )


def _new_target(root: Path) -> Path:
    target = (
        root
        / "results"
        / "candidate-evaluations"
        / hashlib.sha256(os.urandom(32)).hexdigest()[:24]
    )
    target.mkdir(parents=True, exist_ok=False)
    return target


def run_candidate_evaluation(
    root: Path, target: Path
) -> list[CandidateEvaluationRecord]:
    """Run exactly one tool-enabled Qwen result for each fixed corpus case.

    The caller must already be an explicitly approved private-VM process.
    This function deliberately has no baseline arm: Phase 4's Qwen gate is
    defined over 12 diagnosis cases, not the legacy Claude paired comparison.
    """
    environment = dict(os.environ)
    image_digest, manifest_digest = _assert_candidate_environment(environment)
    # The shared 360s default (domain.py's Budgets) is tight for this
    # CPU-only local model: even with "think": false, a single repair-turn
    # HTTP call can measure 120-200s, and one repair is often enough to
    # legitimately exhaust 360s total before final_assessment ever runs
    # (measured live: 362s for a run that self-corrected correctly and
    # simply ran out of clock). Scoped to this candidate-evaluation CLI only
    # -- the shared default (and Claude/replay's fast real-world latency)
    # is untouched.
    budgets = Budgets(wall_clock_seconds=900)
    git_sha, git_dirty = _git_provenance(root)
    runbook_corpus_version = RunbookIndex().corpus_version
    if runbook_corpus_version is None:
        raise RuntimeError("candidate corpus version is required")
    records: list[CandidateEvaluationRecord] = []
    for family in EVALUATION_FAMILIES:
        for seed_name in EVALUATION_SEEDS:
            incident_id = start_scenario(root, family, seed_name)
            try:
                paths = run_paths(root, incident_id)
                incident = StoredIncident.model_validate_json(
                    paths.incident_file.read_text(encoding="utf-8")
                )
                model, registry, model_name = build_ollama_candidate_model_and_registry(
                    incident, paths, budgets, environment
                )
                if model_name != APPROVED_QWEN_MODEL:
                    raise RuntimeError(
                        "candidate composition returned an unapproved model"
                    )
                recorder = RunRecorder(utc_now)
                result = run_graph_investigation(
                    incident.scope,
                    incident.packet,
                    incident.evidence,
                    model,
                    registry,
                    recorder,
                    budgets,
                    utc_now,
                    model_name=APPROVED_QWEN_MODEL,
                    suppress_escalation=True,
                )
                if not isinstance(result, InvestigationResult):
                    raise RuntimeError("candidate evaluation unexpectedly paused")
                finalize_investigation(
                    root / "results",
                    result.report,
                    recorder.events,
                    result.evidence,
                    result.receipts,
                    render_markdown_report(
                        result.report, result.evidence, result.receipts, model_name
                    ),
                )
                record = CandidateEvaluationRecord(
                    case_id=f"{family}/{seed_name}",
                    investigation_id=result.report.investigation_id,
                    model_name=APPROVED_QWEN_MODEL,
                    ollama_image_digest=image_digest,
                    model_manifest_digest=manifest_digest,
                    git_sha=git_sha,
                    git_dirty=git_dirty,
                    versions=result.report.versions,
                    fixture_sha256=hashlib.sha256(
                        (root / "lab" / "scenarios" / f"{family}.json").read_bytes()
                    ).hexdigest(),
                    runbook_corpus_version=runbook_corpus_version,
                    retrieval_index_identity=f"fts5:{runbook_corpus_version}",
                    configured_budgets=budgets,
                    runbook_query_count=sum(
                        receipt.tool.value == "search_runbooks"
                        and receipt.policy_result is PolicyResult.ALLOWED
                        and receipt.outcome is ToolOutcome.EXECUTED
                        for receipt in result.receipts
                    ),
                    retrieval_mode=result.report.retrieval_mode,
                    scores=score_run(
                        result.report,
                        result.evidence,
                        result.receipts,
                        _load_expected_outcome(paths.root),
                    ),
                    sequence_valid=_sequence_is_valid(recorder),
                    policy_escape=_policy_escape(result, recorder),
                    disposition=result.report.disposition,
                    failure_reason=result.report.reason_code,
                    wall_clock_ms=result.report.latency_ms,
                    # VM and local-Ollama allocation are not measured yet;
                    # zero would be a false claim of a free candidate run.
                    local_compute_cost_usd=None,
                )
                records.append(record)
                write_jsonl(target / "records.jsonl", records)
                if record.policy_escape or record.disposition.value == "FAILED_SAFE":
                    return records
            finally:
                already_failing = sys.exc_info()[0] is not None
                try:
                    reset_scenario(root, incident_id)
                except Exception as reset_error:
                    if already_failing:
                        print(
                            f"FAIL RESET_SCENARIO_FAILED_DURING_CLEANUP {reset_error}"
                        )
                    else:
                        raise
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="causalops-qwen-evaluate")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    arguments = parser.parse_args(argv)
    try:
        _assert_candidate_environment(dict(os.environ))
        target = _new_target(arguments.root.resolve())
        records = run_candidate_evaluation(arguments.root.resolve(), target)
    except (OSError, ProviderDisabledError, RuntimeError) as error:
        print(f"FAIL QWEN_CANDIDATE_EVALUATION {error}")
        return 1
    print(f"candidate records: {len(records)} ({target})")
    return 0


if __name__ == "__main__":  # pragma: no cover - package entry point calls main
    raise SystemExit(main())
