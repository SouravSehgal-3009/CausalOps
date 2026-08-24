"""`causalops-evaluate`: the Unit 3c paired live comparison.

`TECHNICAL_SPEC.md` §10: "same model and same answer-neutral initial alert
compare a no-tool baseline against the tool-enabled LangGraph workflow,"
over a predefined paired set of held-out incidents, without ever invoking
the escalation path (HITL is demonstrated and tested separately).

This is a genuinely separate console script from `causalops` -- registered
under its own `[project.scripts]` entry (`causalops-evaluate`) in
`pyproject.toml` -- not a subcommand of it. `causalops.cli` never imports
this module, and this module never imports `causalops.cli`;
`tests/security/test_evaluate_cli_isolation.py` proves the first half
directly. Both scripts share their live-model/tool-registry construction
through `causalops.live_setup`, the neutral module neither one owns.

This script drives real, billed Anthropic requests through the exact same
`cost_ledger.py` reservation/settlement machinery every other live call in
this project already uses, against the same application-wide
`LIVE_EVALUATION_MAX_USD` ceiling -- it is not a separate budget. It also
requires the local synthetic lab running (`causalops lab up`) first, the
same precondition `causalops scenario start` already has, since each
incident is seeded through `scenario_control.start_scenario`.
"""

import argparse
import hashlib
import json
import os
import subprocess
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path

from causalops.cost_ledger import run_cost_totals
from causalops.doctor import ProjectPaths, find_project_root
from causalops.domain import Budgets, InvestigationResult, StoredIncident, utc_now
from causalops.evaluation import EvaluationRecord, ExpectedOutcome, score_run
from causalops.evidence import new_opaque_id
from causalops.graph import run_graph_investigation
from causalops.live_setup import build_model_and_registry, live_evaluation_ceiling_usd
from causalops.pricing import CLAUDE_SONNET_5_PRICING
from causalops.report import render_report as render_markdown_report
from causalops.run_records import (
    RunEvent,
    RunRecorder,
    RunRecordError,
    finalize_investigation,
    write_jsonl,
)
from causalops.runbooks import RunbookIndex
from causalops.scenario_control import (
    LabError,
    reset_scenario,
    run_paths,
    start_scenario,
)

# The frozen four-pair held-out corpus. `lab/scenarios/*.json` has exactly
# these four families today, each already carrying a `seed_variants.
# evaluation` block distinct from `seed_variants.development` --
# `TECHNICAL_SPEC.md` §10 permits up to six held-out incidents; this project
# chose four (`CLAUDE.md`'s Unit 3c scoping note). Adding a fifth or sixth
# family later is a one-line change here, not a redesign.
EVALUATION_FAMILIES: tuple[str, ...] = (
    "ambiguous_telemetry",
    "configuration_change",
    "downstream_timeout_retry_amplification",
    "resource_pool_saturation",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="causalops-evaluate",
        description=(
            "Run the frozen four-pair paired live comparison against the "
            "local synthetic lab (`causalops lab up` first) and score it "
            "against evaluator-only expected outcomes. Sends real, billed "
            "Anthropic requests under the application-wide "
            "LIVE_EVALUATION_MAX_USD ceiling."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"causalops-evaluate {version('causalops')}",
    )
    return parser


def _git_provenance(root: Path) -> tuple[str, bool]:
    """`(sha, dirty)` for `TECHNICAL_SPEC.md` §10's "Record Git SHA,
    clean/dirty status." A published evaluation record from a dirty tree is
    not reproducible -- recording both facts here lets whoever reads the
    record judge that for themselves rather than this script silently
    deciding it for them."""
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return sha, bool(status.strip())


def _fixture_sha256(root: Path, family: str) -> str:
    """SHA-256 of the exact `lab/scenarios/<family>.json` bytes this
    incident's family was started from -- a content hash rather than a
    hand-maintained version string, so it cannot silently drift from what
    was actually used and needs no change to the frozen scenario files."""
    path = root / "lab" / "scenarios" / f"{family}.json"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_expected_outcome(paths_root: Path) -> ExpectedOutcome:
    """Reads `runs/<incident_id>/evaluator/expected.json` directly --
    `telemetry.RunPaths` deliberately has no accessor for the `evaluator/`
    directory (`tests/security/test_ground_truth_isolation.py`'s own
    boundary: "code that cannot name a path cannot read it by accident").
    This script is the evaluator side of that line, not the investigator
    side, so it builds the path itself rather than widening `RunPaths`."""
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


def _run_id_from_events(events: Sequence[RunEvent]) -> str:
    """`run_id` is internal bookkeeping, deliberately absent from
    `InvestigationReport` itself (see `graph.py`'s own comment on why) --
    but `cost_ledger.run_cost_totals` is keyed by `run_id`, not
    `investigation_id`, so this script recovers it from the
    `investigation_started` event every run records first (`graph.py`'s
    `run_graph_investigation`, Unit 3c)."""
    for event in events:
        if event.name == "investigation_started":
            run_id = event.fields.get("run_id")
            if isinstance(run_id, str):
                return run_id
    raise RuntimeError(
        "no investigation_started event with a run_id field -- "
        "cannot look up this run's cost"
    )


def _run_one(
    *,
    root: Path,
    family: str,
    incident: StoredIncident,
    expected: ExpectedOutcome,
    fixture_sha256: str,
    runbook_corpus_version: str | None,
    git_sha: str,
    git_dirty: bool,
    configured_ceiling_usd: float,
    no_tool_baseline: bool,
) -> EvaluationRecord:
    """One scored run -- baseline or tool-enabled -- against an already
    seeded incident. `suppress_escalation=True` on every call: §10 forbids
    the paired comparison from ever invoking the escalation path.

    No `checkpointer` is passed to `run_graph_investigation`, so it defaults
    to a fresh, process-local `InMemorySaver()`: a suppressed run never
    pauses, so there is nothing to resume across a process boundary, and
    this avoids writing scored-run graph checkpoints into the shared
    `checkpoints.db` at all. The cost ledger is a separate connection to
    that same file, opened directly here -- unrelated to the graph
    checkpointer, and still the one ledger every live call shares.
    """
    budgets = Budgets()
    checkpoints_db = ProjectPaths(root=root).checkpoints_db
    checkpoints_db.parent.mkdir(parents=True, exist_ok=True)
    paths = run_paths(root, incident.scope.incident_id)
    model, registry, model_name, ledger_conn = build_model_and_registry(
        incident, paths, budgets, "claude", checkpoints_db
    )
    assert ledger_conn is not None, "causalops-evaluate always uses the live model"
    try:
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
            model_name=model_name,
            suppress_escalation=True,
            no_tool_baseline=no_tool_baseline,
        )
        if not isinstance(result, InvestigationResult):
            raise RuntimeError(
                f"{family}: a scored run (no_tool_baseline={no_tool_baseline}) "
                "paused for escalation despite suppress_escalation=True -- "
                "structurally unreachable if build_graph's suppression is "
                "wired correctly"
            )
        run_id = _run_id_from_events(recorder.events)
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
        scores = score_run(result.report, result.evidence, result.receipts, expected)
        reserved_usd, actual_usd = run_cost_totals(ledger_conn, run_id)
        # A settled ledger row can never carry `actual_usd == 0.0` (tokens
        # always cost something > 0), so `reserved_usd > 0` with
        # `actual_usd == 0.0` unambiguously means "reserved, never
        # settled" -- a crash, timeout, or refusal mid-request -- and the
        # record should say so honestly (`None`) rather than report a real
        # zero-dollar run.
        never_settled = reserved_usd > 0 and actual_usd == 0.0
        mode = "no_tool_baseline" if no_tool_baseline else "tool_enabled"
        return EvaluationRecord(
            run_key=f"{incident.scope.incident_id}/{model_name}/{mode}",
            investigation_id=result.report.investigation_id,
            incident_id=result.report.incident_id,
            expected=expected,
            scores=scores,
            git_sha=git_sha,
            git_dirty=git_dirty,
            versions=result.report.versions,
            retrieval_mode=result.report.retrieval_mode,
            runbook_corpus_version=runbook_corpus_version,
            fixture_sha256=fixture_sha256,
            model_name=model_name,
            pricing_source=CLAUDE_SONNET_5_PRICING.source,
            pricing_verified_on=CLAUDE_SONNET_5_PRICING.verified_on,
            configured_ceiling_usd=configured_ceiling_usd,
            reserved_usd=reserved_usd,
            actual_usd=None if never_settled else actual_usd,
        )
    finally:
        ledger_conn.close()


def run_evaluation(root: Path, target: Path) -> list[EvaluationRecord]:
    """`target` (a directory under `results/evaluations/`, already created
    with an empty `records.jsonl` by `main`'s `_new_evaluation_target`) is
    rewritten with the complete list of records produced SO FAR after every
    single `_run_one` call returns -- not batched to the end. Each of the
    eight scored runs in the full corpus is a real, billed Anthropic
    request; an exception partway through (a realistic risk -- this
    project's own live calls have hit failures before) must not discard
    every already-completed, already-paid-for record along with it. Full
    rewrite rather than a true append is deliberate: `write_jsonl` already
    exists, writes the complete file in one `path.write_text` call, and the
    corpus is at most eight records -- reusing it keeps this function
    simple and gives `records.jsonl` a single, uniform "whatever this file
    contains is exactly what has been scored so far" contract, rather than
    a second, append-only code path with its own correctness questions
    (partial lines, encoding boundaries) `write_jsonl` was never written to
    answer.
    """
    git_sha, git_dirty = _git_provenance(root)
    configured_ceiling_usd = live_evaluation_ceiling_usd(os.environ)
    runbook_corpus_version = RunbookIndex().corpus_version
    records_path = target / "records.jsonl"

    records: list[EvaluationRecord] = []
    for family in EVALUATION_FAMILIES:
        incident_id = start_scenario(root, family, "evaluation")
        try:
            paths = run_paths(root, incident_id)
            incident = StoredIncident.model_validate_json(
                paths.incident_file.read_text(encoding="utf-8")
            )
            expected = _load_expected_outcome(paths.root)
            fixture_sha256 = _fixture_sha256(root, family)
            for no_tool_baseline in (True, False):
                records.append(
                    _run_one(
                        root=root,
                        family=family,
                        incident=incident,
                        expected=expected,
                        fixture_sha256=fixture_sha256,
                        runbook_corpus_version=runbook_corpus_version,
                        git_sha=git_sha,
                        git_dirty=git_dirty,
                        configured_ceiling_usd=configured_ceiling_usd,
                        no_tool_baseline=no_tool_baseline,
                    )
                )
                # Durable the moment a real, billed result exists -- see
                # this function's own docstring for why this cannot wait
                # until every family has finished.
                write_jsonl(records_path, records)
        finally:
            reset_scenario(root, incident_id)
    return records


def _new_evaluation_target(root: Path) -> Path:
    """Mints this run's `results/evaluations/<opaque-id>/` directory and
    seeds it with an empty `records.jsonl` before `run_evaluation` ever
    starts -- so `main` has a real, existing path to report even if the
    very first scored run fails before producing anything, and
    `run_evaluation` has a file already in place to keep overwriting as
    real records land."""
    target = root / "results" / "evaluations" / new_opaque_id()
    target.mkdir(parents=True, exist_ok=True)
    write_jsonl(target / "records.jsonl", [])
    return target


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    start = Path.cwd()
    root = find_project_root(start)
    if root is None:
        print(f"FAIL PROJECT_ROOT_NOT_FOUND No pyproject.toml at or above {start}.")
        return 1
    target = _new_evaluation_target(root)
    records_path = target / "records.jsonl"
    try:
        records = run_evaluation(root, target)
    except (LabError, RunRecordError) as refusal:
        print(f"FAIL {refusal.reason_code.value} {refusal}")
        print(f"records so far: {records_path}")
        return 1
    except Exception as error:
        # A batch evaluation script should never hand the owner a raw
        # traceback for a real, actionable failure (a missing git binary,
        # a corrupt evaluator/expected.json, ...) -- the same "no raw
        # traceback reaches the owner" posture `causalops.cli` already
        # keeps for its own refusal paths, applied here to failure modes
        # specific to this script rather than the same three exception
        # types. Whatever records completed before this failure are
        # already durable on disk at `records_path` -- `run_evaluation`
        # itself rewrites that file after every completed run, not only
        # once this function returns -- so this branch still has a real
        # path to report, not nothing.
        print(f"FAIL INTERNAL_ERROR {error}")
        print(f"records so far: {records_path}")
        return 1
    for record in records:
        print(
            f"{record.run_key} diagnosis_correct={record.scores.diagnosis_correct} "
            f"disposition_correct={record.scores.disposition_correct} "
            f"reserved_usd={record.reserved_usd:.4f} actual_usd={record.actual_usd}"
        )
    print(f"records: {records_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
