"""`causalops-evaluate`: the paired live comparison.

Runs the same model against the same answer-neutral initial alert to
compare a no-tool baseline against the tool-enabled LangGraph workflow,
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
import sys
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from causalops.approvals import CheckpointStoreError
from causalops.cost_ledger import run_cost_totals
from causalops.doctor import API_KEY_VARIABLE, ProjectPaths, find_project_root
from causalops.domain import (
    Budgets,
    InvestigationResult,
    ReasonCode,
    RetrievalMode,
    StoredIncident,
    utc_now,
)
from causalops.evaluation import (
    EvaluationRecord,
    EvaluationSummary,
    ExpectedOutcome,
    score_run,
    summarize_evaluation,
)
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
# evaluation` block distinct from `seed_variants.development`. Adding a
# fifth or sixth family later is a one-line change here, not a redesign.
EVALUATION_FAMILIES: tuple[str, ...] = (
    "ambiguous_telemetry",
    "configuration_change",
    "downstream_timeout_retry_amplification",
    "resource_pool_saturation",
)

# The two arms of the paired comparison: the same model and same
# answer-neutral initial alert, run once with no tools and once through the
# tool-enabled LangGraph workflow. `_run_one` writes one of
# these two words as `run_key`'s final `/`-separated segment
# (`f"{incident_id}/{model_name}/{mode}"`); `_arm_of` below is the one place
# that reads it back out, so a change to either constant only has to update
# both ends of that same encoding.
MODE_NO_TOOL_BASELINE = "no_tool_baseline"
MODE_TOOL_ENABLED = "tool_enabled"

INFRASTRUCTURE_ABORT_REASONS = frozenset(
    {
        ReasonCode.COST_CEILING_EXCEEDED,
        ReasonCode.INPUT_TOKEN_CAP_EXCEEDED,
        ReasonCode.AMBIGUOUS_MODEL_REQUEST,
        ReasonCode.WALL_CLOCK_EXPIRED,
        ReasonCode.INTERNAL_ERROR,
    }
)


class EvaluationAborted(Exception):
    def __init__(self, reason: ReasonCode) -> None:
        self.reason = reason
        super().__init__(reason.value)


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
    """`(sha, dirty)`, recorded so every evaluation record states exactly
    what code produced it. An evaluation record from a dirty tree is not
    reproducible -- recording both facts here lets whoever reads the
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
    `run_graph_investigation`)."""
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
        reserved_usd, actual_usd, fully_settled = run_cost_totals(ledger_conn, run_id)
        # `actual_usd == 0.0` alone is not a reliable "nothing settled"
        # signal: a run whose first three of four model calls settle while
        # the fourth stays `RESERVED` (a timeout, a crash mid-call) has a
        # non-zero but PARTIAL `actual_usd` -- reporting that partial sum as
        # the run's complete cost would understate it silently. `cost_
        # ledger.run_cost_totals` now checks every row's state directly and
        # reports that as `fully_settled`; a run with any outstanding
        # reservation records `actual_usd=None` here rather than a number
        # that looks complete but is not.
        incomplete_settlement = not fully_settled
        mode = MODE_NO_TOOL_BASELINE if no_tool_baseline else MODE_TOOL_ENABLED
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
            actual_usd=None if incomplete_settlement else actual_usd,
            failure_reason=result.report.reason_code,
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
                record = _run_one(
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
                records.append(record)
                # Durable the moment a real, billed result exists -- see
                # this function's own docstring for why this cannot wait
                # until every family has finished.
                write_jsonl(records_path, records)
                if record.failure_reason in INFRASTRUCTURE_ABORT_REASONS:
                    raise EvaluationAborted(record.failure_reason)
        finally:
            # `reset_scenario` still runs unconditionally, on every path out
            # of `try` -- success or failure -- exactly as before this
            # comment. What changed: if the `try` block is already
            # propagating a real, billed run failure AND `reset_scenario`
            # itself also raises, Python's ordinary exception chaining would
            # otherwise replace the original exception with this cleanup
            # failure, burying the actual reason a billed run failed behind
            # an unrelated lab-reset problem.
            #
            # `already_failing` is read HERE, at the top of `finally` and
            # BEFORE the nested `try` below -- not from inside the nested
            # `except Exception as reset_error:` handler further down. That
            # distinction is the actual fix: `sys.exc_info()` reports the
            # exception the *nearest enclosing* `except` block is currently
            # handling, at the point it is called. Called from inside
            # `except Exception as reset_error:`, the nearest enclosing
            # handler is that very `except` -- `sys.exc_info()` there always
            # describes `reset_error` itself, never an outer exception from
            # this function's own `try` block, so a check placed there could
            # never see past its own just-caught exception. Called here,
            # before the nested `try` exists, the nearest enclosing handler
            # is whatever `finally` is unwinding for -- correctly the outer
            # `try` body's own exception, or `None` if that body succeeded.
            already_failing = sys.exc_info()[0] is not None
            try:
                reset_scenario(root, incident_id)
            except Exception as reset_error:
                if already_failing:
                    print(
                        f"FAIL RESET_SCENARIO_FAILED_DURING_CLEANUP {family}/"
                        f"{incident_id}: {reset_error}"
                    )
                else:
                    raise
    return records


def _new_evaluation_target(root: Path) -> Path:
    """Mints this run's `results/evaluations/<opaque-id>/` directory and
    seeds it with an empty `records.jsonl` before `run_evaluation` ever
    starts -- so `main` has a real, existing path to report even if the
    very first scored run fails before producing anything, and
    `run_evaluation` has a file already in place to keep overwriting as
    real records land.

    Can raise `OSError` (a read-only filesystem, a full disk, a permission
    error) -- `main` guards this call explicitly rather than letting that
    escape uncaught, the same "no raw traceback reaches the owner" posture
    it already keeps for `run_evaluation`'s own failures.
    """
    target = root / "results" / "evaluations" / new_opaque_id()
    target.mkdir(parents=True, exist_ok=True)
    write_jsonl(target / "records.jsonl", [])
    return target


def _range_str(low: float | int | None, high: float | int | None) -> str:
    if low is None or high is None:
        return "no data"
    if isinstance(low, int) and isinstance(high, int):
        return f"{low}-{high}"
    return f"{low:.4f}-{high:.4f}"


def render_evaluation_summary(summary: EvaluationSummary) -> str:
    """Counts and ranges only, one line per figure -- no percentile or mean,
    since a sample this small (a handful of held-out incidents) cannot
    support one honestly (see `EvaluationSummary`'s own docstring).
    `causalops doctor`'s `render_report` is this project's other CLI-summary
    formatter; this follows its "one line per fact, plain labels" shape
    rather than inventing a second style.

    Renders one group's figures only -- `render_paired_evaluation_summary`
    below calls this once per `(arm, retrieval_mode)` group; the trailing
    batch-wide total it appends afterward is a plain record count, not a
    second call into this function (see that function's own docstring for
    why). This function itself knows nothing about arms or retrieval
    modes."""
    total = summary.total_records
    lines = [
        f"evaluation summary: {total} record(s)",
        f"  correct_and_grounded:{summary.correct_and_grounded_count}/"
        f"{summary.citations_sufficient_applicable_count} "
        f"({summary.citations_sufficient_applicable_count}/{total} applicable)",
        f"  diagnosis_correct:   {summary.diagnosis_correct_count}/{total}",
        f"  disposition_correct: {summary.disposition_correct_count}/{total}",
        f"  citations_valid:     {summary.citations_valid_count}/{total}",
        f"  citations_sufficient:{summary.citations_sufficient_count}/"
        f"{summary.citations_sufficient_applicable_count} "
        f"({summary.citations_sufficient_applicable_count}/{total} applicable)",
        f"  scorer_versions: {', '.join(summary.scorer_versions)}",
        "  latency_ms:      "
        f"{_range_str(summary.latency_ms_min, summary.latency_ms_max)}",
        "  model_calls:     "
        f"{_range_str(summary.model_calls_min, summary.model_calls_max)}",
        "  tools_executed:  "
        f"{_range_str(summary.tools_executed_min, summary.tools_executed_max)}",
        "  control.denied:            "
        f"{_range_str(summary.denied_min, summary.denied_max)}",
        "  control.duplicate:         "
        f"{_range_str(summary.duplicate_min, summary.duplicate_max)}",
        "  control.out_of_scope:      "
        f"{_range_str(summary.out_of_scope_min, summary.out_of_scope_max)}",
        "  control.invalid_responses: "
        f"{_range_str(summary.invalid_responses_min, summary.invalid_responses_max)}",
        "  control.unsettled:         "
        f"{_range_str(summary.unsettled_min, summary.unsettled_max)}",
        "  input_tokens:    "
        f"{_range_str(summary.input_tokens_min, summary.input_tokens_max)} "
        f"({summary.input_tokens_known_count}/{total} known)",
        "  output_tokens:   "
        f"{_range_str(summary.output_tokens_min, summary.output_tokens_max)} "
        f"({summary.output_tokens_known_count}/{total} known)",
        "  reserved_usd:    "
        f"{_range_str(summary.reserved_usd_min, summary.reserved_usd_max)}",
        "  actual_usd:      "
        f"{_range_str(summary.actual_usd_min, summary.actual_usd_max)} "
        f"({summary.actual_usd_known_count}/{total} known)",
    ]
    return "\n".join(lines)


def _arm_of(record: EvaluationRecord) -> str:
    """The mode `_run_one` encoded as `run_key`'s final `/`-segment -- see
    `MODE_NO_TOOL_BASELINE`/`MODE_TOOL_ENABLED`'s own comment for why this is
    the one place that decodes it back out."""
    return record.run_key.rsplit("/", 1)[-1]


# `RetrievalMode`'s own declared member order -- used only to give
# `summarize_paired_evaluation`'s group ordering a fixed, deterministic
# sequence within an arm, independent of dict/set iteration order.
_RETRIEVAL_MODE_ORDER: tuple[RetrievalMode, ...] = (
    RetrievalMode.DISABLED,
    RetrievalMode.FTS5_LEXICAL,
    RetrievalMode.PINECONE_SEMANTIC,
)


class EvaluationGroupSummary(BaseModel):
    """One `(arm, retrieval_mode)` group's own batch summary.

    A benchmark aggregate must never silently fall back, mix retrieval
    modes together, or represent FTS5 as semantic retrieval. Partitioning
    summaries by arm alone (`baseline`/`tool_enabled`) is not sufficient,
    because `retrieval_mode` is not reliably coupled to arm. The no-tool
    baseline is always `RetrievalMode.
    DISABLED` structurally: `graph.py`'s baseline path never calls any tool,
    including `search_runbooks`. But within the TOOL-ENABLED arm,
    `retrieval_mode` depends on whether the model actually chose to call
    `search_runbooks` during that specific run (`graph.py`'s
    `dispatch_tool`, which only moves `state["retrieval_mode"]` away from
    its initial `DISABLED` default once a retrieval call actually happens).
    So two tool-enabled runs in the same batch can legitimately carry
    different `retrieval_mode` values, and an arm-only partition can
    silently blend them into one reported figure -- exactly what this rule
    forbids. Partitioning by the `(arm, retrieval_mode)` pair instead means
    every group this class represents came from records that share both
    facts, so no group can mix retrieval modes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    arm: str
    retrieval_mode: RetrievalMode
    summary: EvaluationSummary


class PairedEvaluationSummary(BaseModel):
    """Replaces a fixed three-field `baseline`/`tool_enabled`/`combined`
    shape used previously. `groups` holds one `EvaluationGroupSummary`
    per distinct `(arm, retrieval_mode)` pair actually present in the batch --
    however many distinct pairs that turns out to be (one group for the
    baseline arm, since it is always `DISABLED`; one or two for the
    tool-enabled arm, depending on whether the model ever retrieved) --
    rather than assuming a fixed small set of buckets that a future third
    retrieval mode or a mixed-mode batch could silently overflow.

    A `combined` field alongside `groups` used to report a full
    `EvaluationSummary` (diagnosis/citation/control/latency/cost figures)
    computed across every record regardless of arm or retrieval mode, only
    labeled in the rendered output as spanning every mode. `TECHNICAL_SPEC.
    md` (line ~281)'s "Never silently fall back, mix modes in one benchmark
    aggregate, or represent FTS5 as semantic retrieval" is three separate
    prohibitions joined by "or" -- "silently" grammatically modifies only
    "fall back," not "mix modes in one benchmark aggregate," which is its
    own unconditional rule. A clear label does not change what `combined`
    was: one benchmark aggregate blending records from more than one
    retrieval mode, exactly what that clause forbids regardless of labeling.
    `total_records` replaces it -- a plain count carries no diagnosis,
    citation, control, latency, or cost figure, so it is not a "benchmark
    aggregate" in the sense the spec is protecting against; it implies
    nothing about diagnostic quality or cost under mixed conditions, only
    how many rows the batch produced.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    groups: tuple[EvaluationGroupSummary, ...]
    total_records: int


def summarize_paired_evaluation(
    records: Sequence[EvaluationRecord],
) -> PairedEvaluationSummary:
    """Partitions `records` by the `(arm, retrieval_mode)` pair -- see
    `EvaluationGroupSummary`'s own docstring for why arm alone is not
    enough -- then calls the existing arm/mode-agnostic
    `summarize_evaluation` once per distinct pair present.
    `summarize_evaluation` itself is taught nothing about arms or retrieval
    modes; it predates this script's paired design and stays usable on any
    flat list of records, here or elsewhere. `total_records` is a bare count of
    the whole batch, not a call to `summarize_evaluation` on the unpartitioned
    records -- see `PairedEvaluationSummary`'s own docstring for why no
    benchmark figure may span more than one retrieval mode.

    A record whose `run_key` carries neither known arm word is a data-shape
    bug upstream (a `run_key` no `_run_one` call in this script's history
    could have produced), not a value to silently drop from a scored
    figure -- raising here surfaces that immediately rather than quietly
    under-counting one arm.
    """
    unrecognized = [
        record
        for record in records
        if _arm_of(record) not in (MODE_NO_TOOL_BASELINE, MODE_TOOL_ENABLED)
    ]
    if unrecognized:
        raise ValueError(
            f"{len(unrecognized)} record(s) carry a run_key whose final "
            f"segment is neither {MODE_NO_TOOL_BASELINE!r} nor "
            f"{MODE_TOOL_ENABLED!r} -- cannot partition this batch by arm"
        )

    grouped: dict[tuple[str, RetrievalMode], list[EvaluationRecord]] = {}
    for record in records:
        key = (_arm_of(record), record.retrieval_mode)
        grouped.setdefault(key, []).append(record)

    def _sort_key(key: tuple[str, RetrievalMode]) -> tuple[int, int]:
        arm, mode = key
        arm_rank = 0 if arm == MODE_NO_TOOL_BASELINE else 1
        return (arm_rank, _RETRIEVAL_MODE_ORDER.index(mode))

    groups = tuple(
        EvaluationGroupSummary(
            arm=arm, retrieval_mode=mode, summary=summarize_evaluation(group_records)
        )
        for (arm, mode), group_records in sorted(
            grouped.items(), key=lambda item: _sort_key(item[0])
        )
    )
    return PairedEvaluationSummary(groups=groups, total_records=len(records))


def render_paired_evaluation_summary(paired: PairedEvaluationSummary) -> str:
    """One block per `(arm, retrieval_mode)` group, then a plain batch-wide
    record count -- never a benchmark figure that spans more than one
    retrieval mode. Each group's own label states both facts it was
    partitioned on, so two tool-enabled groups that differ only in
    retrieval mode -- exactly the blending this project's evaluation
    reporting forbids -- render as visibly separate blocks, never one merged
    number. The trailing total is a count only, not a call into
    `render_evaluation_summary` (which reports diagnosis/citation/control/
    latency/cost figures) -- see `PairedEvaluationSummary`'s own docstring
    for why even a clearly labeled version of that figure is still the
    thing this project's evaluation reporting forbids."""
    blocks = [
        f"[{group.arm}, retrieval_mode={group.retrieval_mode.value}]\n"
        f"{render_evaluation_summary(group.summary)}"
        for group in paired.groups
    ]
    blocks.append(
        f"total_records (all arms and retrieval modes): {paired.total_records}"
    )
    return "\n\n".join(blocks)


def _write_json_atomic(path: Path, payload: BaseModel) -> None:
    """Writes `payload` to `path` in one atomic replace, never a
    truncate-then-write in place -- the same pattern `run_records.
    write_jsonl` already uses for `records.jsonl`, applied here to the
    single-object `summary.json`. A plain `path.write_text` truncates its
    target before writing a byte of the new content, so a hard process
    kill mid-write -- not a catchable `OSError`, so no `except` clause below
    ever runs to warn about it -- could leave `path` as a truncated,
    corrupted file on disk. Building the complete content in a sibling
    temporary file first, then atomically renaming it onto `path`
    (`Path.replace`, atomic on POSIX), means a crash before the rename
    leaves no file at all where none existed before, never a corrupted one.
    """
    content = payload.model_dump_json(indent=2)
    tmp_path = path.with_name(f"{path.name}.tmp-{uuid4().hex}")
    try:
        tmp_path.write_text(content, encoding="utf-8")
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    start = Path.cwd()
    root = find_project_root(start)
    if root is None:
        print(f"FAIL PROJECT_ROOT_NOT_FOUND No pyproject.toml at or above {start}.")
        return 1
    if not os.environ.get(API_KEY_VARIABLE, "").strip():
        print("FAIL MISSING_API_KEY Set ANTHROPIC_API_KEY before a live evaluation.")
        return 1
    try:
        target = _new_evaluation_target(root)
    except OSError as error:
        # No `records_path` to report here -- target creation is what would
        # have produced it, and it failed, so there is genuinely nothing on
        # disk yet to point the owner at.
        print(f"FAIL EVALUATION_TARGET_UNWRITABLE {error}")
        return 1
    records_path = target / "records.jsonl"
    try:
        records = run_evaluation(root, target)
    except EvaluationAborted as aborted:
        print(f"FAIL EVALUATION_ABORTED {aborted.reason.value}")
        print(f"records so far: {records_path}")
        return 1
    except (LabError, RunRecordError, CheckpointStoreError) as refusal:
        # `CheckpointStoreError` reaches here from `live_setup.
        # live_evaluation_ceiling_usd` (called both directly by
        # `run_evaluation` and indirectly through `build_model_and_registry`
        # in `_run_one`) whenever `LIVE_EVALUATION_MAX_USD` is configured but
        # unusable -- malformed, non-finite, or too small to ever authorize
        # a reservation. Without this in the typed tuple, that refusal fell
        # through to the generic `except Exception` below and reported the
        # opaque `FAIL INTERNAL_ERROR` instead of its own stable reason
        # code, contradicting `.env.example`'s documented `FAIL
        # CEILING_BELOW_RESERVATION_BUFFER`/`FAIL CEILING_MALFORMED` output.
        # `cli.py`'s `main` already catches this type alongside the same two
        # exceptions for its own `investigate`/`approve`/`reject` commands.
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
    # Partitioned by arm, not one blended total -- see
    # `PairedEvaluationSummary`'s own docstring for why a single combined
    # figure cannot show the paired comparison this evaluation exists for.
    summary = summarize_paired_evaluation(records)
    print(render_paired_evaluation_summary(summary))
    summary_path = target / "summary.json"
    try:
        # A crash mid-write here cannot destroy any EARLIER content -- this
        # is the first and only write to `summary_path` -- but it can still
        # leave `summary.json` itself half-written on disk: a hard process
        # kill during `write_text` is not a catchable `OSError`, so nothing
        # below would run to clean it up, and a reader would see a truncated
        # file with no exception ever having fired to warn them. `_write_
        # json_atomic` (below) closes that gap the same way `write_jsonl`
        # already does for `records.jsonl`: build the complete content in a
        # sibling temp file, then atomically rename it onto the real path,
        # so a crash before the rename leaves no file at all where none
        # existed before, never a corrupted one.
        _write_json_atomic(summary_path, summary)
    except OSError as error:
        # The same "no raw traceback reaches the owner" posture
        # `_new_evaluation_target`'s own `OSError` guard already keeps,
        # applied to this later write site (disk full, permission error)
        # instead of leaving it to escape uncaught. Every real, billed
        # result is already safe in `records_path` by this point -- eight
        # calls' worth of work is not lost just because the summary itself
        # could not be written, so this reports that explicitly rather than
        # letting the owner wonder.
        print(f"FAIL SUMMARY_WRITE_FAILED {error}")
        print(f"records (unaffected): {records_path}")
        return 1
    print(f"summary: {summary_path}")
    print(f"records: {records_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
