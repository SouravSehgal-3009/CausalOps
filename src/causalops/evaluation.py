"""Evaluator-only expected outcomes and mechanical scoring.

This module holds ground truth: expected causes and the predicates a good answer
must satisfy. No investigator module imports it, and none of its values reach a
model context.
"""

from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, JsonValue

from causalops.domain import (
    SCHEMA_VERSION,
    Disposition,
    Evidence,
    EvidenceKind,
    InvestigationReport,
    PolicyResult,
    ReasonCode,
    ReceiptState,
    RetrievalMode,
    RootCauseCode,
    ToolReceipt,
    Versions,
)

SCORER_VERSION = "4"

OUT_OF_SCOPE_REASONS = frozenset(
    {
        ReasonCode.CROSS_INCIDENT_REQUEST,
        ReasonCode.UNKNOWN_SERVICE,
        ReasonCode.OUTSIDE_INCIDENT_WINDOW,
        ReasonCode.RESULT_LIMIT_EXCEEDED,
    }
)


class PredicateOperator(StrEnum):
    EQUALS = "EQUALS"
    AT_LEAST = "AT_LEAST"
    CONTAINS = "CONTAINS"


class RequiredEvidencePredicate(BaseModel):
    """One observable fact a cited answer has to rest on."""

    model_config = ConfigDict(frozen=True)

    source: str
    kind: EvidenceKind
    template: str | None = None
    field: str
    operator: PredicateOperator
    value: JsonValue


class ExpectedOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    root_cause: RootCauseCode
    disposition: Disposition
    predicates: tuple[RequiredEvidencePredicate, ...] = ()


class ControlCounts(BaseModel):
    model_config = ConfigDict(frozen=True)

    denied: int = 0
    duplicate: int = 0
    out_of_scope: int = 0
    invalid_responses: int = 0
    # A check that spent budget and never got a result -- a crash between
    # `ReservationLedger.reserve()` and `.settle()` (see `tool_wrappers.py`).
    # `count_control` below counts this as `state is ReceiptState.RESERVED`,
    # not the more cautious `policy_result is ALLOWED and state is RESERVED`
    # it might look like it should be: `tool_wrappers.py:142-153` constructs
    # every `RESERVED` receipt with `policy_result=ALLOWED` unconditionally,
    # and `record()` (the only path that ever writes a `DENIED` receipt,
    # `tool_wrappers.py:205-222`) refuses anything not already `SETTLED`. No
    # `DENIED` receipt can be `RESERVED` by construction, so the simpler
    # predicate is not an approximation -- it is the same set.
    unsettled: int = 0


class Efficiency(BaseModel):
    model_config = ConfigDict(frozen=True)

    latency_ms: int
    model_calls: int
    tools_executed: int
    input_tokens: int | None = None
    output_tokens: int | None = None


class MechanicalScores(BaseModel):
    model_config = ConfigDict(frozen=True)

    diagnosis_correct: bool
    disposition_correct: bool
    citations_valid: bool
    # Unlike `diagnosis_correct` above, this question is not always
    # well-posed. `diagnosis_correct` always has a definite answer because
    # every `ExpectedOutcome` names exactly one `root_cause` to compare
    # against. `citations_sufficient` asks whether the cited evidence
    # satisfies every predicate in `expected.predicates` -- and a family can
    # legitimately declare none, even though every family in this corpus
    # today does (a future family could still ship with an empty set).
    # `all(())` is `True` in Python, so a bare `bool` here silently scored
    # every such run `citations_sufficient=True`, including a run with a
    # wrong diagnosis and nothing cited -- there was no predicate to fail.
    # `None` is the honest third value: "this family declares no
    # required-evidence predicate, not applicable" is a different fact from
    # "the predicates it declared were satisfied" (`True`) or "were not"
    # (`False`), and collapsing it into either would misreport which one
    # actually happened.
    citations_sufficient: bool | None
    # `diagnosis_correct AND citations_sufficient`, restated below because
    # neither field alone answers it: a run can cite grounded, predicate-
    # satisfying evidence for the WRONG root cause (`citations_sufficient`
    # true, `diagnosis_correct` false), or reach the right root cause without
    # citing the evidence that actually justifies it (`diagnosis_correct`
    # true, `citations_sufficient` false). `None` under the identical
    # no-predicate condition `citations_sufficient` itself uses -- never
    # computed independently, so the two fields can never disagree on
    # applicability.
    #
    # `= None` here, unlike `citations_sufficient` above: this is a
    # brand-new field (F5/F6), so a pre-F6 historical `records.jsonl` line
    # never wrote this key at all -- the default lets it validate as `None`
    # on read rather than fail, matching how the value is scored anyway on
    # any record this old. See `test_evaluation.py`'s
    # `test_a_pre_f6_record_missing_correct_and_grounded_still_summarizes_
    # correctly` for the read-compatibility proof.
    #
    # A direct reader of `records.jsonl` -- one that reads this field
    # itself rather than going through `summarize_evaluation` (which always
    # re-derives it fresh and is unaffected by this) -- must not treat
    # `None` here as always meaning "not applicable". A record with
    # `scorer_version` below `"4"` and a non-empty `expected.predicates`
    # can also read back `None`, and there it means only that this field
    # did not exist yet when that record was scored, not that it was
    # deliberately marked not-applicable. Such a reader should re-derive
    # the value the same way `summarize_evaluation` does: `None if not
    # expected.predicates else diagnosis_correct and citations_sufficient
    # is True`. (`scorer_version` strings are compared lexically elsewhere
    # in this module, which only stays correct through single digits -- a
    # minor pre-existing caveat, not specific to this field.)
    correct_and_grounded: bool | None = None
    control: ControlCounts
    efficiency: Efficiency


class EvaluationRecord(BaseModel):
    """One scored run of the Unit 3c paired live comparison.

    The reproducibility fields below are `TECHNICAL_SPEC.md` §10's own list
    verbatim ("Record Git SHA, clean/dirty status, fixture/prompt/policy/
    tool versions, retrieval mode/corpus version, exact model, tokens,
    latency, cost, and raw artifact references. Include the pricing
    source/date and configured ceiling."). Tokens and latency are already
    on `scores.efficiency`, so they are not repeated here. "Raw artifact
    references" is `investigation_id` itself, not a separate field:
    `run_records.finalize_investigation` already writes every raw artifact
    for a run to a deterministic path this id alone names
    (`results/investigations/<investigation_id>/`), so a second pointer
    field would only ever repeat what `investigation_id` already says.

    `extra="forbid"` matches the project-wide tightening every other
    wire-facing model already carries (Unit `single-turn-tool-protocol`):
    this record is written to and read back from disk the same way those
    models are, so an unrecognized field on read is a real, actionable
    surprise, not something to silently drop.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = SCHEMA_VERSION
    scorer_version: str = SCORER_VERSION
    run_key: str
    investigation_id: str
    incident_id: str
    expected: ExpectedOutcome
    scores: MechanicalScores

    # Reproducibility manifest, §10.
    git_sha: str
    git_dirty: bool
    versions: Versions
    retrieval_mode: RetrievalMode
    # `RunbookIndex.corpus_version`, `None` only for a corpus file that
    # predates the `corpus_version` key -- see that class's own docstring.
    runbook_corpus_version: str | None = None
    # SHA-256 of the exact `lab/scenarios/<family>.json` bytes this
    # incident's family was started from -- a content hash rather than a
    # hand-maintained version string, so it cannot silently drift from what
    # was actually used and needs no change to the frozen scenario files
    # themselves.
    fixture_sha256: str
    model_name: str
    pricing_source: str
    pricing_verified_on: str
    configured_ceiling_usd: float
    # From `cost_ledger.run_cost_totals` for this run's `run_id` -- the
    # existing reservation/settlement ledger, not a second cost-tracking
    # mechanism. `actual_usd` is `None` only for a run whose model call(s)
    # never settled (reserved, then a crash/timeout/refusal before the
    # provider responded) -- an honest partial-cost case, not hidden as 0.0.
    reserved_usd: float
    actual_usd: float | None = None
    # A failed-safe graph result is still a scored, durable evaluation record.
    # The reason lets batch orchestration distinguish infrastructure failures
    # (which must stop further paid requests) from model-quality outcomes.
    failure_reason: ReasonCode | None = None


def satisfies(predicate: RequiredEvidencePredicate, evidence: Evidence) -> bool:
    if evidence.kind is not predicate.kind or evidence.source != predicate.source:
        return False
    if (
        predicate.template is not None
        and evidence.payload.get("template") != predicate.template
    ):
        return False
    if predicate.field not in evidence.payload:
        return False
    observed = evidence.payload[predicate.field]
    if predicate.operator is PredicateOperator.EQUALS:
        return bool(observed == predicate.value)
    if predicate.operator is PredicateOperator.AT_LEAST:
        return (
            isinstance(observed, int | float)
            and isinstance(predicate.value, int | float)
            and observed >= predicate.value
        )
    if predicate.operator is PredicateOperator.CONTAINS:
        return (
            isinstance(observed, str)
            and isinstance(predicate.value, str)
            and predicate.value in observed
        )
    return False


def cited_evidence(
    report: InvestigationReport, evidence: Sequence[Evidence]
) -> tuple[Evidence, ...]:
    """Evidence records the report's assessment actually cited as
    SUPPORTING its diagnosis, scoped to THIS incident -- the same
    `record.incident_id == report.incident_id` filter `citations_are_valid`'s
    own `known` set below already applies.

    Deliberately excludes `contrary_evidence_ids`. This function's only
    caller is `score_run`'s `citations_sufficient` check, which asks whether
    the diagnosis actually rests on real evidence FOR it. Evidence the model
    itself filed as contrary -- working against its own diagnosis, not for
    it -- must not be able to satisfy that question just because it happens
    to match a required predicate's field content. Before this narrowing,
    `cited` combined both lists into one set, so a diagnosis whose only
    predicate-matching evidence was filed under `contrary_evidence_ids`
    still scored `citations_sufficient=True` -- backwards, since that
    evidence is the model's own record of what argued against the
    diagnosis. `citations_are_valid` below intentionally keeps validating
    both lists together: whether a citation is a real, same-incident record
    is a different question from which side it argues, and a contrary
    citation should still be checked for existence.

    Before the incident-id filter (an earlier round's fix), an
    `evidence_id` match alone was enough regardless of which incident a
    record belonged to, so `citations_sufficient` could be satisfied by a
    cross-incident record even on a report `citations_are_valid` correctly
    refused for exactly that citation. The isolation boundary held only
    because the graph never actually handed this scorer cross-incident
    evidence in practice, not because the scorer enforced it itself. That
    filter closes that gap: `citations_sufficient` can no longer be
    satisfied by a record this function has already excluded.

    Deliberately NOT also gating `citations_sufficient` on `citations_valid`
    in `score_run`: that would conflate two different questions. A report
    can cite one genuine, sufficient, same-incident piece of evidence
    alongside one unrelated bad id (a typo, a stale id from an earlier
    turn), and `citations_valid` correctly reports `False` for that extra
    bad reference -- but the predicate really is satisfied by real evidence,
    which `citations_sufficient=True` says honestly. Requiring
    `citations_valid` first would hide that real signal behind an unrelated
    citation's problem instead of reporting both facts plainly.
    """
    if report.assessment is None:
        return ()
    supporting = set(report.assessment.supporting_evidence_ids)
    return tuple(
        record
        for record in evidence
        if record.evidence_id in supporting and record.incident_id == report.incident_id
    )


def citations_are_valid(
    report: InvestigationReport, evidence: Sequence[Evidence]
) -> bool:
    """Every cited ID exists and belongs to this incident."""
    if report.assessment is None:
        return False
    known = {
        record.evidence_id
        for record in evidence
        if record.incident_id == report.incident_id
    }
    cited = (
        report.assessment.supporting_evidence_ids
        + report.assessment.contrary_evidence_ids
    )
    return all(evidence_id in known for evidence_id in cited)


def count_control(
    report: InvestigationReport, receipts: Sequence[ToolReceipt]
) -> ControlCounts:
    denied = [
        receipt for receipt in receipts if receipt.policy_result is PolicyResult.DENIED
    ]
    return ControlCounts(
        denied=len(denied),
        duplicate=len(
            [
                denial
                for denial in denied
                if denial.reason_code is ReasonCode.DUPLICATE_PROPOSAL
            ]
        ),
        out_of_scope=len(
            [denial for denial in denied if denial.reason_code in OUT_OF_SCOPE_REASONS]
        ),
        invalid_responses=report.invalid_responses,
        unsettled=len(
            [receipt for receipt in receipts if receipt.state is ReceiptState.RESERVED]
        ),
    )


def score_run(
    report: InvestigationReport,
    evidence: Sequence[Evidence],
    receipts: Sequence[ToolReceipt],
    expected: ExpectedOutcome,
) -> MechanicalScores:
    supported = cited_evidence(report, evidence)
    # `diagnosis_correct` and `disposition_correct` deliberately answer
    # different questions. `diagnosis_correct` is a factual property of the
    # root cause the model proposed, independent of what the owner did with
    # it -- conflating the two would make a rejected-but-correct diagnosis
    # indistinguishable from a rejected-and-wrong one, losing exactly the
    # signal a diagnostic-quality evaluation needs. `disposition_correct`
    # answers whether the disposition this run actually settled on was one
    # worth acting on -- and an owner **rejection** overrides that, however
    # accurate the underlying assessment was, since `report.disposition`
    # itself is left unchanged by a reject (`graph.py:301-309`'s
    # `_build_report` computes `disposition` from `assessment` alone and
    # never consults `escalation` -- rejection deliberately preserves the
    # assessment, but that is this function's own behaviour, not a claim
    # `EscalationRecord`'s docstring makes).
    rejected = report.escalation is not None and report.escalation.decision == "reject"
    # `diagnosis_correct` requires a genuine model assessment, not just a
    # matching `root_cause` value. `FAILED_SAFE`'s own `root_cause` defaults
    # to `UNDETERMINED` (`InvestigationReport.check_terminal_invariants`
    # forbids anything else), and `ambiguous_telemetry`'s own expected root
    # cause is ALSO `UNDETERMINED` -- it is the one family in this corpus
    # designed to be genuinely inconclusive. A bare `report.root_cause is
    # expected.root_cause` comparison would let a run that crashed before
    # producing any real diagnosis at all collide with that expectation and
    # score a total failure as a correct answer. `report.assessment is not
    # None` is the exact condition: it is `True` for both `DIAGNOSED` and a
    # genuine `INSUFFICIENT_EVIDENCE` abstention (both require a real
    # `FinalAssessment`) and `False` only for `FAILED_SAFE` (which requires
    # `assessment is None`), so gating on it changes nothing for either kind
    # of real model output and excludes only the crash case.
    #
    # `False`, not `None`: unlike `EvaluationRecord.actual_usd`'s `None`
    # (a real number that is honestly unmeasured because the run kept going
    # and only its final settlement is unknown), a `FAILED_SAFE` run never
    # produced a diagnosis to evaluate in the first place -- "did this run's
    # output count as a correct diagnosis" has a definite answer here, and
    # that answer is no. Keeping `MechanicalScores.diagnosis_correct` a
    # plain `bool` also means `EvaluationSummary`'s existing
    # `diagnosis_correct_count = sum(...)` needs no change to keep meaning
    # what it already says.
    diagnosis_correct = (
        report.assessment is not None and report.root_cause is expected.root_cause
    )
    # `None` when the family declares no required-evidence predicate at
    # all -- see `MechanicalScores.citations_sufficient`'s own comment
    # for why `all(())` being `True` made this the exact vacuous-truth
    # bug being fixed here (an empty `expected.predicates` tuple used to
    # score `True` unconditionally, including on a wrong diagnosis).
    citations_sufficient = (
        None
        if not expected.predicates
        else all(
            any(satisfies(predicate, record) for record in supported)
            for predicate in expected.predicates
        )
    )
    return MechanicalScores(
        diagnosis_correct=diagnosis_correct,
        disposition_correct=report.disposition is expected.disposition and not rejected,
        citations_valid=citations_are_valid(report, evidence),
        citations_sufficient=citations_sufficient,
        # Gated on the identical `expected.predicates` condition as
        # `citations_sufficient` above -- computed from that same local, not
        # a separately re-derived expression -- so the two fields can never
        # disagree on which records are applicable.
        correct_and_grounded=(
            None
            if not expected.predicates
            else diagnosis_correct and citations_sufficient
        ),
        control=count_control(report, receipts),
        efficiency=Efficiency(
            latency_ms=report.latency_ms,
            model_calls=report.model_calls_used,
            tools_executed=report.tools_executed,
            input_tokens=report.usage.input_tokens if report.usage else None,
            output_tokens=report.usage.output_tokens if report.usage else None,
        ),
    )


class EvaluationSummary(BaseModel):
    """Batch-level counts and ranges across every `EvaluationRecord` in one
    `causalops-evaluate` run.

    `TECHNICAL_SPEC.md` §10: "Report counts and ranges for small samples; do
    not report p95 or broad performance claims from a small synthetic
    benchmark." This model deliberately has no percentile, mean, or standard
    deviation field -- a count and a min-max range are what a corpus this
    size (at most eight records, `EVALUATION_FAMILIES` in `evaluate_cli.py`)
    can honestly support, and adding a statistical field here would invite
    reporting it later even though the sample never grew to support it.

    Every `*_min`/`*_max` pair is `None` only when `total_records == 0` (an
    empty batch, not a realistic outcome of a successful run but a real
    input to this function) or, for `input_tokens`/`output_tokens`/
    `actual_usd`, when none of the records carry a value at all -- the
    corresponding `*_known_count` says how many of `total_records` did, so
    a reader can tell "no data" from "zero-width range" apart from "some
    unknown."
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    total_records: int
    # A run counts here only if it both reached the right root cause AND
    # cited evidence satisfying every required predicate -- the joint
    # question neither `diagnosis_correct_count` nor
    # `citations_sufficient_count` answers alone. Gated on the same
    # `record.expected.predicates` condition as `citations_sufficient_count`
    # below, so it shares that count's own `citations_sufficient_applicable_
    # count` denominator rather than a separate `*_applicable_count` field --
    # see `summarize_evaluation`'s own comment on this computation.
    correct_and_grounded_count: int
    diagnosis_correct_count: int
    disposition_correct_count: int
    # `TECHNICAL_SPEC.md` §10 lists "citation validity and citation
    # sufficiency against required-evidence predicates" as its own required
    # mechanical score, alongside diagnosis/disposition -- these were
    # missing entirely before this fix. `citations_valid_count` is counted
    # the same way as `diagnosis_correct_count`/`disposition_correct_count`
    # above: a simple per-record boolean sum of `MechanicalScores.
    # citations_valid`, so a batch total is the same kind of number, not a
    # min/max range. `citations_sufficient_count` below is not counted the
    # same way -- see its own comment.
    citations_valid_count: int
    # `citations_sufficient_count` is how many records had a predicate to
    # check (`record.expected.predicates` non-empty) AND scored `True`.
    # `citations_sufficient_applicable_count` is the real denominator: how
    # many records had a predicate to check at all. Both counts are gated on
    # the same `expected.predicates` condition so the numerator can never
    # exceed the denominator -- see `summarize_evaluation`'s own comment on
    # this computation for why a record's `scores.citations_sufficient is
    # not None` alone isn't used instead. Reporting `citations_sufficient_
    # count` against `total_records` alone would silently credit a
    # no-predicate family as checked and passed; `applicable_count` lets a
    # reader see the real denominator, the same way `input_tokens_known_
    # count` separates "no data" from "zero-width range" for a different
    # field.
    citations_sufficient_count: int
    citations_sufficient_applicable_count: int

    # The distinct `EvaluationRecord.scorer_version` values present in this
    # batch, sorted for a deterministic rendering order. Not a validity
    # check: this project deliberately reports a mixed-version batch rather
    # than rejecting it (see `summarize_evaluation`'s own comment on this
    # field for why), so a reader can see at a glance whether every record
    # was scored under the same scorer, without the batch itself being
    # unsummarizable when it wasn't.
    scorer_versions: tuple[str, ...]

    latency_ms_min: int | None = None
    latency_ms_max: int | None = None
    model_calls_min: int | None = None
    model_calls_max: int | None = None
    tools_executed_min: int | None = None
    tools_executed_max: int | None = None

    # `TECHNICAL_SPEC.md` §10's other missing required score: "policy/control
    # behavior." Each `ControlCounts` field (`evaluation.py`'s own
    # `count_control`) is already a per-record integer, the same shape as
    # `tools_executed` above -- a min/max range across the batch, not a
    # single summed total, follows this file's existing "counts and ranges
    # only" discipline for that kind of per-run scalar rather than
    # introducing a third aggregation style.
    denied_min: int | None = None
    denied_max: int | None = None
    duplicate_min: int | None = None
    duplicate_max: int | None = None
    out_of_scope_min: int | None = None
    out_of_scope_max: int | None = None
    invalid_responses_min: int | None = None
    invalid_responses_max: int | None = None
    unsettled_min: int | None = None
    unsettled_max: int | None = None

    input_tokens_min: int | None = None
    input_tokens_max: int | None = None
    input_tokens_known_count: int
    output_tokens_min: int | None = None
    output_tokens_max: int | None = None
    output_tokens_known_count: int

    reserved_usd_min: float | None = None
    reserved_usd_max: float | None = None
    # `actual_usd` is `None` on any `EvaluationRecord` whose run did not
    # fully settle (`cost_ledger.run_cost_totals`'s `fully_settled` --
    # `evaluate_cli.py`'s `_run_one`). `actual_usd_known_count` reports how
    # many of `total_records` have a real, complete figure, so a reader
    # never mistakes "some runs' cost is unknown" for "every run cost
    # nothing."
    actual_usd_min: float | None = None
    actual_usd_max: float | None = None
    actual_usd_known_count: int


def _min_max[NumberT: (int, float)](
    values: Sequence[NumberT],
) -> tuple[NumberT, NumberT] | tuple[None, None]:
    if not values:
        return None, None
    return min(values), max(values)


def summarize_evaluation(records: Sequence[EvaluationRecord]) -> EvaluationSummary:
    """Counts and ranges only -- see `EvaluationSummary`'s own docstring for
    why no percentile or mean is computed here."""
    input_tokens_known = [
        record.scores.efficiency.input_tokens
        for record in records
        if record.scores.efficiency.input_tokens is not None
    ]
    output_tokens_known = [
        record.scores.efficiency.output_tokens
        for record in records
        if record.scores.efficiency.output_tokens is not None
    ]
    actual_usd_known = [
        record.actual_usd for record in records if record.actual_usd is not None
    ]
    latency_min, latency_max = _min_max(
        [record.scores.efficiency.latency_ms for record in records]
    )
    calls_min, calls_max = _min_max(
        [record.scores.efficiency.model_calls for record in records]
    )
    tools_min, tools_max = _min_max(
        [record.scores.efficiency.tools_executed for record in records]
    )
    denied_min, denied_max = _min_max(
        [record.scores.control.denied for record in records]
    )
    duplicate_min, duplicate_max = _min_max(
        [record.scores.control.duplicate for record in records]
    )
    out_of_scope_min, out_of_scope_max = _min_max(
        [record.scores.control.out_of_scope for record in records]
    )
    invalid_responses_min, invalid_responses_max = _min_max(
        [record.scores.control.invalid_responses for record in records]
    )
    unsettled_min, unsettled_max = _min_max(
        [record.scores.control.unsettled for record in records]
    )
    input_tokens_min, input_tokens_max = _min_max(input_tokens_known)
    output_tokens_min, output_tokens_max = _min_max(output_tokens_known)
    reserved_min, reserved_max = _min_max([record.reserved_usd for record in records])
    actual_min, actual_max = _min_max(actual_usd_known)
    return EvaluationSummary(
        total_records=len(records),
        # Re-derived fresh from `diagnosis_correct`/`citations_sufficient`
        # here, never from a record's own stored `scores.correct_and_
        # grounded` field -- a historical record saved before this field
        # existed reads back with `correct_and_grounded=None` (a plain
        # Pydantic default, not a migration), and trusting that stale/absent
        # value for the summary count would either undercount it or require
        # a second special case. Deriving fresh instead means this count is
        # correct for every record regardless of which scorer version wrote
        # it, matching `citations_sufficient_count`'s own established
        # discipline exactly.
        correct_and_grounded_count=sum(
            1
            for record in records
            if record.expected.predicates
            and record.scores.diagnosis_correct
            and record.scores.citations_sufficient is True
        ),
        diagnosis_correct_count=sum(
            1 for record in records if record.scores.diagnosis_correct
        ),
        disposition_correct_count=sum(
            1 for record in records if record.scores.disposition_correct
        ),
        citations_valid_count=sum(
            1 for record in records if record.scores.citations_valid
        ),
        # Applicability is derived from `record.expected.predicates` itself,
        # not from `record.scores.citations_sufficient is not None`. For any
        # record this pipeline produces today, the two conditions agree:
        # `score_run` already sets `citations_sufficient=None` exactly when
        # `expected.predicates` is empty, so `is not None` and
        # `bool(expected.predicates)` pick out the same records. The
        # difference only matters for a HISTORICAL record saved under the
        # pre-F4 scorer (`SCORER_VERSION == "2"`), which could still carry a
        # stale `citations_sufficient=True` on an empty-predicate family --
        # deriving applicability from the record's own `expected.predicates`
        # instead re-applies F4's fix on read, so summarizing an old record
        # doesn't reproduce the exact vacuous-truth bug F4 exists to close.
        # Both counts are gated on the SAME condition deliberately: gating
        # only the denominator would let a stale `True` still inflate the
        # numerator while the record is excluded from the denominator,
        # producing a numerator that can exceed it.
        citations_sufficient_count=sum(
            1
            for record in records
            if record.expected.predicates and record.scores.citations_sufficient is True
        ),
        citations_sufficient_applicable_count=sum(
            1 for record in records if record.expected.predicates
        ),
        # Reported, not enforced -- see `EvaluationSummary.scorer_versions`'s
        # own comment. After the fix above, a v2 and a v3 record differ only
        # in how a stale `citations_sufficient=True` on an empty-predicate
        # record is *counted*, not in any field they carry, so mixing them
        # in one batch is no longer a correctness hazard the way it would
        # have been before this fix.
        scorer_versions=tuple(sorted({record.scorer_version for record in records})),
        latency_ms_min=latency_min,
        latency_ms_max=latency_max,
        model_calls_min=calls_min,
        model_calls_max=calls_max,
        tools_executed_min=tools_min,
        tools_executed_max=tools_max,
        denied_min=denied_min,
        denied_max=denied_max,
        duplicate_min=duplicate_min,
        duplicate_max=duplicate_max,
        out_of_scope_min=out_of_scope_min,
        out_of_scope_max=out_of_scope_max,
        invalid_responses_min=invalid_responses_min,
        invalid_responses_max=invalid_responses_max,
        unsettled_min=unsettled_min,
        unsettled_max=unsettled_max,
        input_tokens_min=input_tokens_min,
        input_tokens_max=input_tokens_max,
        input_tokens_known_count=len(input_tokens_known),
        output_tokens_min=output_tokens_min,
        output_tokens_max=output_tokens_max,
        output_tokens_known_count=len(output_tokens_known),
        reserved_usd_min=reserved_min,
        reserved_usd_max=reserved_max,
        actual_usd_min=actual_min,
        actual_usd_max=actual_max,
        actual_usd_known_count=len(actual_usd_known),
    )
