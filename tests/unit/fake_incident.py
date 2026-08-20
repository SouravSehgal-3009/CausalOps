"""One synthetic incident the unit tests share, plus doubles for the model and checks.

The IDs here stand in for what the scenario controller will produce in Step 3.
"""

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import NamedTuple

from pydantic import BaseModel, JsonValue

from causalops.domain import (
    CheckOutcome,
    Evidence,
    EvidenceKind,
    FinalAssessment,
    GatewaySymptom,
    Hypothesis,
    HypothesisUpdate,
    IncidentScope,
    InitialAlertPacket,
    InitialPlan,
    ModelDisposition,
    ModelUsage,
    ReasonCode,
    RootCauseCode,
    RunCheck,
    ToolOutcome,
    ToolProposal,
)
from causalops.evidence import content_hash
from causalops.models import ModelRequest, ModelResponse, ReplayReasoningModel
from causalops.prometheus import MAX_METRIC_SAMPLES
from causalops.telemetry import RunPaths
from causalops.tool_wrappers import ToolWrapper, dispatch_registry
from causalops.tools import (
    GetTopologyArguments,
    ListRecentChangesArguments,
    LogFilter,
    MetricTemplate,
    QueryLogsArguments,
    QueryMetricArguments,
    ToolName,
)

FIXTURE_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "causalops" / "replay_fixtures"
)

INCIDENT_ID = "0f9c2b7e4a1d4f0b8c6e5d3a2b1c0d9e"
SYMPTOM_EVIDENCE_ID = "5c1d0a9b8e7f6a5b4c3d2e1f0a9b8c7d"
TOPOLOGY_EVIDENCE_ID = "9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d"
WINDOW_START = datetime(2026, 8, 16, 10, 0, tzinfo=UTC)
WINDOW_END = datetime(2026, 8, 16, 10, 30, tzinfo=UTC)
SERVICES = ("gateway", "orders", "inventory")


def incident_scope() -> IncidentScope:
    return IncidentScope(
        incident_id=INCIDENT_ID,
        services=SERVICES,
        started_at=WINDOW_START,
        ended_at=WINDOW_END,
        endpoint="/api/orders",
    )


def alert_packet() -> InitialAlertPacket:
    return InitialAlertPacket(
        incident_id=INCIDENT_ID,
        window_start=WINDOW_START,
        window_end=WINDOW_END,
        endpoint="/api/orders",
        symptom=GatewaySymptom.ELEVATED_LATENCY,
        services=SERVICES,
        alerted_at=WINDOW_START,
        alert_source_version="alert-1",
        symptom_evidence_id=SYMPTOM_EVIDENCE_ID,
        topology_evidence_id=TOPOLOGY_EVIDENCE_ID,
    )


def packet_evidence() -> tuple[Evidence, Evidence]:
    symptom_payload: dict[str, JsonValue] = {"symptom": "ELEVATED_LATENCY"}
    topology_payload: dict[str, JsonValue] = {
        "edges": ["gateway>orders", "orders>inventory"]
    }
    return (
        Evidence(
            evidence_id=SYMPTOM_EVIDENCE_ID,
            incident_id=INCIDENT_ID,
            kind=EvidenceKind.SYMPTOM,
            source="alert",
            observed_at=WINDOW_START,
            summary="gateway latency is elevated on /api/orders",
            payload=symptom_payload,
            content_hash=content_hash(symptom_payload),
        ),
        Evidence(
            evidence_id=TOPOLOGY_EVIDENCE_ID,
            incident_id=INCIDENT_ID,
            kind=EvidenceKind.TOPOLOGY,
            source="alert",
            observed_at=WINDOW_START,
            summary="gateway calls orders, orders calls inventory",
            payload=topology_payload,
            content_hash=content_hash(topology_payload),
        ),
    )


def write_log(
    paths: RunPaths, rows: list[dict[str, object]], service: str = "orders"
) -> None:
    paths.logs.mkdir(parents=True, exist_ok=True)
    lines = "".join(json.dumps(row) + "\n" for row in rows)
    (paths.logs / f"{service}.jsonl").write_text(lines, encoding="utf-8")


def log_row(
    offset: int,
    severity: str = "error",
    event: str = "config_rejected_request",
    detail: str = "x",
    service: str = "orders",
) -> dict[str, object]:
    return {
        "at": (WINDOW_START + timedelta(seconds=offset)).isoformat(),
        "request_id": f"r{offset}",
        "service": service,
        "severity": severity,
        "event": event,
        "fields": {"config_key": "require_order_token", "detail": detail},
    }


def write_changes(paths: RunPaths, entries: list[dict[str, object]]) -> None:
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.changes_file.write_text(json.dumps(entries), encoding="utf-8")


def change_row(
    offset: int, service: str = "orders", summary: str = "config update"
) -> dict[str, object]:
    return {
        "at": (WINDOW_START + timedelta(seconds=offset)).isoformat(),
        "service": service,
        "summary": summary,
    }


def write_topology(
    paths: RunPaths, edges: list[str], services: tuple[str, ...] = SERVICES
) -> None:
    paths.root.mkdir(parents=True, exist_ok=True)
    manifest = {"services": list(services), "edges": edges}
    paths.topology_file.write_text(json.dumps(manifest), encoding="utf-8")


def metric_proposal(service: str = "gateway") -> ToolProposal:
    return ToolProposal(
        arguments=QueryMetricArguments(
            template=MetricTemplate.GATEWAY_LATENCY_P95,
            service=service,
            window_start=WINDOW_START,
            window_end=WINDOW_END,
        ),
        evidence_gap="how latency moved during the window",
        expected_observation="a latency rise at the gateway",
    )


def logs_proposal(row_limit: int = 20) -> ToolProposal:
    return ToolProposal(
        arguments=QueryLogsArguments(
            log_filter=LogFilter.TIMEOUTS_ONLY,
            service="inventory",
            window_start=WINDOW_START,
            window_end=WINDOW_END,
            row_limit=row_limit,
        ),
        evidence_gap="whether inventory timed out",
        expected_observation="timeout rows from inventory",
    )


def changes_proposal(service: str = "orders") -> ToolProposal:
    return ToolProposal(
        arguments=ListRecentChangesArguments(
            service=service, window_start=WINDOW_START, window_end=WINDOW_END
        ),
        evidence_gap="whether orders had a recent change",
        expected_observation="a change matching the errors",
    )


def topology_proposal(incident_id: str = INCIDENT_ID) -> ToolProposal:
    return ToolProposal(
        arguments=GetTopologyArguments(incident_id=incident_id),
        evidence_gap="how orders connects to its dependencies",
        expected_observation="the recorded service topology",
    )


def hypotheses() -> tuple[Hypothesis, Hypothesis]:
    return (
        Hypothesis(
            root_cause=RootCauseCode.DOWNSTREAM_TIMEOUT_RETRY_AMPLIFICATION,
            rank=1,
            missing_evidence="inventory timeout rate in the window",
        ),
        Hypothesis(
            root_cause=RootCauseCode.RESOURCE_POOL_SATURATION,
            rank=2,
            missing_evidence="orders pool usage in the window",
        ),
    )


def plan_json(
    proposal: ToolProposal | None = None, stop_reason: str | None = None
) -> dict[str, JsonValue]:
    plan = InitialPlan(
        hypotheses=hypotheses(), proposal=proposal, stop_reason=stop_reason
    )
    return plan.model_dump(mode="json")


def update_json(
    proposal: ToolProposal | None = None, stop_reason: str | None = None
) -> dict[str, JsonValue]:
    update = HypothesisUpdate(
        hypotheses=hypotheses(), proposal=proposal, stop_reason=stop_reason
    )
    return update.model_dump(mode="json")


def assessment_json(
    disposition: ModelDisposition = ModelDisposition.DIAGNOSED,
    root_cause: RootCauseCode = RootCauseCode.DOWNSTREAM_TIMEOUT_RETRY_AMPLIFICATION,
    supporting: tuple[str, ...] = (SYMPTOM_EVIDENCE_ID,),
) -> dict[str, JsonValue]:
    assessment = FinalAssessment(
        disposition=disposition,
        root_cause=root_cause,
        supporting_evidence_ids=supporting,
        uncertainty="one check could not separate the two remaining causes",
        next_step="ask the owner to confirm the inventory timeout setting",
    )
    return assessment.model_dump(mode="json")


def replay_model(
    tmp_path: Path, responses: dict[str, list[dict[str, JsonValue]]]
) -> ReplayReasoningModel:
    """Write a throwaway fixture so a test can script the stages it cares about."""
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps({"responses": responses}), encoding="utf-8")
    return ReplayReasoningModel(fixture)


class UsageReportingModel:
    """Wraps a replay model so the workflow sees provider usage on every call.

    `requests` delegates to the wrapped model, the same passthrough
    `ReplayToolCallingModel.requests` already uses for its own `inner` --
    that is what lets this class stand in as `ReplayToolCallingModel`'s
    `inner` too (`ReplayToolCallingModel(UsageReportingModel(...))`), so a
    graph-level test can see the same accumulated usage a loop-level one
    does, through one more layer of wrapping.
    """

    def __init__(self, inner: ReplayReasoningModel, usage: ModelUsage) -> None:
        self.inner = inner
        self.usage = usage

    @property
    def requests(self) -> list[ModelRequest]:
        return self.inner.requests

    def respond(self, request: ModelRequest) -> ModelResponse:
        return self.inner.respond(request).model_copy(update={"usage": self.usage})


class StepClock:
    """A clock that advances a fixed step every time it is read."""

    def __init__(self, step_seconds: float = 1.0) -> None:
        self.now = WINDOW_START
        self.step = timedelta(seconds=step_seconds)

    def __call__(self) -> datetime:
        reading = self.now
        self.now += self.step
        return reading


def check_runner(
    outcome: ToolOutcome = ToolOutcome.EXECUTED,
    kind: EvidenceKind = EvidenceKind.METRIC,
    reason_code: ReasonCode | None = None,
) -> RunCheck:
    """A stand-in for the registry-backed runner Step 3 supplies."""

    def run(proposal: ToolProposal, scope: IncidentScope) -> CheckOutcome:
        payload: dict[str, JsonValue] = {
            "template": "gateway_latency_p95",
            "p95_ms": 900,
        }
        return CheckOutcome(
            outcome=outcome,
            kind=kind,
            source=proposal.tool.value,
            summary="p95 latency rose to 900 ms",
            payload=payload if outcome is ToolOutcome.EXECUTED else {},
            reason_code=reason_code,
            duration_ms=12,
        )

    return run


class RecordingBackend[ArgsT: BaseModel]:
    """A backend seam stand-in that records every call instead of touching the
    lab -- the shape every `_make_wrapper` backend now takes,
    `Callable[[ArgsT, IncidentScope], CheckOutcome]`. Matches `FakeProbe.disk_paths`:
    a plain list a test can assert against with whole-list `==`, so "the spy
    was never called" is a one-line assertion, independently, per tool.

    `calls` records the full `(arguments, scope)` pair, not just the
    arguments -- `query_metric`'s backend is the one seam that actually reads
    the scope it is handed (the PromQL `incident` label, cross-incident
    isolation for the only networked backend), and only recording the
    arguments would leave no spy able to prove a wrapper forwarded the right
    scope at dispatch time rather than some other one.

    Set `raises` to make the backend fail mid-dispatch instead of returning,
    for testing that a crash still leaves a visible reserved receipt. Each
    tool's named subclass below only supplies its own default `CheckOutcome`
    shape -- the recording/raising behaviour lives here once.
    """

    def __init__(
        self,
        default_outcome: CheckOutcome,
        outcome: CheckOutcome | None = None,
        raises: Exception | None = None,
    ) -> None:
        self.calls: list[tuple[ArgsT, IncidentScope]] = []
        self.raises = raises
        self.outcome = outcome or default_outcome

    def __call__(self, arguments: ArgsT, scope: IncidentScope) -> CheckOutcome:
        self.calls.append((arguments, scope))
        if self.raises is not None:
            raise self.raises
        return self.outcome


class RecordingMetricBackend(RecordingBackend[QueryMetricArguments]):
    def __init__(
        self, outcome: CheckOutcome | None = None, raises: Exception | None = None
    ) -> None:
        super().__init__(
            CheckOutcome(
                outcome=ToolOutcome.EXECUTED,
                kind=EvidenceKind.METRIC,
                source="query_metric",
                summary="p95 latency rose to 900 ms",
                payload={"sample_count": 1},
                duration_ms=5,
            ),
            outcome,
            raises,
        )


class RecordingLogsBackend(RecordingBackend[QueryLogsArguments]):
    def __init__(
        self, outcome: CheckOutcome | None = None, raises: Exception | None = None
    ) -> None:
        super().__init__(
            CheckOutcome(
                outcome=ToolOutcome.EXECUTED,
                kind=EvidenceKind.LOG,
                source="query_logs",
                summary="1 row matched",
                payload={"row_count": 1},
                duration_ms=5,
            ),
            outcome,
            raises,
        )


class RecordingChangesBackend(RecordingBackend[ListRecentChangesArguments]):
    def __init__(
        self, outcome: CheckOutcome | None = None, raises: Exception | None = None
    ) -> None:
        super().__init__(
            CheckOutcome(
                outcome=ToolOutcome.EXECUTED,
                kind=EvidenceKind.CHANGE,
                source="list_recent_changes",
                summary="1 recent change on orders",
                payload={"change_count": 1},
                duration_ms=5,
            ),
            outcome,
            raises,
        )


class RecordingTopologyBackend(RecordingBackend[GetTopologyArguments]):
    def __init__(
        self, outcome: CheckOutcome | None = None, raises: Exception | None = None
    ) -> None:
        super().__init__(
            CheckOutcome(
                outcome=ToolOutcome.EXECUTED,
                kind=EvidenceKind.TOPOLOGY,
                source="get_topology",
                summary="1 service edge",
                payload={"edge_count": 1},
                duration_ms=5,
            ),
            outcome,
            raises,
        )


class UnexpectedBackendCall(BaseException):
    """A logs-only test's script proposed a tool its registry only stubs.

    `BaseException`, not `Exception`: a test-double guard must escape the
    code under test, not be caught by it. `graph.py`'s `dispatch_tool` (and
    the now-retired `workflow.py`'s own `run_investigation` -- a different
    function from `cli.py`'s dispatcher of the same name today -- before it)
    both catch `Exception` around a backend call -- an `Exception` subclass
    here would be swallowed
    into a normal `FAILED_SAFE`/`INTERNAL_ERROR` report, which is exactly the
    outcome a test asserting `_unexpected_call` was never reached is supposed
    to be impossible to produce by accident. The seventh variant of this
    project's containment defect: a signal meant to fail a test loudly,
    silently absorbed by the same blanket handler that protects a real run
    from a real backend crash.
    """


def _unexpected_call(*args: object, **kwargs: object) -> CheckOutcome:
    """Wired into the three tool slots a `logs_only_registry` test does not
    care about. Raising here instead of returning a benign outcome means a
    test whose script unexpectedly proposes one of those tools fails loudly.
    The raise stays after `dispatch()`'s own `ledger.reserve()` -- moving it
    earlier would drop the reserved-receipt guarantee `tool_wrappers.py`
    exists to hold even for a crashing backend -- and the recorded event
    logs only `type(error).__name__` (the retired loop's own deliberate
    rule, since error text can quote untrusted input), so the diagnostic
    lives in this exception's own name instead: `events.jsonl` reads
    `backend_crashed {'error': 'UnexpectedBackendCall'}`, self-explanatory
    without needing this message at all."""
    raise UnexpectedBackendCall("this tool backend was not expected to be called")


def registry_with(
    *,
    run_metric: Callable[
        [QueryMetricArguments, IncidentScope], CheckOutcome
    ] = _unexpected_call,
    run_logs: Callable[
        [QueryLogsArguments, IncidentScope], CheckOutcome
    ] = _unexpected_call,
    run_changes: Callable[
        [ListRecentChangesArguments, IncidentScope], CheckOutcome
    ] = _unexpected_call,
    run_topology: Callable[
        [GetTopologyArguments, IncidentScope], CheckOutcome
    ] = _unexpected_call,
) -> dict[ToolName, ToolWrapper]:
    """The full four-tool registry `dispatch_registry` now requires, with
    only the backends a test's script actually calls wired to a real or spy
    callable -- every other slot defaults to `_unexpected_call`, so a script
    that unexpectedly proposes one of them fails loudly instead of silently
    returning a benign outcome. Generalises `logs_only_registry` below (kept
    as a thin, still-named alias -- most existing tests only ever script
    `query_logs`) to the two- and three-tool scripts newer tests need."""
    return dispatch_registry(
        run_metric=run_metric,
        run_logs=run_logs,
        run_changes=run_changes,
        run_topology=run_topology,
    )


def logs_only_registry(
    run_logs: Callable[[QueryLogsArguments, IncidentScope], CheckOutcome],
) -> dict[ToolName, ToolWrapper]:
    """The full four-tool registry `dispatch_registry` now requires, with
    only `query_logs` backed by a real or spy callable -- for the many
    existing tests that only ever script a `query_logs` proposal. See
    `_unexpected_call` for what happens if that assumption ever stops
    holding for one of them."""
    return registry_with(run_logs=run_logs)


class RecordingPrometheus(NamedTuple):
    """The loopback server's address, plus every PromQL string it received."""

    url: str
    queries: list[str]


def prometheus_body(sample_count: int = MAX_METRIC_SAMPLES + 5) -> bytes:
    """More samples than `MAX_METRIC_SAMPLES` by default, matching
    `test_telemetry.py`'s own truncation test -- a real-backend test that
    only wants a successful response doesn't care that it is truncated."""
    values = [[float(index), str(index * 0.5)] for index in range(sample_count)]
    return json.dumps(
        {"status": "success", "data": {"result": [{"values": values}]}}
    ).encode("utf-8")
