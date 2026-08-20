"""One synthetic incident the unit tests share, plus doubles for the model and checks.

The IDs here stand in for what the scenario controller will produce in Step 3.
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import JsonValue

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
from causalops.telemetry import RunPaths
from causalops.tools import (
    LogFilter,
    MetricTemplate,
    QueryLogsArguments,
    QueryMetricArguments,
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
    """Wraps a replay model so the workflow sees provider usage on every call."""

    def __init__(self, inner: ReplayReasoningModel, usage: ModelUsage) -> None:
        self.inner = inner
        self.usage = usage

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
