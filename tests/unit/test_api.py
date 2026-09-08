from fastapi.testclient import TestClient

from causalops.api import (
    CreateInvestigationRequest,
    DecisionRequest,
    InvestigationStatus,
    InvestigationView,
    TimelineEvent,
    create_app,
)


class FakeControlPlane:
    def __init__(self) -> None:
        self.owners: list[str] = []

    def create(
        self, owner_email: str, request: CreateInvestigationRequest
    ) -> InvestigationView:
        self.owners.append(owner_email)
        return InvestigationView(
            investigation_id="run-1", status=InvestigationStatus.RUNNING
        )

    def status(self, owner_email: str, investigation_id: str) -> InvestigationView:
        return InvestigationView(
            investigation_id=investigation_id, status=InvestigationStatus.PAUSED
        )

    def events(self, owner_email: str, investigation_id: str) -> list[TimelineEvent]:
        return [
            TimelineEvent(
                ordinal=0, name="investigation_started", fields={"id": investigation_id}
            )
        ]

    def decide(
        self, owner_email: str, investigation_id: str, decision: DecisionRequest
    ) -> InvestigationView:
        return InvestigationView(
            investigation_id=investigation_id, status=InvestigationStatus.RUNNING
        )

    def report(self, owner_email: str, investigation_id: str) -> str:
        return "# Cited replay report"


class FakeIdentityVerifier:
    def verify(self, bearer_token: str) -> str:
        if bearer_token != "verified-token":
            raise ValueError("invalid token")
        return "owner@example.com"


def client() -> tuple[TestClient, FakeControlPlane]:
    backend = FakeControlPlane()
    return TestClient(
        create_app(
            backend,
            allowed_owners=frozenset({"owner@example.com"}),
            identity_verifier=FakeIdentityVerifier(),
        )
    ), backend


def test_replay_api_is_owner_scoped_and_has_no_model_input() -> None:
    http, backend = client()
    denied = http.post(
        "/v1/investigations",
        json={"scenario_family": "configuration_change", "seed": "development"},
    )
    assert denied.status_code == 401

    created = http.post(
        "/v1/investigations",
        headers={"Authorization": "Bearer verified-token"},
        json={"scenario_family": "configuration_change", "seed": "development"},
    )
    assert created.status_code == 200
    assert backend.owners == ["owner@example.com"]

    invalid = http.post(
        "/v1/investigations",
        headers={"Authorization": "Bearer verified-token"},
        json={
            "scenario_family": "configuration_change",
            "seed": "development",
            "model": "claude",
        },
    )
    assert invalid.status_code == 422

    for unsupported_scenario in (
        "ambiguous_telemetry",
        "downstream_timeout_retry_amplification",
        "resource_pool_saturation",
    ):
        unsupported = http.post(
            "/v1/investigations",
            headers={"Authorization": "Bearer verified-token"},
            json={"scenario_family": unsupported_scenario, "seed": "development"},
        )
        assert unsupported.status_code == 422

    other_owner = TestClient(
        create_app(
            backend,
            allowed_owners=frozenset({"different@example.com"}),
            identity_verifier=FakeIdentityVerifier(),
        )
    )
    assert (
        other_owner.get(
            "/v1/investigations/run-1",
            headers={"Authorization": "Bearer verified-token"},
        ).status_code
        == 403
    )


def test_replay_api_exposes_status_events_decisions_and_reports() -> None:
    http, _ = client()
    headers = {"Authorization": "Bearer verified-token"}
    assert (
        http.get("/v1/investigations/run-1", headers=headers).json()["status"]
        == "PAUSED"
    )
    assert (
        http.get("/v1/investigations/run-1/events", headers=headers).json()[0]["name"]
        == "investigation_started"
    )
    assert (
        http.post(
            "/v1/investigations/run-1/decision",
            headers=headers,
            json={"decision": "accept"},
        ).status_code
        == 200
    )
    assert (
        "Cited replay report"
        in http.get("/v1/investigations/run-1/report", headers=headers).json()
    )
    assert (
        http.post(
            "/v1/investigations/run-1/decision",
            headers=headers,
            json={"decision": "reject"},
        ).status_code
        == 422
    )


def test_dashboard_uses_google_sign_in_without_a_model_control() -> None:
    http, _ = client()
    page = http.get("/")
    assert page.status_code == 200
    assert "accounts.google.com/gsi/client" in page.text
    assert 'id="create-form"' in page.text
    assert '"/v1/investigations"' in page.text
    assert "Resume safely" in page.text
    assert "model" not in page.text.casefold()
