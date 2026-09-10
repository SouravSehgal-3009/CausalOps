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
        self.idempotency_keys: list[str] = []

    def create(
        self,
        owner_email: str,
        request: CreateInvestigationRequest,
        idempotency_key: str,
    ) -> InvestigationView:
        self.owners.append(owner_email)
        self.idempotency_keys.append(idempotency_key)
        return InvestigationView(
            investigation_id="run-1", status=InvestigationStatus.RUNNING
        )

    def status(self, owner_email: str, investigation_id: str) -> InvestigationView:
        return InvestigationView(
            investigation_id=investigation_id,
            status=InvestigationStatus.PAUSED_APPROVAL,
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
    idempotency_headers = {
        "Authorization": "Bearer verified-token",
        "Idempotency-Key": "key-1",
    }
    denied = http.post(
        "/api/v1/investigations",
        headers={"Idempotency-Key": "key-1"},
        json={"scenario_family": "configuration_change"},
    )
    assert denied.status_code == 401

    missing_key = http.post(
        "/api/v1/investigations",
        headers={"Authorization": "Bearer verified-token"},
        json={"scenario_family": "configuration_change"},
    )
    assert missing_key.status_code == 400

    created = http.post(
        "/api/v1/investigations",
        headers=idempotency_headers,
        json={"scenario_family": "configuration_change"},
    )
    assert created.status_code == 200
    assert backend.owners == ["owner@example.com"]
    assert backend.idempotency_keys == ["key-1"]

    invalid = http.post(
        "/api/v1/investigations",
        headers=idempotency_headers,
        json={
            "scenario_family": "configuration_change",
            "seed": "development",
            "model": "claude",
        },
    )
    assert invalid.status_code == 422

    for index, supported_scenario in enumerate(
        (
            "ambiguous_telemetry",
            "downstream_timeout_retry_amplification",
            "resource_pool_saturation",
        )
    ):
        supported = http.post(
            "/api/v1/investigations",
            headers={**idempotency_headers, "Idempotency-Key": f"key-family-{index}"},
            json={"scenario_family": supported_scenario},
        )
        assert supported.status_code == 200

    unsupported = http.post(
        "/api/v1/investigations",
        headers={**idempotency_headers, "Idempotency-Key": "key-unsupported"},
        json={"scenario_family": "not_a_real_family"},
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
            "/api/v1/investigations/run-1",
            headers={"Authorization": "Bearer verified-token"},
        ).status_code
        == 403
    )


def test_replay_api_exposes_status_events_decisions_and_reports() -> None:
    http, _ = client()
    headers = {"Authorization": "Bearer verified-token"}
    assert (
        http.get("/api/v1/investigations/run-1", headers=headers).json()["status"]
        == "PAUSED_APPROVAL"
    )
    assert (
        http.get("/api/v1/investigations/run-1/events", headers=headers).json()[0][
            "name"
        ]
        == "investigation_started"
    )
    assert (
        http.post(
            "/api/v1/investigations/run-1/decision",
            headers=headers,
            json={"decision": "accept"},
        ).status_code
        == 200
    )
    assert (
        "Cited replay report"
        in http.get("/api/v1/investigations/run-1/report", headers=headers).json()
    )
    assert (
        http.post(
            "/api/v1/investigations/run-1/decision",
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
    assert '"/api/v1/investigations"' in page.text
    assert "Resume safely" in page.text
    assert "model" not in page.text.casefold()
