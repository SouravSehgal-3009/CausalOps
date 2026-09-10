"""Replay-only HTTP control-plane boundary.

The deployed identity adapter verifies a Google ID token before it supplies an
email to the owner allowlist. Neither this API nor its dashboard accepts a
provider or model choice: hosted investigations always use replay wiring.
"""

# ruff: noqa: E501

import json
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Protocol

from fastapi import FastAPI, Header, HTTPException, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from causalops.approvals import OwnerDecision


class ScenarioFamily(StrEnum):
    """Scenario families backed by verified hosted replay fixtures."""

    CONFIGURATION_CHANGE = "configuration_change"


class ReplaySeed(StrEnum):
    """The one checked-in replay data set the hosted API is ever allowed to
    select -- never user-controlled, never an evaluator seed (those exist
    only for `causalops-evaluate`'s own frozen corpus, a wholly separate,
    non-hosted code path)."""

    DEVELOPMENT = "development"


class InvestigationStatus(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    PAUSED_APPROVAL = "PAUSED_APPROVAL"
    COMPLETED = "COMPLETED"
    FAILED_SAFE = "FAILED_SAFE"


class CreateInvestigationRequest(BaseModel):
    """No incident ID, model, tool, query, or evaluator seed is accepted --
    the server always uses `ReplaySeed.DEVELOPMENT` (see `api_runtime.py`'s
    `SqliteReplayControlPlane.create`); a client cannot select an evaluator
    seed through this endpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_family: ScenarioFamily


class DecisionRequest(OwnerDecision):
    model_config = ConfigDict(extra="forbid", frozen=True)


class InvestigationView(BaseModel):
    model_config = ConfigDict(frozen=True)

    investigation_id: str
    status: InvestigationStatus


class TimelineEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    ordinal: int = Field(ge=0)
    name: str
    fields: Mapping[str, JsonValue]


class WorkerClaim(CreateInvestigationRequest):
    """The fixed replay inputs a worker receives after atomically claiming a
    job. `seed` is declared here, not on `CreateInvestigationRequest` --
    the server always stores `ReplaySeed.DEVELOPMENT` (`create()`'s own
    docstring), but the worker still needs the stored value back to start
    the scenario.

    Lives here, not in `api_runtime.py`/`firestore_control_plane.py`, so
    both control-plane backends' worker-facing methods return the exact
    same type -- `WorkerControlPlane`'s structural typing only works if
    `FirestoreReplayControlPlane.claim_next()` and
    `SqliteReplayControlPlane.claim_next()` agree on one nominal class, not
    two independently-defined lookalikes.
    """

    model_config = ConfigDict(frozen=True)

    seed: ReplaySeed
    investigation_id: str
    owner_email: str
    checkpoint_id: str | None
    incident_id: str | None = None
    owner_decision: DecisionRequest | None = None
    claim_token: str


class DeliveryClaim(InvestigationView):
    """A report delivery work item addressed only to the investigation
    owner. See `WorkerClaim`'s own docstring for why this lives here."""

    model_config = ConfigDict(frozen=True)

    report_artifact: str
    report_content: str
    report_sha256: str
    destination_email: str
    delivery_id: str
    claim_token: str


class ControlPlaneNotFoundError(LookupError):
    """The caller does not own a requested investigation."""


class ControlPlaneConflictError(RuntimeError):
    """A requested state transition cannot safely be applied."""


class ReplayControlPlane(Protocol):
    """Owner-scoped operations; implementations hold replay-only wiring."""

    def create(
        self,
        owner_email: str,
        request: CreateInvestigationRequest,
        idempotency_key: str,
    ) -> InvestigationView: ...

    def status(self, owner_email: str, investigation_id: str) -> InvestigationView: ...

    def events(
        self, owner_email: str, investigation_id: str
    ) -> Sequence[TimelineEvent]: ...

    def decide(
        self, owner_email: str, investigation_id: str, decision: DecisionRequest
    ) -> InvestigationView: ...

    def report(self, owner_email: str, investigation_id: str) -> str: ...


class IdentityVerifier(Protocol):
    """Deployment seam around Google ID-token verification."""

    def verify(self, bearer_token: str) -> str: ...


class ControlPlaneWorkerService(Protocol):
    """Lifecycle boundary for the process that drains durable work."""

    def start(self) -> None: ...

    def stop(self) -> None: ...


def _dashboard_html(google_client_id: str) -> str:
    """A deliberately small owner dashboard with no model controls."""
    client_id = json.dumps(google_client_id)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>CausalOps</title></head>
<body><main><h1>CausalOps replay investigations</h1>
<p id="identity">Sign in with an approved Google account.</p>
<div id="g_id_onload" data-client_id={client_id}
 data-callback="onGoogleCredential"></div>
<div class="g_id_signin" data-type="standard"></div>
<section id="investigations" hidden>
<h2>Create a replay investigation</h2>
<form id="create-form"><label>Scenario
<select id="scenario-family">
<option value="configuration_change">Configuration change</option>
</select></label><button>Create</button></form>
<h2>Investigation</h2><label>ID <input id="investigation-id" required></label>
<button id="refresh" type="button">Refresh</button><pre id="result"></pre>
<form id="decision-form"><label>Decision <select id="decision">
<option value="accept">Accept</option><option value="reject">Reject</option>
</select></label><label>Rejection reason <input id="rejection-note"></label>
<button>Resume safely</button></form><h2>Report</h2><pre id="report"></pre>
</section>
</main><script>
function headers() {{
  return {{"Authorization": "Bearer " + window.causalOpsBearerToken,
    "Content-Type": "application/json"}};
}}
async function request(path, options) {{
  const response = await fetch(path, options);
  if (!response.ok) throw new Error("Request failed: " + response.status);
  return response.json();
}}
function id() {{ return document.getElementById("investigation-id").value.trim(); }}
async function refresh() {{
  const investigationId = id();
  if (!investigationId) return;
  const base = "/api/v1/investigations/" + encodeURIComponent(investigationId);
  const [view, events] = await Promise.all([
    request(base, {{headers: headers()}}), request(base + "/events", {{headers: headers()}})
  ]);
  document.getElementById("result").textContent = JSON.stringify({{view, events}}, null, 2);
  if (view.status === "COMPLETED") {{
    document.getElementById("report").textContent = await request(base + "/report", {{headers: headers()}});
  }}
}}
function onGoogleCredential(response) {{
  window.causalOpsBearerToken = response.credential;
  document.getElementById("identity").textContent =
    "Signed in with an approved account.";
  document.getElementById("investigations").hidden = false;
}}
window.onGoogleCredential = onGoogleCredential;
document.getElementById("create-form").addEventListener("submit", async (event) => {{
  event.preventDefault();
  const createHeaders = headers();
  createHeaders["Idempotency-Key"] = crypto.randomUUID();
  const view = await request("/api/v1/investigations", {{method: "POST", headers: createHeaders,
    body: JSON.stringify({{scenario_family: document.getElementById("scenario-family").value}})}});
  document.getElementById("investigation-id").value = view.investigation_id;
  await refresh();
}});
document.getElementById("refresh").addEventListener("click", () => refresh().catch(showError));
document.getElementById("decision-form").addEventListener("submit", async (event) => {{
  event.preventDefault();
  const decision = document.getElementById("decision").value;
  const payload = {{decision: decision}};
  if (decision === "reject") payload.rejection_note = document.getElementById("rejection-note").value;
  await request("/api/v1/investigations/" + encodeURIComponent(id()) + "/decision", {{
    method: "POST", headers: headers(), body: JSON.stringify(payload)}});
  await refresh();
}});
function showError(error) {{ document.getElementById("result").textContent = error.message; }}
</script><script src="https://accounts.google.com/gsi/client" async defer>
</script></body></html>"""


def create_app(
    control_plane: ReplayControlPlane,
    *,
    allowed_owners: frozenset[str],
    identity_verifier: IdentityVerifier,
    google_client_id: str = "",
    worker_service: ControlPlaneWorkerService | None = None,
) -> FastAPI:
    """Build the API with a verified-owner allowlist at its trust boundary."""
    normalized_owners = frozenset(
        owner.strip().casefold() for owner in allowed_owners if owner.strip()
    )
    app = FastAPI(title="CausalOps replay control plane", version="1")

    if worker_service is not None:

        @app.on_event("startup")
        def start_workers() -> None:
            worker_service.start()

        @app.on_event("shutdown")
        def stop_workers() -> None:
            worker_service.stop()

    def owner(authorization: str | None) -> str:
        if authorization is None or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
        token = authorization.removeprefix("Bearer ").strip()
        if not token:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
        try:
            verified_owner = identity_verifier.verify(token).strip().casefold()
        except ValueError as error:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED) from error
        if not verified_owner or verified_owner not in normalized_owners:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
        return verified_owner

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def dashboard() -> HTMLResponse:
        return HTMLResponse(_dashboard_html(google_client_id))

    def idempotency_key(raw: str | None) -> str:
        if raw is None or not raw.strip():
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)
        return raw.strip()

    @app.post("/api/v1/investigations", response_model=InvestigationView)
    def create(
        request: CreateInvestigationRequest,
        authorization: str | None = Header(default=None),
        idempotency_key_header: str | None = Header(
            default=None, alias="Idempotency-Key"
        ),
    ) -> InvestigationView:
        return control_plane.create(
            owner(authorization), request, idempotency_key(idempotency_key_header)
        )

    @app.get(
        "/api/v1/investigations/{investigation_id}", response_model=InvestigationView
    )
    def get_status(
        investigation_id: str, authorization: str | None = Header(default=None)
    ) -> InvestigationView:
        try:
            return control_plane.status(owner(authorization), investigation_id)
        except (ControlPlaneNotFoundError, PermissionError) as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from error

    @app.get(
        "/api/v1/investigations/{investigation_id}/events",
        response_model=list[TimelineEvent],
    )
    def get_events(
        investigation_id: str, authorization: str | None = Header(default=None)
    ) -> Sequence[TimelineEvent]:
        try:
            return control_plane.events(owner(authorization), investigation_id)
        except (ControlPlaneNotFoundError, PermissionError) as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from error

    @app.post(
        "/api/v1/investigations/{investigation_id}/decision",
        response_model=InvestigationView,
    )
    def decide(
        investigation_id: str,
        request: DecisionRequest,
        authorization: str | None = Header(default=None),
    ) -> InvestigationView:
        try:
            return control_plane.decide(owner(authorization), investigation_id, request)
        except (ControlPlaneNotFoundError, PermissionError) as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from error
        except ControlPlaneConflictError as error:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT) from error

    @app.get("/api/v1/investigations/{investigation_id}/report", response_model=str)
    def get_report(
        investigation_id: str, authorization: str | None = Header(default=None)
    ) -> str:
        try:
            return control_plane.report(owner(authorization), investigation_id)
        except (ControlPlaneNotFoundError, PermissionError) as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from error
        except ControlPlaneConflictError as error:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT) from error

    return app
