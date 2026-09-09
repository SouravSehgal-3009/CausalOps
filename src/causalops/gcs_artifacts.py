"""Durable Cloud Storage upload for finalized investigation artifacts.

Wires the already-provisioned `infra/phase2/main.tf` bucket
(`google_storage_bucket.replay_artifacts`) into the application -- spec
§3.1: "Cloud Storage stores immutable final artifacts under
`investigations/{investigation_id}/`... Writes use generation
preconditions." Local disk (`finalize_investigation`, `run_records.py`)
stays the working/in-progress store and the source this module uploads
from; this module makes Cloud Storage the durable remote copy spec §3.1
describes, once `api_runtime.py`'s composition root wires it in.

Nothing here is reachable from `cli.py`/`evaluate_cli.py` -- only
`api_runtime.py`, the hosted control plane, ever imports this module.
"""

from collections.abc import Mapping
from typing import Protocol

import google.auth
import google.cloud.storage as storage
from google.api_core.exceptions import PreconditionFailed
from google.auth.impersonated_credentials import Credentials as ImpersonatedCredentials

# GCS-only, not the broader `cloud-platform` scope -- this identity's only
# real job is reading and writing objects in one bucket.
_IMPERSONATION_SCOPES = ("https://www.googleapis.com/auth/devstorage.read_write",)

ARTIFACT_NAMES: tuple[str, ...] = (
    "report.json",
    "report.md",
    "events.jsonl",
    "evidence.jsonl",
    "receipts.jsonl",
)


class _Blob(Protocol):
    def upload_from_string(
        self, data: str, if_generation_match: int | None = None
    ) -> None: ...


class _Bucket(Protocol):
    def blob(self, path: str) -> _Blob: ...


class _Client(Protocol):
    """The narrow seam `GcsArtifactStore` actually calls -- a real
    `storage.Client` satisfies this structurally; tests inject a fake
    client shaped the same way, the same seam-testing approach this
    project already uses for `IdentityVerifier`/`ReplayControlPlane`."""

    def bucket(self, name: str) -> _Bucket: ...


class ArtifactStore(Protocol):
    """`api_runtime.ReplayGraphJobRunner`'s own dependency seam -- a real
    `GcsArtifactStore` satisfies this structurally; `api_runtime.py`'s own
    tests inject a fake recording store shaped the same way."""

    def write_investigation_artifacts(
        self, investigation_id: str, artifacts: Mapping[str, str]
    ) -> None: ...


def _impersonated_client(target_service_account: str) -> storage.Client:
    """The VM's own runtime identity is its default Compute Engine service
    account, never granted a bucket role of its own -- only permission to
    impersonate `target_service_account` for exactly as long as one GCS
    request (`infra/phase2/main.tf`'s own
    `google_service_account_iam_member.vm_impersonates_control_plane`
    comment explains why). `google.auth.default()` resolves the VM's own
    ambient credentials; `ImpersonatedCredentials` mints a short-lived
    token for the target identity from those, scoped to GCS only.
    """
    source_credentials, _ = google.auth.default()
    impersonated = ImpersonatedCredentials(  # type: ignore[no-untyped-call]
        source_credentials=source_credentials,
        target_principal=target_service_account,
        target_scopes=list(_IMPERSONATION_SCOPES),
    )
    return storage.Client(credentials=impersonated)


class GcsArtifactStore:
    """Upload the 5 finalized artifacts for one investigation.

    Without `target_service_account`, a real `storage.Client()` resolves
    Application Default Credentials directly (the ambient identity making
    the call). With it, credentials are impersonated instead (see
    `_impersonated_client`). Either way this resolution happens at
    construction time -- deliberately not at import time, and never inside
    `cli.py`/`evaluate_cli.py`'s own import chain (see this module's own
    docstring).
    """

    def __init__(
        self,
        bucket_name: str,
        client: _Client | None = None,
        target_service_account: str | None = None,
    ) -> None:
        if client is not None:
            self._client: _Client = client
        elif target_service_account:
            self._client = _impersonated_client(target_service_account)
        else:
            self._client = storage.Client()
        self._bucket = self._client.bucket(bucket_name)

    def write_investigation_artifacts(
        self, investigation_id: str, artifacts: Mapping[str, str]
    ) -> None:
        """Uploads each `artifacts[name]` under
        `investigations/{investigation_id}/{name}`, one object per call,
        with `if_generation_match=0` so a create only ever succeeds against
        an absent object.

        A `PreconditionFailed` (412 -- the object already exists) is
        treated as success, not an error: the caller's own retry may land
        here after an earlier attempt already uploaded this exact artifact
        (the bucket's `objectCreator` IAM grant permits create only, never
        overwrite, by design -- `infra/phase2/main.tf`'s own comment), so a
        repeat upload of byte-identical content is the expected idempotent
        case, not a real failure. Any other exception (network, auth,
        permission) propagates for the caller's own bounded-retry
        machinery to handle, the same way every other external I/O failure
        in this project is handled.
        """
        for name, content in artifacts.items():
            blob = self._bucket.blob(f"investigations/{investigation_id}/{name}")
            try:
                blob.upload_from_string(content, if_generation_match=0)
            except PreconditionFailed:
                continue
