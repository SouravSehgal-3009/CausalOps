"""`GcsArtifactStore`'s own upload/idempotency contract, against a fake
client -- no real GCP access, no network."""

from collections.abc import Mapping

import pytest
from google.api_core.exceptions import PreconditionFailed

from causalops.gcs_artifacts import ARTIFACT_NAMES, GcsArtifactStore


class FakeBlob:
    def __init__(self, store: dict[str, str], path: str) -> None:
        self._store = store
        self._path = path

    def upload_from_string(
        self, data: str, if_generation_match: int | None = None
    ) -> None:
        if if_generation_match == 0 and self._path in self._store:
            raise PreconditionFailed("object already exists")  # type: ignore[no-untyped-call]
        self._store[self._path] = data


class FakeBucket:
    def __init__(self, store: dict[str, str]) -> None:
        self._store = store

    def blob(self, path: str) -> FakeBlob:
        return FakeBlob(self._store, path)


class FakeClient:
    """A real `storage.Client()` is never constructed in these tests --
    `GcsArtifactStore.__init__` accepts an injected client precisely so
    ADC resolution never has to happen here."""

    def __init__(self) -> None:
        self.objects: dict[str, str] = {}

    def bucket(self, name: str) -> FakeBucket:
        return FakeBucket(self.objects)


def sample_artifacts() -> Mapping[str, str]:
    return {name: f"content-for-{name}" for name in ARTIFACT_NAMES}


def test_uploads_all_five_artifacts_under_the_investigation_prefix() -> None:
    client = FakeClient()
    store = GcsArtifactStore("test-bucket", client=client)

    store.write_investigation_artifacts("inv-1", sample_artifacts())

    assert set(client.objects) == {
        f"investigations/inv-1/{name}" for name in ARTIFACT_NAMES
    }
    assert client.objects["investigations/inv-1/report.md"] == "content-for-report.md"


def test_a_repeated_upload_of_the_same_artifact_is_treated_as_success() -> None:
    """A duplicate delivery (a retried finalize) lands on an object that
    already exists -- `PreconditionFailed` must be swallowed, not raised,
    since the bucket's own IAM grant (`infra/gcp/storage.tf`) is
    create-only and this is the expected idempotent case, not a real
    failure."""
    client = FakeClient()
    store = GcsArtifactStore("test-bucket", client=client)
    artifacts = sample_artifacts()

    store.write_investigation_artifacts("inv-1", artifacts)
    store.write_investigation_artifacts("inv-1", artifacts)  # must not raise

    assert len(client.objects) == len(ARTIFACT_NAMES)


def test_target_service_account_requests_impersonated_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When `target_service_account` is set, `GcsArtifactStore` must
    request impersonated credentials for that exact principal -- the VM's
    own default Compute Engine service account is never granted a bucket
    role of its own (`infra/gcp/storage.tf`'s `vm_impersonates_control_plane`
    grant is the only thing that lets it act as `causalops-replay-control`
    at all), so this is a real security property, not an implementation
    detail."""
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        "causalops.gcs_artifacts.google.auth.default",
        lambda: ("source-creds", None),
    )

    class FakeImpersonatedCredentials:
        def __init__(self, **kwargs: object) -> None:
            calls.append(kwargs)

    monkeypatch.setattr(
        "causalops.gcs_artifacts.ImpersonatedCredentials", FakeImpersonatedCredentials
    )

    built_clients: list[object] = []

    class FakeStorageClient:
        def __init__(self, credentials: object) -> None:
            built_clients.append(credentials)

        def bucket(self, name: str) -> FakeBucket:
            return FakeBucket({})

    monkeypatch.setattr("causalops.gcs_artifacts.storage.Client", FakeStorageClient)

    GcsArtifactStore(
        "test-bucket", target_service_account="causalops-replay-control@example.iam"
    )

    assert len(calls) == 1
    assert calls[0]["target_principal"] == "causalops-replay-control@example.iam"
    assert calls[0]["source_credentials"] == "source-creds"
    assert isinstance(calls[0]["target_scopes"], list)
    assert len(built_clients) == 1
    assert isinstance(built_clients[0], FakeImpersonatedCredentials)


def test_a_genuine_upload_failure_still_propagates() -> None:
    """Only `PreconditionFailed` is swallowed -- every other exception
    (network, auth, permission) must still reach the caller's own
    bounded-retry machinery, the same way every other external I/O
    failure in this project is handled."""

    class FailingBlob:
        def upload_from_string(
            self, data: str, if_generation_match: int | None = None
        ) -> None:
            raise ConnectionError("simulated network failure")

    class FailingBucket:
        def blob(self, path: str) -> FailingBlob:
            return FailingBlob()

    class FailingClient:
        def bucket(self, name: str) -> FailingBucket:
            return FailingBucket()

    store = GcsArtifactStore("test-bucket", client=FailingClient())

    with pytest.raises(ConnectionError, match="simulated network failure"):
        store.write_investigation_artifacts("inv-1", sample_artifacts())
