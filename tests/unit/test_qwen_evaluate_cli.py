"""The Qwen batch command must refuse before touching the lab or model."""

import pytest

from causalops.live_setup import ProviderDisabledError
from causalops.qwen_evaluate_cli import (
    OLLAMA_IMAGE_DIGEST_VARIABLE,
    QWEN_MANIFEST_DIGEST_VARIABLE,
    _assert_candidate_environment,
)


def test_candidate_batch_requires_vm_approval_before_identity_or_runtime() -> None:
    with pytest.raises(ProviderDisabledError, match="private-VM-only"):
        _assert_candidate_environment({})


def test_candidate_batch_requires_immutable_identities_after_vm_approval() -> None:
    with pytest.raises(ProviderDisabledError, match="OLLAMA_IMAGE_DIGEST"):
        _assert_candidate_environment(
            {
                "CAUSALOPS_EXECUTION_ENV": "vm",
                "CAUSALOPS_CANDIDATE_EVALUATION": "true",
            }
        )


def test_candidate_batch_accepts_only_sha256_identities() -> None:
    digest = "sha256:" + "a" * 64
    assert _assert_candidate_environment(
        {
            "CAUSALOPS_EXECUTION_ENV": "vm",
            "CAUSALOPS_CANDIDATE_EVALUATION": "true",
            OLLAMA_IMAGE_DIGEST_VARIABLE: digest,
            QWEN_MANIFEST_DIGEST_VARIABLE: digest,
        }
    ) == (digest, digest)
