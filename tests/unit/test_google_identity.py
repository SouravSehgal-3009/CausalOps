import pytest

from causalops.google_identity import GoogleIdentityVerifier


def test_google_identity_verifier_accepts_a_verified_email() -> None:
    verifier = GoogleIdentityVerifier(
        "client-id",
        validator=lambda token, audience: {
            "email": "owner@example.com",
            "email_verified": True,
            "aud": audience,
        },
    )

    assert verifier.verify("id-token") == "owner@example.com"


@pytest.mark.parametrize(
    "claims",
    (
        {"email": "owner@example.com", "email_verified": False},
        {"email": "", "email_verified": True},
    ),
)
def test_google_identity_verifier_refuses_unverified_or_missing_email(
    claims: dict[str, object],
) -> None:
    verifier = GoogleIdentityVerifier(
        "client-id", validator=lambda token, audience: claims
    )

    with pytest.raises(ValueError):
        verifier.verify("id-token")


def test_google_identity_verifier_translates_validator_failure() -> None:
    def failed_validator(token: str, audience: str) -> dict[str, object]:
        raise RuntimeError("certificate service unavailable")

    verifier = GoogleIdentityVerifier("client-id", validator=failed_validator)

    with pytest.raises(ValueError, match="Google ID token is invalid"):
        verifier.verify("id-token")
