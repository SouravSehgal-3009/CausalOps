"""Google ID-token verification at the HTTP composition boundary."""

from collections.abc import Callable, Mapping
from typing import cast

GoogleTokenValidator = Callable[[str, str], Mapping[str, object]]


def _verify_google_id_token(token: str, audience: str) -> Mapping[str, object]:
    """Verify signature, issuer, expiry, and audience using Google's library."""
    from google.auth.transport.requests import Request
    from google.oauth2.id_token import verify_oauth2_token

    return cast(
        Mapping[str, object],
        verify_oauth2_token(token, Request(), audience),  # type: ignore[no-untyped-call]
    )


class GoogleIdentityVerifier:
    """Returns a verified Google email only when its claim is verified."""

    def __init__(
        self, client_id: str, validator: GoogleTokenValidator = _verify_google_id_token
    ) -> None:
        self._client_id = client_id.strip()
        if not self._client_id:
            raise ValueError("Google OAuth client ID must not be blank")
        self._validator = validator

    def verify(self, bearer_token: str) -> str:
        try:
            claims = self._validator(bearer_token, self._client_id)
        except Exception as error:
            raise ValueError("Google ID token is invalid") from error
        email = claims.get("email")
        if not isinstance(email, str) or not email.strip():
            raise ValueError("Google ID token has no email claim")
        if claims.get("email_verified") is not True:
            raise ValueError("Google ID token email is not verified")
        return email
