"""API-key authentication for the HTTP transport.

Over stdio the client launches the server as a local subprocess, so trust is
implicit. Over HTTP the server is reachable by anyone who has the URL, so every
request must prove it holds a shared secret, sent in the standard header:

    Authorization: Bearer <key>

This is the simplest real auth. Production systems usually use OAuth — the MCP
spec defines an OAuth flow and the SDK has a TokenVerifier hook where it slots in.
"""

from __future__ import annotations

import hmac
import os

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

API_KEY_ENV = "CREDIT_RISK_API_KEY"  # the secret is read from this env var
BEARER_PREFIX = "Bearer "
# Paths reachable WITHOUT a key (hosting platforms ping these for health checks).
PUBLIC_PATHS = {"/health"}


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Reject requests lacking a valid 'Authorization: Bearer <key>' header.

    Health-check paths pass through; the API key is read from the environment
    (fail closed with 500 if it's not configured) and compared in constant time.
    """

    async def dispatch(self, request: Request, call_next):
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        expected = os.environ.get(API_KEY_ENV)

        if not expected:
            return JSONResponse({"error": "server misconfigured"}, status_code=500)

        header = request.headers.get("Authorization", "")

        if not header.startswith(BEARER_PREFIX):
            return JSONResponse(
                {"error": "unauthorized"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )

        token = header[len(BEARER_PREFIX):]

        if not hmac.compare_digest(token, expected):
            return JSONResponse(
                {"error": "unauthorized"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )

        return await call_next(request)
