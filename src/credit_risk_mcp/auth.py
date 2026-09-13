"""API-key authentication for the HTTP transport (Phase 4).

>>> SCAFFOLD — implement dispatch(). tests/test_auth.py drives you.

Why this file only appears now: with stdio (Phases 2-3) the client launches the
server as a local subprocess, so trust is implicit — no auth needed. Over HTTP
the server is reachable by anyone who has the URL, so every request must prove it
holds a shared secret, sent in the standard header:

    Authorization: Bearer <key>

This is the simplest real auth and is enough to learn the concept. Production
systems usually use OAuth — the MCP spec defines an OAuth flow and the SDK has a
TokenVerifier hook where it slots in (see PHASE4_GUIDE.md).
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

    TODO — implement dispatch below. Steps:
      1. If request.url.path is in PUBLIC_PATHS, let it through:
         `return await call_next(request)`.
      2. expected = os.environ.get(API_KEY_ENV). If it's missing/empty the server
         is misconfigured — return JSONResponse(status_code=500) and do NOT serve
         tools. (Fail closed: never fall back to "no auth".)
      3. header = request.headers.get("Authorization", ""). It must start with
         BEARER_PREFIX, and the part after it must equal `expected`. Compare with
         hmac.compare_digest(a, b) — constant-time, avoids timing attacks — not ==.
      4. On mismatch: return JSONResponse({"error": "unauthorized"},
         status_code=401)  (optionally headers={"WWW-Authenticate": "Bearer"}).
      5. On success: `return await call_next(request)`.
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
