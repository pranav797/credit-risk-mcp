"""Tests for the HTTP transport's bearer-key auth middleware.

We build the ASGI app and drive it with Starlette's TestClient (no real network
needed).

Note: the MCP session manager can only be started once per process, so we build
the app ONCE (module-scoped) and share it. The auth middleware reads the key
from the environment on every request, so individual tests can still vary it.
"""

from __future__ import annotations

import os

import pytest
from starlette.testclient import TestClient

API_KEY = "test-key-123"


@pytest.fixture(scope="module")
def client():
    os.environ["CREDIT_RISK_API_KEY"] = API_KEY
    from credit_risk_mcp.server import build_http_app

    with TestClient(build_http_app()) as c:
        yield c


def test_health_is_public(client):
    # Health check must work with no key so the host can monitor the server.
    assert client.get("/health").status_code == 200


def test_mcp_requires_a_key(client):
    assert client.post("/mcp").status_code == 401


def test_mcp_rejects_wrong_key(client):
    r = client.post("/mcp", headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401


def test_mcp_accepts_correct_key(client):
    # A correct key gets PAST auth. The MCP layer may then reject for unrelated
    # reasons (missing session headers, host check) — but it must never be 401.
    r = client.post("/mcp", headers={"Authorization": f"Bearer {API_KEY}"})
    assert r.status_code != 401


def test_server_misconfigured_without_env_key(client, monkeypatch):
    # Fail closed: if the secret isn't configured, tools must not be served.
    monkeypatch.delenv("CREDIT_RISK_API_KEY", raising=False)
    r = client.post("/mcp", headers={"Authorization": "Bearer anything"})
    assert r.status_code == 500
