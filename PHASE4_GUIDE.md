# Phase 4 Guide — Remote Transport + Auth (the "learn MCP" milestone)

Phases 2-3 ran the server locally over **stdio**: Claude launched it as a
subprocess on your own machine. Phase 4 makes it a real **networked service** so
*someone else's* Claude could call it over the internet. Three moves:

1. **Transport:** stdio → **Streamable HTTP** (the server listens on a URL).
2. **Auth:** because a URL is reachable by anyone, require a secret key on every
   request (`Authorization: Bearer <key>`).
3. **Deploy:** run it on a free host and point a Claude client at the URL.

Only #2 is fill-in code. #1 and #3 are wiring you read + config you run.

## Why auth appears only now

Over stdio there's nothing to authenticate — the client *is* the parent process,
so trust is implicit. The moment the server answers HTTP requests from the open
internet, anyone with the URL can call your tools unless you gate them. That gate
is the whole security lesson of MCP servers: **a remote MCP server must
authenticate every request.**

```
stdio (local):   Claude ──spawns──▶ server            (trust implicit)
HTTP (remote):   Claude ──HTTPS + Bearer key──▶ server (must verify the key)
```

## What you fill in: `src/credit_risk_mcp/auth.py`

Implement `BearerAuthMiddleware.dispatch`. The docstring lists the exact steps;
the shape is: let `/health` through, load the expected key from the env var,
compare it (constant-time via `hmac.compare_digest`) to the `Bearer` token in the
`Authorization` header, return **401** on mismatch, **500** if the key isn't
configured (fail closed), otherwise call the next handler.

The transport switch and the ASGI wiring are already done for you in
`server.py` — read `build_http_app()` and `serve_http()` to see how the health
route and your middleware wrap the MCP app. `main()` still runs stdio, so your
Phase 3 local setup keeps working unchanged.

## Your grade: the tests
```bash
cd C:/Users/prana/Documents/Development/credit-risk-mcp
uv run --extra dev pytest -q
```
`tests/test_auth.py` checks: `/health` is public, `/mcp` needs a key, wrong key →
401, right key gets past auth, and no key configured → 500. Green = auth done.

## Run it over HTTP locally

Generate a key once:
```bash
uv run python -c "import secrets; print(secrets.token_urlsafe(32))"
```
Start the HTTP server (PowerShell):
```powershell
$env:CREDIT_RISK_API_KEY="paste-the-key-here"; uv run credit-risk-http
```
It now listens on http://localhost:8000 (MCP at `/mcp`, health at `/health`).

Smoke-test from another terminal:
```bash
curl http://localhost:8000/health                 # -> {"status":"ok"}
curl -X POST http://localhost:8000/mcp            # -> 401
curl -X POST http://localhost:8000/mcp -H "Authorization: Bearer paste-the-key-here"   # -> past auth (not 401)
```
Or point **MCP Inspector** at it: run `npx @modelcontextprotocol/inspector`,
choose transport **Streamable HTTP**, URL `http://localhost:8000/mcp`, and add an
`Authorization: Bearer <key>` header.

Connect a **Claude client** to the local HTTP server:
```bash
claude mcp add credit-risk-http --transport http http://localhost:8000/mcp --header "Authorization: Bearer paste-the-key-here"
```
(This is separate from your stdio `credit-risk` registration — keep both.)

## Deploy to a free host (Render, Docker path)

Prereq: push this project to a GitHub repo (see "Ship it" below).

1. On https://render.com → **New → Web Service** → connect the repo.
2. Environment: **Docker** (it auto-detects the `Dockerfile`).
3. Add an environment variable **`CREDIT_RISK_API_KEY`** = your key (mark secret).
   Render injects `$PORT` automatically; `serve_http()` already reads it.
4. Health check path: `/health`.
5. Deploy. You get a URL like `https://credit-risk-mcp.onrender.com`.

Connect Claude to the deployed server:
```bash
claude mcp add credit-risk-remote --transport http https://YOUR-APP.onrender.com/mcp --header "Authorization: Bearer YOUR_KEY"
```
Railway/Fly work the same way (Dockerfile + a `CREDIT_RISK_API_KEY` secret + the
injected `$PORT`). **Free-tier note:** these sleep when idle, so the first call
after a pause is slow — fine for a demo, just warm it with a `/health` hit first.

## Where real OAuth would go (talking point)

API keys are the simplest real auth and enough here. The MCP spec also defines an
OAuth 2 flow, and the Python SDK exposes a `TokenVerifier` / `auth=` hook on
`FastMCP` where you'd validate real access tokens instead of a shared secret.
Knowing *where* that slots in (and why you chose a key for a portfolio demo) is a
good interview answer.

## Ship it (git)
```bash
cd C:/Users/prana/Documents/Development/credit-risk-mcp
git init && git add -A && git commit -m "Credit risk MCP server (phases 1-4)"
```
Then create a GitHub repo and push. `.gitignore` keeps `.venv` out but **keeps
`artifacts/`** (the model), so the Docker build has what it needs.

## Checklist
- [ ] `BearerAuthMiddleware.dispatch` implemented; `pytest -q` fully green
- [ ] server runs via `uv run credit-risk-http`; curl smoke-tests behave
- [ ] deployed to a host with `CREDIT_RISK_API_KEY` set; `/health` returns 200
- [ ] a Claude client calls the remote URL and scores a borrower
- [ ] you can explain: stdio vs HTTP, why remote needs auth, where OAuth fits

Phase 5 is portfolio polish — README, a demo script, screenshots of the Claude
conversation.
```bash
uv run --extra dev pytest -q
```
