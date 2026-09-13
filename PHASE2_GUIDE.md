# Phase 2 Guide — Build the MCP Server (stdio)

## Why this phase exists (plain words)

Phase 1 gave you Python functions (`score`, `explain`, ...). But a function in a
file isn't something Claude can call. **Phase 2 turns those functions into MCP
"tools"** — the callable units a Claude client sees on a menu and invokes.

You do that with one decorator: `@mcp.tool()`. It reads a function's **name,
type hints, and docstring** and builds the JSON schema Claude reads. So in this
phase the docstring stops being a comment and becomes the tool's **user manual**:
it's literally the text Claude uses to decide *whether* to call the tool and
*how* to fill its arguments. Write docstrings for Claude, not for yourself.

The only other new idea is the **transport** — how messages travel between the
client and your server. Phase 2 uses `stdio` (standard in/out), which is what
you use when the server runs on the same machine as the client. Phase 4 will
switch this single line to HTTP so it can run remotely.

```
Claude client  ──picks a tool off the menu──▶  @mcp.tool() functions
   (stdio)      ◀──gets back your dict result──   in server.py
                                                        │ call
                                                        ▼
                                              model_service (Phase 1)
```

## What you're filling in

Only one real file: `src/credit_risk_mcp/server.py`. It's already scaffolded
with the `FastMCP` app and 5 tool stubs. For each stub you:
1. **Expand the docstring** so it clearly tells Claude what the tool does.
2. **Fill the body** — almost always a one-liner that calls the matching
   `model_service` function.

| Tool | Body should call |
|------|------------------|
| `score_borrower` | `model_service.score(profile)` |
| `explain_prediction` | `model_service.explain(profile, top_n=top_n)` |
| `get_model_info` | `model_service.model_info()` |
| `get_feature_importance` | `model_service.global_importance(top_n=top_n)` |
| `compare_borrowers` | `model_service.score(...)` for each, then sort by `default_probability` desc |

`compare_borrowers` is the only one with logic: score each profile, tag each
result so applicants are distinguishable (e.g. their list index), sort by
probability descending, and return `{"ranked": [ {rank, index,
default_probability, risk_tier}, ... ]}` (the test expects a `"ranked"` list).

## Your grade: the tests

```bash
cd C:/Users/prana/Documents/Development/credit-risk-mcp
uv run --extra dev pytest -q
```
`tests/test_server.py` is your Phase 2 target (Phase 1 tests should stay green
too). It checks all 5 tools are registered *with descriptions* and that each is
wired correctly.

## See it run for real — MCP Inspector

Tests prove the wiring; Inspector lets you *see the menu Claude would see* and
click each tool by hand. This is the standard MCP debugging tool — learn it now,
you'll use it every phase.

1. Start the server under Inspector:
   ```bash
   npx @modelcontextprotocol/inspector uv run python -m credit_risk_mcp.server
   ```
   (First run downloads the inspector; it then opens a local web UI in your
   browser.)
2. In the UI: **List Tools** → you should see your 5 tools, each with the
   description (your docstring) and an input form built from `BorrowerProfile`.
3. Click `score_borrower`, fill `annual_income` and `credit_amount`, **Run**,
   and confirm you get a probability back. Try `get_model_info` (no inputs).

If a description looks thin or an input field is confusing, that's Claude's
experience too — go improve the docstring / field descriptions and reload.

## Run the server directly (optional sanity check)

```bash
uv run python -m credit_risk_mcp.server
```
It will sit silently waiting for a client on stdio (correct behaviour). Ctrl+C
to stop. Connecting it to Claude Desktop / Claude Code is **Phase 3** — don't
worry about that yet.

## Checklist
- [ ] all 5 tool bodies filled in
- [ ] each docstring reads like an instruction to Claude (what it does, what it
      returns, the educational-only caveat where relevant)
- [ ] `uv run --extra dev pytest -q` fully green
- [ ] you've listed and called at least 2 tools in MCP Inspector

When that's done, Phase 3 is just pointing a real Claude client at this server.
```bash
uv run --extra dev pytest -q
```
