# Phase 3 Guide — Connect the Server to a Real Claude

Phase 2 tested the server with Inspector. Now you connect it to an actual Claude
client so you can use it in plain conversation. **No server code changes** — this
phase is registration + testing + polishing docstrings based on what you observe.

## Key idea: host vs server

- Your `server.py` is the **server** (it exposes tools).
- Claude (Claude Code CLI, or Claude Desktop) is the **host/client** (it decides
  when to call them).
- "Connecting" = telling the client a shell command that launches your server.
  The client runs that command, speaks MCP over stdio, and the tools appear.

The launch command (same one Inspector used):
```
uv --directory "C:\Users\prana\Documents\Development\credit-risk-mcp" run python -m credit_risk_mcp.server
```

## Option A — Claude Code CLI (recommended, you already have it)

Register the server once (user scope = available in all your projects):
```bash
claude mcp add credit-risk -s user -- "C:\Users\prana\.local\bin\uv.exe" --directory "C:\Users\prana\Documents\Development\credit-risk-mcp" run python -m credit_risk_mcp.server
```
Everything after `--` is the launch command; `claude` runs it verbatim.

Verify and inspect:
```bash
claude mcp list
claude mcp get credit-risk
```
`list` should show `credit-risk` as connected. Then start a **new** Claude Code
session (servers load at session start) and try the test prompts below.

To remove it later: `claude mcp remove credit-risk`.

## Option B — Claude Desktop

Open the config file (create it if missing):
```
C:\Users\prana\AppData\Roaming\Claude\claude_desktop_config.json
```
Add (merge into `mcpServers` if the file already has one):
```json
{
  "mcpServers": {
    "credit-risk": {
      "command": "C:\\Users\\prana\\.local\\bin\\uv.exe",
      "args": [
        "--directory",
        "C:\\Users\\prana\\Documents\\Development\\credit-risk-mcp",
        "run",
        "python",
        "-m",
        "credit_risk_mcp.server"
      ]
    }
  }
}
```
Note the **doubled backslashes** (JSON escaping) and the **full path to uv.exe**
(the Desktop app may not have your PATH). Fully quit and reopen Claude Desktop;
the tools appear behind the tools/plug icon.

## Test prompts (this is the real work of Phase 3)

Run these in a conversation and watch which tool Claude picks and how it fills
arguments:

1. *"What can the credit-risk model tell me, and how accurate is it?"*
   → expect `get_model_info`.
2. *"Score a borrower: 34 years old, $54k annual income, wants a $450k loan,
   working, single, external credit scores around 0.15, 0.22, 0.10."*
   → expect `score_borrower` with those fields mapped correctly.
3. *"Why did it come out that risky?"* → expect `explain_prediction`.
4. *"What features matter most to this model overall?"* → `get_feature_importance`.
5. *"Compare two applicants: one on $200k with strong scores, one on $40k with
   weak scores."* → `compare_borrowers`, ranked.

### What to watch for (and fix via docstrings)
- **Unit confusion:** if you say "monthly income $4,500", does Claude annualise
  it (×12) before calling? If not, sharpen the `annual_income` description in
  `schemas.py` (e.g. "TOTAL YEARLY income").
- **Wrong tool:** if "why" triggers `score_borrower` instead of
  `explain_prediction`, make the explain docstring more clearly about reasons.
- **Made-up fields:** if Claude invents values you didn't give, that's fine —
  they default to medians — but check the `fields_defaulted` list is surfaced.

Change a docstring → restart the session/app → re-test. That loop *is* Phase 3.

## Optional polish: graceful errors
Try an invalid input in Inspector or chat (e.g. `annual_income: -5`). Pydantic
already returns a structured validation error to the client rather than crashing
the server — good enough for a portfolio. If you want friendlier messages later,
you can catch `ValidationError` in the tools, but it's not required.

## Checklist
- [ ] server registered (Option A or B) and shows as connected
- [ ] all 5 test prompts call the right tool with sensible arguments
- [ ] docstrings tightened for anything Claude got wrong
- [ ] you've seen a full score → explain conversation end to end

When this works, **Phase 4** makes it remote: swap stdio for HTTP + add an API
key so someone else's Claude can call it too.
```

## The one-line version to run now
```
claude mcp add credit-risk -s user -- "C:\Users\prana\.local\bin\uv.exe" --directory "C:\Users\prana\Documents\Development\credit-risk-mcp" run python -m credit_risk_mcp.server
```
