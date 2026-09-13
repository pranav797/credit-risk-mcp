# Credit Risk MCP Server — Project Plan

> Wrap the Home Credit Default Risk XGBoost model (from
> `C:\Users\prana\Documents\Development\Antigravity\home-credit-default`) in an
> MCP server so any Claude client can score borrowers, explain predictions, and
> query the model conversationally. Portfolio piece + hands-on MCP learning project.

---

## 1. Goal & Success Criteria

**Goal:** A working MCP server, built end-to-end (tool definitions → request
handling → auth → deployment), that exposes the trained credit risk model as
callable tools.

**Done means:**
1. Claude Desktop / Claude Code connects to the server and lists its tools.
2. A plain-English prompt ("score this borrower: 34yo, $54k income, ...")
   produces a correct default probability + SHAP explanation, with no notebook involved.
3. The server runs remotely over HTTP with an API key, so *someone else's*
   Claude could call it (the "other people's Claude" part of the pitch).
4. A README with a 60-second demo script exists — you can run the demo cold in
   an interview.

**Explicitly out of scope (v1):** retraining the model, a web UI, multi-model
support, real OAuth flows (API-key auth is enough to learn the concept).

---

## 2. What Exists Already (inputs from home-credit-default)

| Asset | Path | Role here |
|---|---|---|
| Trained model | `outputs/model.pkl` (XGBClassifier, 235 features) | The thing we serve |
| Processed data | `data/processed_data.csv` (307k × 235) | Source for feature medians/defaults + column order |
| Feature engineering logic | `notebooks/02_feature_engineering.ipynb` | Must be re-implemented as a function |
| SHAP setup | `notebooks/04_explainability.ipynb` | Re-implemented for single-row explanations |
| Threshold analysis | README (0.15 recall-oriented threshold) | Baked into `risk_tier` output |

**Key technical problem to solve early:** the model expects a 235-column
one-hot-encoded vector, but a user will only give ~8–12 human fields (age,
income, credit amount, employment type, ...). The server needs a
`build_feature_vector(profile)` function that:
- starts from a default row (training medians for numeric, all-zeros for one-hot groups),
- overwrites the fields the user provided (mapping e.g. `employment="Working"` → `NAME_INCOME_TYPE_Working=1`),
- preserves the exact training column order (load once from `processed_data.csv` header or persist as `feature_columns.json`).

This mapping module is ~half the real work and is a great interview talking
point ("productionizing a notebook model").

---

## 3. Architecture

```
Claude Desktop / Claude Code / other MCP clients
        │  (MCP protocol: stdio locally, Streamable HTTP remotely)
        ▼
┌──────────────────────────────────────────────┐
│  server.py  (FastMCP — official Python SDK)  │
│  ── tools ──                                 │
│   score_borrower(profile) → prob, tier       │
│   explain_prediction(profile) → SHAP factors │
│   get_model_info() → AUC, threshold, dataset │
│   get_feature_importance(top_n) → global SHAP│
│   compare_borrowers([profiles]) → ranked     │
├──────────────────────────────────────────────┤
│  model_service.py                            │
│   load model.pkl once at startup             │
│   build_feature_vector(profile)              │
│   predict(), shap_explain()                  │
├──────────────────────────────────────────────┤
│  artifacts/  model.pkl, feature_columns.json,│
│              feature_defaults.json (medians) │
└──────────────────────────────────────────────┘
```

**Stack:** Python 3.12/3.13, `mcp` package (FastMCP), `xgboost`, `shap`,
`pandas`, `pydantic` for input validation. `uv` for env management (what MCP
docs use). No database needed.

**Why FastMCP:** decorator-based (`@mcp.tool()`), auto-generates the JSON
schema Claude sees from your type hints + docstrings — you learn tool
definitions without hand-writing JSON Schema.

---

## 4. Phases

### Phase 0 — Learn the ground you're standing on (0.5 day)
- Read: MCP intro (modelcontextprotocol.io), "Build an MCP server" quickstart,
  concepts pages for **tools**, **transports** (stdio vs Streamable HTTP), and
  **authorization**.
- Run the official quickstart weather-server example locally and connect it to
  Claude Desktop, just to see the plumbing work once before touching ML code.
- **Checkpoint:** you can explain to yourself what a tool definition, a
  transport, and a client are.

### Phase 1 — Extract the model out of the notebooks (1 day) ✅ DONE
> Completed 2026-07-10. All artifacts exported, model_service works end-to-end,
> 13/13 tests pass, golden predictions match the original pickle to 8 decimals.

- Create repo skeleton: `credit-risk-mcp/` with `pyproject.toml`, `src/`, `artifacts/`, `tests/`.
- Copy `model.pkl` into `artifacts/` (never load from the old project path).
- One-off script `scripts/export_artifacts.py`: reads `processed_data.csv`
  from the old project once, writes `feature_columns.json` (ordered list) and
  `feature_defaults.json` (median of each numeric column, 0 for one-hot). After
  this, the MCP project has **no dependency** on the 362 MB CSV.
- Write `model_service.py`: `BorrowerProfile` pydantic model (~10 friendly
  fields, all optional except a few), `build_feature_vector()`, `predict()`.
- Write pytest tests: known profile → probability in [0,1]; column order
  matches model booster feature names; missing fields fall back to defaults.
- **Checkpoint:** `python -c "..."` scores a hardcoded borrower correctly, no notebooks.

### Phase 2 — Build the MCP server, stdio transport (1 day)
- `server.py` with FastMCP; implement the 5 tools. Return structured dicts and
  write rich docstrings — the docstring IS the UX, it's what Claude reads to
  decide when/how to call your tool.
- `explain_prediction`: SHAP TreeExplainer on the single row, return top ±8
  features with human-readable names (map `EXT_SOURCE_3` → "external bureau
  credit score #3") and signed contributions.
- Test with **MCP Inspector** (`npx @modelcontextprotocol/inspector`) — this is
  the standard debug tool; learn it.
- **Checkpoint:** every tool callable and correct in Inspector.

### Phase 3 — Connect to Claude locally (0.5 day)
- Register the server in Claude Desktop (`claude_desktop_config.json`) and/or
  Claude Code (`claude mcp add`).
- Run realistic conversations; iterate on docstrings and error messages until
  Claude reliably picks the right tool and fills fields sensibly (e.g. does it
  convert "monthly income $4,500" to annual? Add units to the schema!).
- Handle bad input gracefully: out-of-range values return a helpful error
  string, not a stack trace.
- **Checkpoint:** the example user flow works end-to-end in Claude Desktop.

### Phase 4 — Remote transport + auth (1–1.5 days) — the "learn MCP" meat
- Switch transport to **Streamable HTTP** (`mcp.run(transport="streamable-http")`).
- Add API-key auth: require `Authorization: Bearer <key>` header, validate in a
  middleware; keys in env vars. (Read the MCP authorization spec so you can
  discuss where OAuth would slot in, even though you're shipping API keys.)
- Deploy to one free-tier host (Render / Railway / Fly.io). Dockerfile is a
  nice bonus but optional.
- Connect Claude to the remote URL; confirm a machine that isn't yours could do the same with a key.
- **Checkpoint:** `curl` without a key → 401; Claude with key + remote URL → scores borrowers.

### Phase 5 — Portfolio polish (0.5–1 day)
- README: problem statement, architecture diagram, GIF/screenshots of the
  Claude conversation, honest limitations section (simulated data caveats,
  AUC 0.75, threshold trade-off).
- `DEMO.md`: exact 60-second interview script — which prompts to type, what to
  point at while it runs.
- Write 3–4 interview talking points: notebook→production refactor, why
  docstrings are the API contract with an LLM, stdio vs HTTP transports,
  recall-vs-precision threshold reasoning.
- Push to GitHub; add to resume/portfolio.

**Total estimate: ~5 focused days**, comfortably 1.5–2 weeks part-time.

---

## 5. Tool Specifications (v1)

| Tool | Inputs | Output | Notes |
|---|---|---|---|
| `score_borrower` | BorrowerProfile (age, annual_income, credit_amount, annuity, employment_type, education, family_status, owns_car, owns_realty, ext_source_1/2/3, gender) — all optional except credit_amount & annual_income | `{default_probability, risk_tier, threshold_used, fields_defaulted}` | `fields_defaulted` lists which features fell back to medians — honesty signal |
| `explain_prediction` | same profile | `{default_probability, top_factors: [{feature, friendly_name, value, shap_contribution, direction}]}` | ±8 factors |
| `get_model_info` | — | model type, AUC 0.75, precision/recall at 0.50 vs 0.15, training data size, class imbalance, limitations | static dict |
| `get_feature_importance` | `top_n` (default 10) | global mean |SHAP| ranking with friendly names | precompute at build time, ship as JSON |
| `compare_borrowers` | list of 2–10 profiles | ranked list with probabilities | reuses score_borrower |

---

## 6. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Feature-vector mismatch (wrong column order → silent garbage predictions) | Assert columns == `model.get_booster().feature_names` at startup; test that reproduces a known notebook prediction |
| SHAP slow at request time | TreeExplainer on 1 row is fast (~ms); precompute the global importance JSON |
| `model.pkl` pickle/version incompatibility across Python versions | Pin xgboost version; consider re-saving via `model.save_model("model.json")` (xgboost-native, portable) — recommended |
| Someone treats this as real credit advice | Disclaimer in `get_model_info` and in every tool docstring: educational Kaggle project, not a lending decision system |
| Free-tier host cold starts kill the demo | Also keep the stdio/local setup as the primary interview demo; remote is the "and it's deployed" flourish |

---

## 7. Repo Layout (target)

```
credit-risk-mcp/
├── PROJECT_PLAN.md          # this file
├── README.md                # portfolio-facing
├── DEMO.md                  # interview demo script
├── pyproject.toml
├── artifacts/
│   ├── model.json           # re-saved xgboost model (portable)
│   ├── feature_columns.json
│   ├── feature_defaults.json
│   └── global_importance.json
├── scripts/
│   └── export_artifacts.py  # one-off: old project → artifacts/
├── src/credit_risk_mcp/
│   ├── server.py            # FastMCP app + tool definitions
│   ├── model_service.py     # load, build_feature_vector, predict, explain
│   ├── schemas.py           # pydantic BorrowerProfile
│   └── auth.py              # bearer-key middleware (Phase 4)
└── tests/
    ├── test_feature_vector.py
    ├── test_predictions.py
    └── test_tools.py
```
