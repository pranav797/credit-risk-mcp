# Credit Risk MCP Server

An MCP (Model Context Protocol) server that exposes a trained **Home Credit
Default Risk** XGBoost model as tools any Claude client can call — score a
borrower, explain the prediction with SHAP, and query model metadata, all in
plain conversation.

> Portfolio piece + hands-on MCP learning project. Wraps the model from
> [`home-credit-default`](../Antigravity/home-credit-default). **Educational —
> not a real lending decision system.**

## Status

- [x] **Phase 1 — model extraction** (this milestone): model pulled out of the
      notebooks into a proper Python package; friendly 13-field profile expands
      to the model's 235 columns; SHAP explanations; full test suite.
- [ ] Phase 2 — FastMCP server (stdio) exposing the tools
- [ ] Phase 3 — connect to Claude Desktop / Claude Code
- [ ] Phase 4 — HTTP transport + API-key auth + deploy
- [ ] Phase 5 — portfolio polish (demo script, screenshots)

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the full plan.

## What's here (Phase 1)

```
src/credit_risk_mcp/
  feature_spec.py    encodings, one-hot groups, engineered-feature formulas
  schemas.py         BorrowerProfile — the friendly ~13-field input
  model_service.py   load model, build_feature_vector, score, explain
scripts/
  export_artifacts.py  one-off: old project -> artifacts/ (run once)
artifacts/           model.json + feature_columns/defaults/global_importance
tests/               13 tests incl. a golden regression vs the original pickle
```

## Setup

```bash
uv sync
# regenerate artifacts from the original project (only needed once / if it moves):
uv run python scripts/export_artifacts.py
uv run --extra dev pytest -q
```

## The model

XGBoost classifier, AUC-ROC ≈ 0.75, trained on the Kaggle Home Credit dataset
(~308k applicants, 235 features). Uses the recall-oriented **0.15** decision
threshold from the modelling notebook — in credit risk, a missed defaulter costs
far more than a false alarm. External credit-bureau scores (`EXT_SOURCE_1/2/3`)
dominate the predictions.
