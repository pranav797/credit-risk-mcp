"""The MCP server: exposes the Phase 1 model functions as callable tools.

>>> SCAFFOLD (Phase 2) — fill in the tool bodies and docstrings.
>>> See PHASE2_GUIDE.md for the concepts, how to run it, and how to test with
>>> MCP Inspector.

Big picture: `model_service` already does the real work. This file is a thin
adapter that registers each of those functions as an MCP *tool* so a Claude
client can call them. The @mcp.tool() decorator turns a function's name, type
hints, and DOCSTRING into the schema Claude reads — so the docstring is the
tool's user manual, not just a comment. Write it for Claude.
"""

from __future__ import annotations

import os
from typing import Any

from mcp.server.fastmcp import FastMCP
from starlette.responses import JSONResponse

from . import model_service
from .auth import BearerAuthMiddleware
from .schemas import BorrowerProfile

# The server instance. The name is how the tool set shows up in a client.
mcp = FastMCP("credit-risk")


# NOTE on argument style: each tool below takes a `BorrowerProfile` (or a list
# of them). FastMCP reads that pydantic model — the same one you built in
# Phase 1 — to generate the input schema, so you get all your field
# descriptions and validation for free. (Alternative: list every field as a
# separate parameter. Reusing the model is DRY and less error-prone.)


@mcp.tool()
def score_borrower(profile: BorrowerProfile) -> dict[str, Any]:
    """Estimate a loan applicant's probability of default.

    Use this to get a single risk number for one borrower. Describe the
    applicant in plain terms — only annual income and loan amount are required;
    every other field falls back to the population median when omitted.

    Returns a dict with:
      - default_probability: estimated probability of default (0-1)
      - risk_tier: "low" / "moderate" / "elevated" / "high"
      - flag_default: True if the probability is at or above the 0.15 decision
        threshold (this model is tuned for recall, not a 50% cutoff)
      - threshold_used: the decision threshold (0.15)
      - fields_defaulted: which optional fields fell back to medians

    Educational model trained on a Kaggle dataset — not real lending advice.
    """
    return model_service.score(profile)


@mcp.tool()
def explain_prediction(profile: BorrowerProfile, top_n: int = 8) -> dict[str, Any]:
    """Explain a default-risk prediction: which factors push risk up or down.

    Use this when the user wants the "why" behind a score, not just the number.
    Runs SHAP on the single applicant and returns the strongest contributing
    features with human-readable names.

    Args:
      profile: the applicant to explain.
      top_n: how many contributing factors to return (default 8).

    Returns a dict with default_probability, risk_tier, and top_factors — a list
    of {feature, friendly_name, value, shap_contribution, direction} ordered by
    impact, where direction is "increases risk" or "decreases risk".

    Educational model — not real lending advice.
    """
    return model_service.explain(profile, top_n=top_n)


@mcp.tool()
def get_model_info() -> dict[str, Any]:
    """Describe the model itself: type, performance, and limitations.

    Use this to answer questions about how good the model is or how it should be
    used — no borrower needed. Returns model type, feature count, training size,
    AUC-ROC, precision/recall at the 0.50 and 0.15 thresholds, the recommended
    threshold, and an honest limitations note (educational, not real lending
    advice).
    """
    return model_service.model_info()


@mcp.tool()
def get_feature_importance(top_n: int = 10) -> dict[str, Any]:
    """Return the model's globally most important features (by mean |SHAP|).

    Use this for "what drives this model overall?" questions, as opposed to the
    reasons behind one borrower's score. Returns {"top_features": [...]} ranked
    by average absolute SHAP impact across a sample, each entry carrying its raw
    name, a friendly name, and its importance score.

    Args:
      top_n: how many features to return (default 10).
    """
    return model_service.global_importance(top_n=top_n)


@mcp.tool()
def compare_borrowers(profiles: list[BorrowerProfile]) -> dict[str, Any]:
    """Score several applicants at once and rank them from most to least risky.

    Use this to compare a batch of borrowers side by side. Each profile is
    scored exactly like score_borrower.

    Args:
      profiles: two or more applicants to compare.

    Returns {"ranked": [...]} sorted by default_probability descending. Each
    entry has: rank (1 = riskiest), index (position in the input list),
    default_probability, and risk_tier.

    Educational model — not real lending advice.
    """
    ranked = []
    for index, profile in enumerate(profiles):
        result = model_service.score(profile)
        ranked.append(
            {
                "index": index,
                "default_probability": result["default_probability"],
                "risk_tier": result["risk_tier"],
            }
        )
    ranked.sort(key=lambda r: r["default_probability"], reverse=True)
    for rank, entry in enumerate(ranked, start=1):
        entry["rank"] = rank
    return {"ranked": ranked}


def build_http_app():
    """Build the ASGI app for remote use: MCP over HTTP + auth + a health check.

    `mcp.streamable_http_app()` returns a Starlette app that serves the MCP
    protocol at /mcp (the "Streamable HTTP" transport — the HTTP equivalent of
    the stdio pipe used locally). On top of it we add:
      - a PUBLIC /health route (hosting platforms ping it; no key required), and
      - the bearer-key auth middleware, which guards everything else.
    """
    app = mcp.streamable_http_app()

    async def health(_request):
        return JSONResponse({"status": "ok"})

    app.add_route("/health", health, methods=["GET"])
    app.add_middleware(BearerAuthMiddleware)
    return app


def serve_http() -> None:
    """Run the server over HTTP. Entry point:  uv run credit-risk-http

    Host/port come from the environment so a deploy platform can inject them
    (Render, Railway, Fly all set $PORT). Set CREDIT_RISK_API_KEY too, or every
    request will be refused with 500 (auth fails closed).
    """
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(build_http_app(), host=host, port=port)


def main() -> None:
    """Local entry point: stdio transport (used by Claude Desktop / Claude Code).

    Run:  uv run python -m credit_risk_mcp.server
    """
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
