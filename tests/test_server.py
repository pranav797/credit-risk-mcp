"""Phase 2 tests: the MCP tools are registered and wired to model_service.

These call the decorated tool functions directly (FastMCP's @mcp.tool() returns
the original function) and also inspect what a client would actually see via
mcp.list_tools(). Make them green by filling in server.py.
"""

from __future__ import annotations

import asyncio

from credit_risk_mcp import server
from credit_risk_mcp.schemas import BorrowerProfile

EXPECTED_TOOLS = {
    "score_borrower",
    "explain_prediction",
    "get_model_info",
    "get_feature_importance",
    "compare_borrowers",
}


def _profile(**kw) -> BorrowerProfile:
    base = {"annual_income": 150000, "credit_amount": 500000}
    base.update(kw)
    return BorrowerProfile(**base)


def test_all_tools_registered_with_descriptions():
    tools = asyncio.run(server.mcp.list_tools())
    names = {t.name for t in tools}
    assert EXPECTED_TOOLS <= names
    # Every tool must carry a non-empty description (its docstring) — that's what
    # Claude reads to decide when to call it.
    for t in tools:
        if t.name in EXPECTED_TOOLS:
            assert t.description and t.description.strip()


def test_score_borrower_tool():
    result = server.score_borrower(_profile())
    assert 0.0 <= result["default_probability"] <= 1.0
    assert result["risk_tier"] in {"low", "moderate", "elevated", "high"}


def test_explain_prediction_tool():
    result = server.explain_prediction(_profile(), top_n=5)
    assert len(result["top_factors"]) == 5


def test_get_model_info_tool():
    assert server.get_model_info()["n_features"] == 235


def test_get_feature_importance_tool():
    assert len(server.get_feature_importance(top_n=6)["top_features"]) == 6


def test_compare_borrowers_ranks_by_risk():
    safe = _profile(annual_income=300000, credit_amount=100000,
                    ext_source_1=0.8, ext_source_2=0.8, ext_source_3=0.8)
    risky = _profile(annual_income=40000, credit_amount=900000,
                     ext_source_1=0.05, ext_source_2=0.05, ext_source_3=0.05)
    result = server.compare_borrowers([safe, risky])
    ranked = result["ranked"]
    probs = [r["default_probability"] for r in ranked]
    assert probs == sorted(probs, reverse=True)  # most risky first
