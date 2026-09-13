"""Prediction-correctness tests, including a golden regression against the
original pickled model's outputs."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from credit_risk_mcp.model_service import (
    _artifacts,
    explain,
    global_importance,
    model_info,
    score,
)
from credit_risk_mcp.schemas import BorrowerProfile

FIXTURE = Path(__file__).parent / "fixtures" / "golden_predictions.json"


def test_golden_predictions_match_original_model():
    """model.json must reproduce the pickled model's probabilities exactly."""
    fixture = json.loads(FIXTURE.read_text())
    art = _artifacts()
    assert fixture["feature_columns"] == art.feature_columns
    frame = pd.DataFrame(
        [c["row"] for c in fixture["cases"]], columns=fixture["feature_columns"]
    )
    probs = art.model.predict_proba(frame)[:, 1]
    for got, case in zip(probs, fixture["cases"]):
        assert math.isclose(got, case["expected_prob"], rel_tol=1e-5, abs_tol=1e-6)


def test_score_returns_valid_probability():
    result = score(BorrowerProfile(annual_income=150000, credit_amount=500000))
    assert 0.0 <= result["default_probability"] <= 1.0
    assert result["risk_tier"] in {"low", "moderate", "elevated", "high"}
    assert result["threshold_used"] == 0.15


def test_high_risk_profile_scores_higher_than_low_risk():
    low = score(
        BorrowerProfile(
            annual_income=300000, credit_amount=200000,
            ext_source_1=0.8, ext_source_2=0.8, ext_source_3=0.8,
        )
    )
    high = score(
        BorrowerProfile(
            annual_income=40000, credit_amount=900000,
            ext_source_1=0.05, ext_source_2=0.05, ext_source_3=0.05,
        )
    )
    assert high["default_probability"] > low["default_probability"]


def test_explain_returns_ranked_factors():
    result = explain(
        BorrowerProfile(annual_income=100000, credit_amount=600000), top_n=8
    )
    factors = result["top_factors"]
    assert len(factors) == 8
    mags = [abs(f["shap_contribution"]) for f in factors]
    assert mags == sorted(mags, reverse=True)  # ranked by magnitude
    assert all(f["direction"] in {"increases risk", "decreases risk"} for f in factors)


def test_model_info_and_global_importance():
    info = model_info()
    assert info["n_features"] == 235
    assert "limitations" in info

    gi = global_importance(top_n=5)
    assert len(gi["top_features"]) == 5
    # EXT_SOURCE features dominate this model.
    assert gi["top_features"][0]["feature"].startswith("EXT_SOURCE")
