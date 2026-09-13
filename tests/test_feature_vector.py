"""Tests for the friendly-profile -> 235-column expansion."""

from __future__ import annotations

import math

from credit_risk_mcp import feature_spec as fs
from credit_risk_mcp.model_service import _artifacts, build_feature_vector
from credit_risk_mcp.schemas import BorrowerProfile


def _minimal() -> BorrowerProfile:
    return BorrowerProfile(annual_income=150000, credit_amount=500000)


def test_vector_shape_and_order():
    frame, _ = build_feature_vector(_minimal())
    expected = _artifacts().feature_columns
    assert list(frame.columns) == expected
    assert frame.shape == (1, 235)


def test_untouched_fields_use_defaults():
    frame, defaulted = build_feature_vector(_minimal())
    defaults = _artifacts().defaults
    # A field the profile never touches must equal the exported median exactly.
    assert frame.iloc[0]["REGION_RATING_CLIENT"] == defaults["REGION_RATING_CLIENT"]
    # Everything optional was defaulted since we only supplied the two required.
    assert "gender" in defaulted
    assert "income_type" in defaulted


def test_required_fields_written():
    frame, _ = build_feature_vector(_minimal())
    assert frame.iloc[0]["AMT_INCOME_TOTAL"] == 150000
    assert frame.iloc[0]["AMT_CREDIT"] == 500000


def test_age_sets_days_birth_and_derived_age_years():
    frame, _ = build_feature_vector(
        BorrowerProfile(annual_income=150000, credit_amount=500000, age=40)
    )
    row = frame.iloc[0]
    assert math.isclose(row["DAYS_BIRTH"], -40 * fs.DAYS_PER_YEAR)
    assert math.isclose(row["AGE_YEARS"], 40.0, abs_tol=1e-6)


def test_ratios_recomputed_from_supplied_amounts():
    frame, _ = build_feature_vector(
        BorrowerProfile(annual_income=100000, credit_amount=300000, annuity=20000)
    )
    row = frame.iloc[0]
    assert math.isclose(row["CREDIT_INCOME_RATIO"], 3.0)
    assert math.isclose(row["ANNUITY_INCOME_RATIO"], 0.2)
    assert math.isclose(row["CREDIT_TERM"], 15.0)


def test_binary_encoding():
    frame, _ = build_feature_vector(
        BorrowerProfile(
            annual_income=150000, credit_amount=500000,
            gender="M", owns_car="Y", owns_realty="N",
        )
    )
    row = frame.iloc[0]
    assert row["CODE_GENDER"] == 1
    assert row["FLAG_OWN_CAR"] == 1
    assert row["FLAG_OWN_REALTY"] == 0


def test_onehot_lights_up_one_column():
    frame, _ = build_feature_vector(
        BorrowerProfile(
            annual_income=150000, credit_amount=500000, income_type="Working"
        )
    )
    row = frame.iloc[0]
    assert row["NAME_INCOME_TYPE_Working"] == 1
    # Every other category in the group is off.
    for col in _artifacts().feature_columns:
        if col.startswith("NAME_INCOME_TYPE_") and col != "NAME_INCOME_TYPE_Working":
            assert row[col] == 0


def test_onehot_reference_category_is_all_zero():
    # "Businessman" was dropped by drop_first -> all dummies must be 0.
    frame, _ = build_feature_vector(
        BorrowerProfile(
            annual_income=150000, credit_amount=500000, income_type="Businessman"
        )
    )
    row = frame.iloc[0]
    for col in _artifacts().feature_columns:
        if col.startswith("NAME_INCOME_TYPE_"):
            assert row[col] == 0
