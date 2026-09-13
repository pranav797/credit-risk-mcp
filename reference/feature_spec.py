"""How a human-friendly borrower description maps onto the model's 235 columns.

This module is the single source of truth for the preprocessing that
``02_feature_engineering.ipynb`` did inline. Everything here mirrors that
notebook exactly so predictions match what the model was trained on:

  * binary label encodings (CODE_GENDER: M=1/F=0, FLAG_OWN_CAR: Y=1/N=0, ...)
  * one-hot groups created with pd.get_dummies(drop_first=True) -- the dropped
    (alphabetically first) category is the REFERENCE level = all dummies 0
  * engineered features derived from the raw columns (AGE_YEARS, the ratios, ...)

Keeping it declarative means the mapping is easy to audit against the notebook.
"""

from __future__ import annotations

from typing import Callable

DAYS_PER_YEAR = 365.25

# --- Binary label-encoded columns: friendly value -> encoded number ----------
# Mirrors the .map({...}) calls in the feature-engineering notebook.
BINARY_ENCODINGS: dict[str, dict[str, int]] = {
    "CODE_GENDER": {"M": 1, "F": 0},
    "FLAG_OWN_CAR": {"Y": 1, "N": 0},
    "FLAG_OWN_REALTY": {"Y": 1, "N": 0},
    "NAME_CONTRACT_TYPE": {"Cash loans": 1, "Revolving loans": 0},
}

# --- One-hot groups: friendly category -> retained column suffix --------------
# The reference (dropped-by-drop_first) category maps to None, meaning "leave
# every dummy in the group at 0". These retained suffixes were read directly
# off the trained model's feature names.
ONEHOT_GROUPS: dict[str, dict[str, str | None]] = {
    "NAME_INCOME_TYPE": {
        "Businessman": None,  # reference level (dropped by drop_first)
        "Commercial associate": "Commercial associate",
        "Maternity leave": "Maternity leave",
        "Pensioner": "Pensioner",
        "State servant": "State servant",
        "Student": "Student",
        "Unemployed": "Unemployed",
        "Working": "Working",
    },
    "NAME_EDUCATION_TYPE": {
        "Academic degree": None,  # reference level
        "Higher education": "Higher education",
        "Incomplete higher": "Incomplete higher",
        "Lower secondary": "Lower secondary",
        "Secondary / secondary special": "Secondary / secondary special",
    },
    "NAME_FAMILY_STATUS": {
        "Civil marriage": None,  # reference level
        "Married": "Married",
        "Separated": "Separated",
        "Single / not married": "Single / not married",
        "Unknown": "Unknown",
        "Widow": "Widow",
    },
    "NAME_HOUSING_TYPE": {
        "Co-op apartment": None,  # reference level
        "House / apartment": "House / apartment",
        "Municipal apartment": "Municipal apartment",
        "Office apartment": "Office apartment",
        "Rented apartment": "Rented apartment",
        "With parents": "With parents",
    },
}

# --- Engineered features: derived column -> (inputs, formula) -----------------
# Recomputed only when one of their raw inputs was supplied by the user, so an
# untouched default row keeps its exact training-median values.
DerivedFn = Callable[[dict[str, float]], float]
DERIVED_FEATURES: dict[str, tuple[tuple[str, ...], DerivedFn]] = {
    "AGE_YEARS": (("DAYS_BIRTH",), lambda r: abs(r["DAYS_BIRTH"]) / DAYS_PER_YEAR),
    "YEARS_EMPLOYED": (
        ("DAYS_EMPLOYED",),
        lambda r: abs(r["DAYS_EMPLOYED"]) / DAYS_PER_YEAR,
    ),
    "YEARS_ID_PUBLISH": (
        ("DAYS_ID_PUBLISH",),
        lambda r: abs(r["DAYS_ID_PUBLISH"]) / DAYS_PER_YEAR,
    ),
    "YEARS_REGISTRATION": (
        ("DAYS_REGISTRATION",),
        lambda r: abs(r["DAYS_REGISTRATION"]) / DAYS_PER_YEAR,
    ),
    "CREDIT_INCOME_RATIO": (
        ("AMT_CREDIT", "AMT_INCOME_TOTAL"),
        lambda r: r["AMT_CREDIT"] / r["AMT_INCOME_TOTAL"],
    ),
    "ANNUITY_INCOME_RATIO": (
        ("AMT_ANNUITY", "AMT_INCOME_TOTAL"),
        lambda r: r["AMT_ANNUITY"] / r["AMT_INCOME_TOTAL"],
    ),
    "CREDIT_TERM": (
        ("AMT_CREDIT", "AMT_ANNUITY"),
        lambda r: r["AMT_CREDIT"] / r["AMT_ANNUITY"],
    ),
}

# --- Human-readable names for model columns (used in SHAP explanations) -------
FRIENDLY_NAMES: dict[str, str] = {
    "EXT_SOURCE_1": "external bureau credit score #1",
    "EXT_SOURCE_2": "external bureau credit score #2",
    "EXT_SOURCE_3": "external bureau credit score #3",
    "AMT_CREDIT": "loan amount",
    "AMT_INCOME_TOTAL": "annual income",
    "AMT_ANNUITY": "loan annuity (per-period payment)",
    "AMT_GOODS_PRICE": "price of the financed goods",
    "AGE_YEARS": "age (years)",
    "DAYS_BIRTH": "age (in days, negative)",
    "YEARS_EMPLOYED": "years employed",
    "CREDIT_INCOME_RATIO": "loan-to-income ratio",
    "ANNUITY_INCOME_RATIO": "payment-to-income ratio",
    "CREDIT_TERM": "loan term (months)",
    "CODE_GENDER": "gender (M=1)",
    "FLAG_OWN_CAR": "owns a car",
    "FLAG_OWN_REALTY": "owns real estate",
    "CNT_CHILDREN": "number of children",
    "CNT_FAM_MEMBERS": "family size",
    "NAME_CONTRACT_TYPE": "cash loan (vs revolving)",
    "NAME_INCOME_TYPE_Working": "income type: working",
    "NAME_INCOME_TYPE_Pensioner": "income type: pensioner",
    "NAME_FAMILY_STATUS_Married": "married",
    "NAME_EDUCATION_TYPE_Higher education": "higher education",
    "REGION_RATING_CLIENT": "region rating",
    "DAYS_EMPLOYED": "days employed (negative)",
}


def friendly_name(column: str) -> str:
    """Human label for a model column, falling back to the raw name."""
    if column in FRIENDLY_NAMES:
        return FRIENDLY_NAMES[column]
    # Generic one-hot fallback: "NAME_FAMILY_STATUS_Widow" -> "family status: widow"
    for group in ONEHOT_GROUPS:
        if column.startswith(group + "_"):
            label = group.replace("NAME_", "").replace("_", " ").lower()
            value = column[len(group) + 1 :]
            return f"{label}: {value}"
    return column
