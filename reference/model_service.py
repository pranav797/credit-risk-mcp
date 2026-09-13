"""Load the trained model once and turn a BorrowerProfile into predictions.

This is the bridge between the friendly ``BorrowerProfile`` schema and the raw
235-column vector the XGBoost model expects. The hard part -- and the main
"productionise a notebook" story of this project -- is ``build_feature_vector``:
a user gives ~10 fields, the model needs 235, so we start from training-median
defaults and overlay the supplied fields using the notebook's exact encodings.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from . import feature_spec as fs
from .schemas import BorrowerProfile

ARTIFACTS = Path(__file__).resolve().parent.parent.parent / "artifacts"

# Recall-oriented business threshold chosen in 03_modelling.ipynb: the cost of a
# missed defaulter (lost principal) far outweighs a false positive (lost interest).
DEFAULT_THRESHOLD = 0.15

# Profile field -> the raw model column it writes. One-hot / engineered fields
# are handled separately below.
_DIRECT_COLUMNS: dict[str, str] = {
    "annual_income": "AMT_INCOME_TOTAL",
    "credit_amount": "AMT_CREDIT",
    "annuity": "AMT_ANNUITY",
    "num_children": "CNT_CHILDREN",
    "ext_source_1": "EXT_SOURCE_1",
    "ext_source_2": "EXT_SOURCE_2",
    "ext_source_3": "EXT_SOURCE_3",
}
_BINARY_FIELDS: dict[str, str] = {
    "gender": "CODE_GENDER",
    "owns_car": "FLAG_OWN_CAR",
    "owns_realty": "FLAG_OWN_REALTY",
}
_ONEHOT_FIELDS: dict[str, str] = {
    "income_type": "NAME_INCOME_TYPE",
    "education": "NAME_EDUCATION_TYPE",
    "family_status": "NAME_FAMILY_STATUS",
    "housing_type": "NAME_HOUSING_TYPE",
}


class _Artifacts:
    """Lazily-loaded model + supporting JSON, kept as a process-wide singleton."""

    def __init__(self) -> None:
        self.feature_columns: list[str] = json.loads(
            (ARTIFACTS / "feature_columns.json").read_text()
        )
        self.defaults: dict[str, float] = json.loads(
            (ARTIFACTS / "feature_defaults.json").read_text()
        )
        self.model = xgb.XGBClassifier()
        self.model.load_model(ARTIFACTS / "model.json")

        booster_names = list(self.model.get_booster().feature_names)
        if booster_names != self.feature_columns:
            raise RuntimeError(
                "feature_columns.json is out of sync with the model's own feature "
                "names -- re-run scripts/export_artifacts.py."
            )
        self._explainer: Any = None

    @property
    def explainer(self) -> Any:
        if self._explainer is None:
            import shap

            self._explainer = shap.TreeExplainer(self.model)
        return self._explainer


@lru_cache(maxsize=1)
def _artifacts() -> _Artifacts:
    return _Artifacts()


def build_feature_vector(profile: BorrowerProfile) -> tuple[pd.DataFrame, list[str]]:
    """Expand a friendly profile into the model's full 235-column row.

    Returns the one-row DataFrame (columns in exact model order) and the list of
    profile fields that fell back to defaults (for transparency in the output).
    """
    art = _artifacts()
    row: dict[str, float] = dict(art.defaults)  # start from training medians
    touched_raw: set[str] = set()
    provided: set[str] = set()

    def set_raw(col: str, value: float) -> None:
        row[col] = float(value)
        touched_raw.add(col)

    # Direct numeric columns.
    for field, col in _DIRECT_COLUMNS.items():
        value = getattr(profile, field)
        if value is not None:
            set_raw(col, value)
            provided.add(field)

    # Binary label-encoded columns.
    for field, col in _BINARY_FIELDS.items():
        value = getattr(profile, field)
        if value is not None:
            set_raw(col, fs.BINARY_ENCODINGS[col][value])
            provided.add(field)

    # Age and employment: user gives years, model stores signed days.
    if profile.age is not None:
        set_raw("DAYS_BIRTH", -profile.age * fs.DAYS_PER_YEAR)
        provided.add("age")
    if profile.years_employed is not None:
        set_raw("DAYS_EMPLOYED", -profile.years_employed * fs.DAYS_PER_YEAR)
        provided.add("years_employed")

    # One-hot groups: zero the whole group, then light up the chosen category
    # (reference category stays all-zero).
    for field, group in _ONEHOT_FIELDS.items():
        value = getattr(profile, field)
        if value is None:
            continue
        provided.add(field)
        for col in art.feature_columns:
            if col.startswith(group + "_"):
                row[col] = 0.0
        suffix = fs.ONEHOT_GROUPS[group][value]
        if suffix is not None:
            row[f"{group}_{suffix}"] = 1.0

    # Recompute engineered features whose raw inputs were supplied, so the row
    # stays internally consistent (e.g. age set -> AGE_YEARS follows).
    for derived, (inputs, formula) in fs.DERIVED_FEATURES.items():
        if any(col in touched_raw for col in inputs):
            row[derived] = float(formula(row))

    frame = pd.DataFrame([row])[art.feature_columns]

    all_optional = (
        set(_DIRECT_COLUMNS) - {"annual_income", "credit_amount"}
        | set(_BINARY_FIELDS)
        | set(_ONEHOT_FIELDS)
        | {"age", "years_employed", "annuity"}
    )
    defaulted = sorted(all_optional - provided)
    return frame, defaulted


def _risk_tier(prob: float) -> str:
    if prob < 0.08:
        return "low"
    if prob < DEFAULT_THRESHOLD:
        return "moderate"
    if prob < 0.30:
        return "elevated"
    return "high"


def score(profile: BorrowerProfile) -> dict[str, Any]:
    """Predict default probability and derive a risk tier + decision flag."""
    art = _artifacts()
    frame, defaulted = build_feature_vector(profile)
    prob = float(art.model.predict_proba(frame)[0, 1])
    return {
        "default_probability": round(prob, 4),
        "risk_tier": _risk_tier(prob),
        "flag_default": prob >= DEFAULT_THRESHOLD,
        "threshold_used": DEFAULT_THRESHOLD,
        "fields_defaulted": defaulted,
    }


def explain(profile: BorrowerProfile, top_n: int = 8) -> dict[str, Any]:
    """SHAP breakdown of a single prediction: the biggest push/pull factors."""
    art = _artifacts()
    frame, defaulted = build_feature_vector(profile)
    prob = float(art.model.predict_proba(frame)[0, 1])

    shap_values = np.asarray(art.explainer.shap_values(frame))
    contributions = shap_values.reshape(len(art.feature_columns))

    order = np.argsort(np.abs(contributions))[::-1][:top_n]
    factors = []
    for idx in order:
        col = art.feature_columns[idx]
        factors.append(
            {
                "feature": col,
                "friendly_name": fs.friendly_name(col),
                "value": round(float(frame.iloc[0, idx]), 4),
                "shap_contribution": round(float(contributions[idx]), 4),
                "direction": "increases risk"
                if contributions[idx] > 0
                else "decreases risk",
            }
        )
    return {
        "default_probability": round(prob, 4),
        "risk_tier": _risk_tier(prob),
        "top_factors": factors,
        "fields_defaulted": defaulted,
    }


def model_info() -> dict[str, Any]:
    """Static description of the model, its performance, and its limits."""
    return {
        "model_type": "XGBoost classifier (XGBClassifier, 100 trees, max_depth=5)",
        "n_features": len(_artifacts().feature_columns),
        "training_rows": 307511,
        "class_balance": "~91.9% repaid / ~8.1% defaulted (SMOTE-balanced for training)",
        "auc_roc": 0.7476,
        "performance": {
            "threshold_0.50": {"precision": 0.613, "recall": 0.009},
            "threshold_0.15": {"precision": 0.212, "recall": 0.474},
        },
        "recommended_threshold": DEFAULT_THRESHOLD,
        "dataset": "Home Credit Default Risk (Kaggle)",
        "limitations": (
            "Educational Kaggle project, NOT a real lending decision system. "
            "Trained on a single application table (no bureau/history joins). "
            "AUC ~0.75 means meaningful but imperfect discrimination. Do not use "
            "for real credit decisions."
        ),
    }


def global_importance(top_n: int = 10) -> dict[str, Any]:
    """Top features by mean |SHAP| across a sample, precomputed at export time."""
    ranking = json.loads((ARTIFACTS / "global_importance.json").read_text())
    top = ranking[:top_n]
    for item in top:
        item["friendly_name"] = fs.friendly_name(item["feature"])
    return {"top_features": top}
