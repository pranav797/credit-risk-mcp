"""One-off exporter: turn the home-credit-default notebook artifacts into small,
portable files this MCP server owns.

Reads (from the original project, path via --source):
    outputs/model.pkl          the trained XGBClassifier (235 features)
    data/processed_data.csv     median-imputed training frame (source of defaults)

Writes (into ../artifacts):
    model.json                  xgboost-native, portable model (no pickle needed)
    feature_columns.json        the 235 feature names IN MODEL ORDER (canonical)
    feature_defaults.json       median of each feature -> the "typical applicant" row
    global_importance.json      top features by mean |SHAP| over a random sample

Run once:  uv run python scripts/export_artifacts.py
After this, the MCP server has no dependency on the 362 MB CSV or the old project.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SOURCE = Path(
    r"C:\Users\prana\Documents\Development\Antigravity\home-credit-default"
)
ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
SHAP_SAMPLE = 2000  # rows sampled for global SHAP importance (kept small = fast)
TOP_N_IMPORTANCE = 25
N_GOLDEN = 5  # real test-set rows captured as a prediction regression fixture


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Path to the home-credit-default project root.",
    )
    args = parser.parse_args()
    source: Path = args.source

    model_pkl = source / "outputs" / "model.pkl"
    processed_csv = source / "data" / "processed_data.csv"
    for p in (model_pkl, processed_csv):
        if not p.exists():
            raise FileNotFoundError(f"Expected source file not found: {p}")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    # 1. Load the pickled model (needs xgboost pinned to the training version).
    print(f"Loading model from {model_pkl} ...")
    import pickle

    with open(model_pkl, "rb") as f:
        model = pickle.load(f)

    feature_columns = list(model.get_booster().feature_names)
    print(f"  model expects {len(feature_columns)} features")

    # 2. Re-save the model in xgboost's portable JSON format (no pickle needed
    #    at serve time, survives library upgrades far better than a pickle).
    model_json = ARTIFACTS / "model.json"
    model.save_model(model_json)
    print(f"  wrote {model_json.name}")

    # 3. Canonical feature order.
    (ARTIFACTS / "feature_columns.json").write_text(
        json.dumps(feature_columns, indent=2)
    )
    print("  wrote feature_columns.json")

    # 4. Per-feature medians = the default "typical applicant" row.
    #    processed_data.csv is already median-imputed, so a column median is a
    #    sensible fill for any field the user does not supply.
    print(f"Loading {processed_csv.name} to compute medians (this reads ~362 MB) ...")
    df = pd.read_csv(processed_csv, usecols=feature_columns, dtype=np.float32)
    # Reindex to guarantee model order, then median per column.
    df = df[feature_columns]
    defaults = {col: float(df[col].median()) for col in feature_columns}
    (ARTIFACTS / "feature_defaults.json").write_text(json.dumps(defaults, indent=2))
    print("  wrote feature_defaults.json")

    # 5. Global feature importance via SHAP on a random sample.
    print(f"Computing SHAP global importance on {SHAP_SAMPLE} sampled rows ...")
    import shap

    sample = df.sample(n=min(SHAP_SAMPLE, len(df)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    mean_abs = np.abs(shap_values).mean(axis=0)
    ranking = sorted(
        ({"feature": c, "mean_abs_shap": float(v)} for c, v in zip(feature_columns, mean_abs)),
        key=lambda d: d["mean_abs_shap"],
        reverse=True,
    )[:TOP_N_IMPORTANCE]
    (ARTIFACTS / "global_importance.json").write_text(json.dumps(ranking, indent=2))
    print("  wrote global_importance.json")

    # 6. Golden regression fixture: replicate the notebook's train/test split and
    #    capture the first few test rows + the pickled model's probabilities.
    #    Tests replay these through the re-saved model.json to prove the export
    #    round-trips predictions exactly -- no 362 MB CSV needed at test time.
    print("Capturing golden prediction fixture ...")
    from sklearn.model_selection import train_test_split

    full = pd.read_csv(processed_csv, dtype=np.float32)
    X = full.drop(columns=["TARGET"])[feature_columns]
    _, X_test = train_test_split(X, test_size=0.20, random_state=42)
    golden_rows = X_test.iloc[:N_GOLDEN]
    golden_probs = model.predict_proba(golden_rows)[:, 1]
    fixture = {
        "feature_columns": feature_columns,
        "cases": [
            {
                "row": [float(v) for v in golden_rows.iloc[i].tolist()],
                "expected_prob": float(golden_probs[i]),
            }
            for i in range(len(golden_rows))
        ],
    }
    FIXTURES.mkdir(parents=True, exist_ok=True)
    (FIXTURES / "golden_predictions.json").write_text(json.dumps(fixture, indent=2))
    print(f"  wrote {N_GOLDEN} golden cases to tests/fixtures/golden_predictions.json")

    print("\nDone. Artifacts written to", ARTIFACTS)


if __name__ == "__main__":
    main()
