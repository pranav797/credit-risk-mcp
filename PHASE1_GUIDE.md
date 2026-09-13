# Phase 1 Guide — Extracting the Model from the Notebooks

## Why this phase exists (the significance)

Your model lives inside Jupyter notebooks. Notebooks are great for *exploring*
but useless as a *service*: you can't call a notebook from Claude. Phase 1's job
is to lift the model out of the notebooks into plain Python functions that any
program (including, later, the MCP server) can import and call.

Sounds trivial — "just load the pickle and call `.predict()`" — but there's a
real gap, and closing it *is* the whole point of this phase:

> **The model was trained on a 235-column table. A human describing a borrower
> can give you maybe 10 facts. Something has to turn "34-year-old, $54k income,
> working, weak credit scores" into a valid 235-number row in the exact shape
> the model expects.**

That translation is `build_feature_vector()`. Getting it *exactly* right — same
column order, same encodings, same engineered features as training — is the
difference between real predictions and silent garbage. This is the single most
common thing that breaks when people move an ML model from a notebook to
production, and being able to talk about it is a strong interview signal.

## The mental model

```
   BorrowerProfile              build_feature_vector()            model.predict_proba()
  (10 friendly fields)   ─────▶  235-column row in the    ─────▶   default probability
                                 model's exact order
        schemas.py               model_service.py +                 model_service.py
                                 feature_spec.py
```

Four files do this, and you'll build them in this order:

| Order | File | What it holds | Difficulty |
|-------|------|---------------|------------|
| 1 | `scripts/export_artifacts.py` | one-off: turns the notebook's `model.pkl` + CSV into small portable files | mechanical |
| 2 | `src/credit_risk_mcp/feature_spec.py` | the *domain knowledge*: how each field is encoded | conceptual ⭐ |
| 3 | `src/credit_risk_mcp/schemas.py` | the friendly input shape (pydantic) | mostly declarative |
| 4 | `src/credit_risk_mcp/model_service.py` | load model, `build_feature_vector`, `score`, `explain` | the core ⭐⭐ |

## How to work

- Each scaffold has `TODO`s and docstrings. Fill them in.
- **The tests are your spec and your grade.** Run them constantly:
  ```bash
  uv run --extra dev pytest -q
  ```
  Everything is red now. Make it green one test at a time. Read the test files —
  they literally tell you the expected behaviour (e.g. `test_binary_encoding`
  shows exactly what `CODE_GENDER="M"` should produce).
- Stuck for more than ~15 min on one thing? Peek at the matching file in
  `reference/`. Understanding it then re-typing it still teaches you.

---

## The facts you need from the notebooks

I've pulled these out of `02_feature_engineering.ipynb` and `03_modelling.ipynb`
so you don't have to reverse-engineer them. This *is* the domain knowledge that
goes into `feature_spec.py`. (Open the notebooks yourself too — seeing it in
context is worth it.)

### Binary label encodings (from notebook 02, the `.map()` calls)
```
CODE_GENDER:         M → 1, F → 0     (the 4 "XNA" rows were changed to F first)
FLAG_OWN_CAR:        Y → 1, N → 0
FLAG_OWN_REALTY:     Y → 1, N → 0
NAME_CONTRACT_TYPE:  "Cash loans" → 1, "Revolving loans" → 0
```

### One-hot groups (from `pd.get_dummies(..., drop_first=True)`)
`drop_first=True` **drops the alphabetically-first category** as a "reference"
level. That dropped category has **no column** — it's represented by *all dummies
in the group being 0*. The categories that KEPT a column:

```
NAME_INCOME_TYPE   (dropped: Businessman)    → Commercial associate, Maternity leave,
                                                Pensioner, State servant, Student,
                                                Unemployed, Working
NAME_EDUCATION_TYPE (dropped: Academic degree) → Higher education, Incomplete higher,
                                                  Lower secondary, Secondary / secondary special
NAME_FAMILY_STATUS (dropped: Civil marriage) → Married, Separated, Single / not married,
                                                Unknown, Widow
NAME_HOUSING_TYPE  (dropped: Co-op apartment)→ House / apartment, Municipal apartment,
                                                Office apartment, Rented apartment, With parents
```
So `income_type="Working"` → set `NAME_INCOME_TYPE_Working = 1`, all other
`NAME_INCOME_TYPE_*` = 0. And `income_type="Businessman"` → *every*
`NAME_INCOME_TYPE_*` = 0 (it's the reference level).

### Engineered features (derived from raw columns — notebook 02)
```
AGE_YEARS            = abs(DAYS_BIRTH) / 365.25
YEARS_EMPLOYED       = abs(DAYS_EMPLOYED) / 365.25
YEARS_ID_PUBLISH     = abs(DAYS_ID_PUBLISH) / 365.25
YEARS_REGISTRATION   = abs(DAYS_REGISTRATION) / 365.25
CREDIT_INCOME_RATIO  = AMT_CREDIT / AMT_INCOME_TOTAL
ANNUITY_INCOME_RATIO = AMT_ANNUITY / AMT_INCOME_TOTAL
CREDIT_TERM          = AMT_CREDIT / AMT_ANNUITY
```
Note the model has BOTH `DAYS_BIRTH` and `AGE_YEARS` as features. So when a user
gives you `age`, you set `DAYS_BIRTH = -age * 365.25` (raw column) AND recompute
`AGE_YEARS` from it — keep them consistent, or the model sees a contradiction.

**Key subtlety:** only recompute a derived feature when one of its *inputs was
supplied by the user*. If you recompute everything from the default row, you'd
replace the true training medians with "ratio-of-medians", which is different and
wrong. Leave untouched defaults exactly as they are.

### Modelling facts (notebook 03)
- No feature scaling — XGBoost is fed the raw frame directly.
- `X = df.drop(columns=["TARGET"])` → 235 features. `SK_ID_CURR` (the applicant
  id) was left in as a feature; just treat it like any other default.
- Default probability = `model.predict_proba(X)[:, 1]` (column 1 = "will default").
- **Business threshold = 0.15**, not 0.5. In credit risk a missed defaulter (lost
  principal) costs far more than a false alarm, so you tune for *recall*.
- Performance: AUC-ROC ≈ 0.7476; at 0.15 threshold precision ≈ 0.21, recall ≈ 0.47.

### Where defaults and column order come from
`processed_data.csv` is the training frame *after* imputation, so:
- the **median of each column** is a sensible "typical applicant" fallback →
  `feature_defaults.json`
- the model's own `model.get_booster().feature_names` is the **canonical column
  order** → `feature_columns.json`. Always build your row in this order.

---

## File-by-file checklist

**1. `scripts/export_artifacts.py`** — run once, produces `artifacts/`:
- [ ] load `model.pkl`, read `feature_names` → write `feature_columns.json`
- [ ] `model.save_model("model.json")` (portable format, no pickle at serve time)
- [ ] load the CSV, compute per-column medians → `feature_defaults.json`
- [ ] SHAP on a small sample → `global_importance.json`
- [ ] capture a few real test rows + their probs → `tests/fixtures/golden_predictions.json`
- (the artifacts already exist from before, so you can do this file last)

**2. `feature_spec.py`** — translate the facts above into data structures:
- [ ] `BINARY_ENCODINGS`, `ONEHOT_GROUPS`, `DERIVED_FEATURES`, `FRIENDLY_NAMES`
- [ ] `friendly_name(column)` helper

**3. `schemas.py`** — the `BorrowerProfile` pydantic model:
- [ ] ~13 optional fields + 2 required (`annual_income`, `credit_amount`)
- [ ] use `Literal[...]` for the categorical fields, `Field(...)` for validation ranges

**4. `model_service.py`** — the engine:
- [ ] load artifacts once (singleton), assert columns match the model
- [ ] `build_feature_vector(profile)` — the heart of Phase 1
- [ ] `score`, `explain`, `model_info`, `global_importance`

When `uv run --extra dev pytest -q` is all green, Phase 1 is genuinely yours.
```bash
uv run --extra dev pytest -q
```
