# MP226 - Housing Society Water Demand (Guwahati)

SCSE3040 MLOps mini project.

| | |
|---|---|
| Group id | G226 |
| Problem id | MP226 |
| Enrolment numbers | S24CSEU1353, S24CSEU1365, S24CSEU1342, S24CSEU1370 |
| Built by | S24CSEU1342 (Ashwin Yadav) - completed individually with the approval of the course faculty |

## What this does

Predicts `litres_per_day` (water used by the society in a day) so the
welfare association can order tankers a day early. The pipeline covers
data versioning (DVC), data checks and cleaning, MLflow tracking and
registry, a FastAPI service, Docker, GitHub Actions, Kubernetes files,
JSON logging, drift monitoring and a retrain.

## Set up

```
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Data Cleaning Decisions

> **Important**: The cleaning function (`clean_train_data` in `src/mp226/clean.py`) is for `train.csv` only. Incoming datasets (`later.csv` and `holdout_features.csv`) arrive clean and are used as given.

All cleaning transformations are implemented in `src/mp226/clean.py` based on empirical analysis of `data/raw/train.csv`:

1. **Spelling Variants**:
   - Both categorical features (`supply_source` and `tower`) match `later.csv` exactly with standard categories (`borewell`, `municipal`, `tanker` and `Tower A`, `Tower B`, `Tower C`). Whitespace stripping is applied defensively.
2. **Sentinel Values**:
   - `tank_level_morning_pct` had `999` recorded 61 times as an impossible placeholder (valid range is 0–100%). These 61 occurrences are set to `NaN`.
3. **Unit Mixup Fix**:
   - `garden_area_m2` for `Tower A` (366 rows) was entered in square feet rather than square metres ($1\text{ m}^2 = 10.764\text{ sq ft}$, factor: `10.764`). Dividing Tower A values by `10.764` brings the distribution in line with Towers B & C and clean `later.csv`.
4. **Duplicate Rows**:
   - No duplicate records existed (0 duplicates found across all feature columns). The deduplication rule drops exact feature duplicates if ever present.
5. **Stuck Sensor Run**:
   - `flats_occupied` was stuck at `184` from `2026-04-23` through `2026-05-11` (91 rows). These 91 readings are converted to `NaN`. Three genuine occurrences of `184` outside this calendar window are preserved.
6. **Target Typos**:
   - `litres_per_day` has no 10x typing errors or orders-of-magnitude jumps (ratio to `meter_litres_next_morning` remains between 0.78 and 1.18). Confirmed 10x typos would be divided by 10; unconfirmed ones remain unaltered.
7. **Imputation Strategy**:
   - Missing values created during cleaning (`NaN`s in `flats_occupied` and `tank_level_morning_pct`) are imputed inside the scikit-learn training/serving Pipeline using median statistics to ensure zero data leakage between splits and exact parity between training and online inference.

## Monitoring, Drift & Regime-Shift Weighting

Drift monitoring between cleaned `train.csv` and production `later.csv` is implemented in `src/mp226/drift.py` and retraining in `src/mp226/retrain.py`.
- **Binning Strategy**: Configured via `drift.binning` in `params.yaml` (`quantile` or `equal_width`, default: `quantile`).
- **Comprehensive Comparison**: Both binning algorithms were evaluated side by side and saved to `reports/psi_both_methods.csv`.
- **Drift Findings**: Under both methods, `supply_hours` experienced severe drift ($\text{PSI} \approx 3.31$), while all other numeric features remained stable ($\text{PSI} < 0.07$).
- **Data Evidence for Regime Shift**:
  - `supply_hours` mean moved from $6.48$ (~$6.5$) hours in `train.csv` to $10.82$ (~$10.8$) hours in `later.csv`.
  - Society-wide per-flat water consumption (`litres_per_day / flats_occupied`) rose from $216.5$ to $327.9$ litres ($+51.5\%$).
  - The surge in consumption was most pronounced in **Tower A**, jumping $+80.7\%$ (from $235.1$ to $425.0$ litres per flat), compared to $+35.5\%$ in Tower B and $+38.0\%$ in Tower C.
- **Subgroup Disparity in v1**: v1 was trained on the cleaned training data (Tower A's unit mix-up already fixed), so units are not the cause. On `later.csv` it under-predicts in every tower: by about 14,072 litres on average in Tower A against about 1,800 in Tower B and 1,600 in Tower C. Supply hours rose sharply (mean 6.5 to 10.8, beyond the 12-hour maximum seen in training) and per-flat consumption rose most in Tower A (+80.7%).
- **Inner-Split Weight Optimization**:
  - To adapt to the new higher-consumption operational regime without overfitting or data leakage, `retrain.py` tunes sample weighting using an **inner time-ordered split** on the first 70% of `later.csv` (fit on clean train + first 2/3 of later rows, validate on the remaining 1/3).
  - Across the grid `[1, 5, 10, 20, 40]`, sample weight $40$ achieved the lowest inner validation MAE ($1,946.65$ litres, down from $4,396.08$ litres for weight 1).
- **Retraining & Promotion**:
  - Refitting on clean train (weight 1) plus all first-70% rows (weight 40) and evaluating on the independent last 30% check set dropped MAE from $7,679.11$ (v1) down to $2,416.10$ litres ($68.54\%$ relative improvement).
  - This substantially surpassed the $15\%$ threshold, earning model v2 promotion to `@champion`.

## Layout

| Path | What it holds |
|---|---|
| `params.yaml` | every setting; read by the code |
| `src/mp226/` | the package: config, data loading, data checks, clean, drift, retrain |
| `app/` | FastAPI service (`POST /predict`, `GET /health`) |
| `data/raw/` | raw data (`.dvc` pointers for train and later) |
| `tests/` | pytest unit, behavioral, API, and drift tests |
| `reports/findings.yaml` | the completed evaluation form |
| `reports/psi_both_methods.csv` | PSI comparison across quantile and equal-width methods |
| `predictions/holdout_predictions.csv` | champion model predictions on holdout_features.csv |
| `mlflow.db` | MLflow experiment tracking SQLite database (versions 1 & 2) |
| `models/model.joblib` | serialized champion model artifact |
| `Dockerfile` | container build configuration (`python:3.12-slim`) |
| `k8s/` | Kubernetes deployment (`mp226-api`) and service (`mp226-svc`) manifests |
| `.github/workflows/` | GitHub Actions CI workflow |

## Status

Complete (Phase 1–4 finished): Data checks, DVC tracking, cleaning module, MLflow tracking with v1 & v2, FastAPI service with JSON logging, Docker container, Kubernetes files, green GitHub Actions CI, PSI drift monitoring, and holdout predictions generated.
