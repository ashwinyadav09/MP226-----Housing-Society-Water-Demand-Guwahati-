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

## Layout

| Path | What it holds |
|---|---|
| `params.yaml` | every setting; read by the code |
| `src/mp226/` | the package: config, data loading, data checks, clean, ... |
| `data/raw/` | raw data (`.dvc` pointers for train and later) |
| `tests/` | pytest tests |
| `reports/findings.yaml` | the filled-in form |
| `mlflow.db` | MLflow experiment tracking SQLite database |
| `models/` | champion model artifact (`model.joblib`) |

## Status

Phase 2: Cleaning module, MLflow tracking, pipeline development, and unit testing.
