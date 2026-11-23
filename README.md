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

## Layout

| Path | What it holds |
|---|---|
| `params.yaml` | every setting; read by the code |
| `src/mp226/` | the package: config, data loading, data checks, ... |
| `data/raw/` | raw data (`.dvc` pointers for train and later) |
| `tests/` | pytest tests |
| `reports/findings.yaml` | the filled-in form |

## Status

Work in progress: Stage 1 (data and DVC).
