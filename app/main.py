"""FastAPI prediction service for MP226 water demand forecasting.

Endpoints:
- GET  /health   -> {"status": "ok"}
- POST /predict  -> Predicts litres_per_day given housing society sensor inputs.

Enforces client business rules via Pydantic Field constraints (HTTP 422 on violation).
Logs each prediction event as structured JSON to stdout only.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import time
from typing import Dict

from fastapi import FastAPI, HTTPException, status
import joblib
import pandas as pd
from pydantic import BaseModel, Field

# Locate repository root and models directory
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.joblib"

# Configure structured JSON logger directed to stdout only (no log files)
logger = logging.getLogger("mp226.service")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

# Load model artifact
model = None


def get_model():
    global model
    if model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(f"Model artifact not found at {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifact at service startup."""
    try:
        get_model()
        logger.info(
            json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": "INFO",
                "event": "startup",
                "message": f"Loaded champion model from {MODEL_PATH}",
            })
        )
    except Exception as e:
        logger.error(
            json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": "ERROR",
                "event": "startup_failed",
                "error": str(e),
            })
        )
    yield


app = FastAPI(
    title="MP226 Housing Society Water Demand Service",
    description="FastAPI service serving daily water demand predictions in Guwahati.",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    """Input payload with client business rules enforced via Field constraints.

    Rule 1: supply_hours > 20 is impossible (rejected with HTTP 422).
    Rule 2: garden_area_m2 < 0 is a typing mistake (rejected with HTTP 422).
    """

    flats_occupied: float
    temp_c: float
    garden_area_m2: float = Field(
        ...,
        ge=0.0,
        description="Garden area watered that day in square metres (cannot be negative)",
    )
    tank_level_morning_pct: float
    leak_complaints: float
    supply_hours: float = Field(
        ...,
        le=20.0,
        description="Hours of municipal water supply (cannot exceed 20 hours)",
    )
    supply_source: str
    tower: str


class PredictResponse(BaseModel):
    prediction: float


@app.get("/health", status_code=status.HTTP_200_OK)
def health() -> Dict[str, str]:
    """Healthcheck endpoint returning standard status."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse, status_code=status.HTTP_200_OK)
def predict(request: PredictRequest) -> PredictResponse:
    """Predict water demand (litres_per_day) and log structured JSON to stdout."""
    start_time = time.time()
    pipe = get_model()

    input_data = {
        "flats_occupied": request.flats_occupied,
        "temp_c": request.temp_c,
        "garden_area_m2": request.garden_area_m2,
        "tank_level_morning_pct": request.tank_level_morning_pct,
        "leak_complaints": request.leak_complaints,
        "supply_hours": request.supply_hours,
        "supply_source": request.supply_source,
        "tower": request.tower,
    }

    df = pd.DataFrame([input_data])
    pred = float(pipe.predict(df)[0])
    duration_ms = (time.time() - start_time) * 1000.0

    # Structured JSON log line emitted to stdout only (no PII)
    log_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": "INFO",
        "event": "prediction",
        "inputs": input_data,
        "prediction": round(pred, 2),
        "duration_ms": round(duration_ms, 2),
    }
    logger.info(json.dumps(log_record))

    return PredictResponse(prediction=pred)
