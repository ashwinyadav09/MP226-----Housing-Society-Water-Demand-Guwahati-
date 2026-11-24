"""Model behavioral and pipeline integrity tests.

All tests run against synthetic data without reading any CSV files from disk.
"""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from mp226.config import ROOT
from mp226.model import build_pipeline


@pytest.fixture
def trained_pipeline():
    """Train a Ridge pipeline (champion model type) on deterministic synthetic data."""
    X = pd.DataFrame([
        {
            "flats_occupied": 100,
            "temp_c": 20.0,
            "garden_area_m2": 500.0,
            "tank_level_morning_pct": 50.0,
            "leak_complaints": 1,
            "supply_hours": 6.0,
            "supply_source": "municipal",
            "tower": "Tower A",
        },
        {
            "flats_occupied": 200,
            "temp_c": 30.0,
            "garden_area_m2": 1000.0,
            "tank_level_morning_pct": 70.0,
            "leak_complaints": 2,
            "supply_hours": 8.0,
            "supply_source": "borewell",
            "tower": "Tower B",
        },
        {
            "flats_occupied": 300,
            "temp_c": 40.0,
            "garden_area_m2": 1500.0,
            "tank_level_morning_pct": 90.0,
            "leak_complaints": 3,
            "supply_hours": 10.0,
            "supply_source": "tanker",
            "tower": "Tower C",
        },
    ])
    # Target strongly correlates with flats and temp
    y = np.array([12000.0, 24000.0, 36000.0])

    pipe = build_pipeline(regressor=Ridge(alpha=10.0), scale_numeric=True)
    pipe.fit(X, y)
    return pipe


def test_pipeline_imputation_and_prediction(trained_pipeline, sample_predict_payload):
    """Pipeline imputes missing values and returns a positive prediction."""
    payload_with_nan = dict(sample_predict_payload)
    payload_with_nan["flats_occupied"] = np.nan
    df = pd.DataFrame([payload_with_nan])

    pred = trained_pipeline.predict(df)[0]
    assert pred > 0
    assert not np.isnan(pred)


def test_more_flats_increases_water_demand(trained_pipeline, sample_predict_payload):
    """Behavioral monotonicity test: more occupied flats should require more water."""
    low_flats = dict(sample_predict_payload, flats_occupied=100)
    high_flats = dict(sample_predict_payload, flats_occupied=280)

    pred_low = trained_pipeline.predict(pd.DataFrame([low_flats]))[0]
    pred_high = trained_pipeline.predict(pd.DataFrame([high_flats]))[0]

    assert pred_high >= pred_low


def test_higher_temp_increases_or_maintains_demand(trained_pipeline, sample_predict_payload):
    """Behavioral test: hotter weather increases water consumption."""
    cool = dict(sample_predict_payload, temp_c=18.0)
    hot = dict(sample_predict_payload, temp_c=42.0)

    pred_cool = trained_pipeline.predict(pd.DataFrame([cool]))[0]
    pred_hot = trained_pipeline.predict(pd.DataFrame([hot]))[0]

    assert pred_hot >= pred_cool


def test_unseen_category_handled_gracefully(trained_pipeline, sample_predict_payload):
    """OneHotEncoder must ignore unseen categorical levels rather than raising an error."""
    unseen_payload = dict(sample_predict_payload, supply_source="river", tower="Tower D")
    pred = trained_pipeline.predict(pd.DataFrame([unseen_payload]))[0]
    assert pred > 0


def test_load_champion_model_and_predict(sample_predict_payload):
    """Load models/model.joblib directly and verify it predicts on request payload."""
    model_path = ROOT / "models" / "model.joblib"
    assert model_path.exists(), "models/model.joblib artifact must exist"

    model = joblib.load(model_path)
    df = pd.DataFrame([sample_predict_payload])
    pred = model.predict(df)[0]

    assert pred > 0, "Predicted water consumption must be positive"
    assert not np.isnan(pred), "Prediction must be a valid number"
