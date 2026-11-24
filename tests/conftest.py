"""Pytest fixtures for unit and behavioral tests.

IMPORTANT: No CSV files are read from disk (per handbook rule),
guaranteeing tests run fast and reliably in CI environments without raw data.
"""

import pandas as pd
import pytest


@pytest.fixture
def sample_raw_df():
    """Synthetic dataframe simulating raw inputs with data defects."""
    return pd.DataFrame([
        {
            "record_id": "rec_001",
            "date": "2026-04-25",  # inside stuck window
            "flats_occupied": 184,   # stuck sensor
            "temp_c": 32.5,
            "garden_area_m2": 10764.0,  # Tower A sq ft
            "tank_level_morning_pct": 999,  # sentinel
            "leak_complaints": 2,
            "supply_hours": 6.5,
            "supply_source": "borewell",
            "tower": "Tower A",
            "meter_litres_next_morning": 21000.0,
            "litres_per_day": 21050.0,
        },
        {
            "record_id": "rec_002",
            "date": "2026-01-15",  # outside stuck window
            "flats_occupied": 184,   # genuine 184
            "temp_c": 22.0,
            "garden_area_m2": 850.0,  # Tower B m^2
            "tank_level_morning_pct": 65,
            "leak_complaints": 0,
            "supply_hours": 8.0,
            "supply_source": "municipal",
            "tower": "Tower B",
            "meter_litres_next_morning": 16500.0,
            "litres_per_day": 16480.0,
        },
        {
            "record_id": "rec_003",
            "date": "2026-06-01",
            "flats_occupied": 210,
            "temp_c": 38.0,
            "garden_area_m2": 920.0,
            "tank_level_morning_pct": 40,
            "leak_complaints": 5,
            "supply_hours": 5.0,
            "supply_source": "tanker",
            "tower": "Tower C",
            "meter_litres_next_morning": 28000.0,
            "litres_per_day": 28100.0,
        },
    ])


@pytest.fixture
def sample_predict_payload():
    """Valid prediction request input matching the /predict schema."""
    return {
        "flats_occupied": 180,
        "temp_c": 30.0,
        "garden_area_m2": 1200.0,
        "tank_level_morning_pct": 75.0,
        "leak_complaints": 3,
        "supply_hours": 6.0,
        "supply_source": "borewell",
        "tower": "Tower A",
    }
