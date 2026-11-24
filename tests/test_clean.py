"""Tests for the data cleaning module."""

import numpy as np
import pandas as pd
from mp226.clean import (
    clean_train_data,
    fix_sentinels,
    fix_stuck_sensor,
    fix_unit_mixup,
)


def test_fix_sentinels(sample_raw_df):
    cleaned = fix_sentinels(sample_raw_df, col="tank_level_morning_pct", sentinel=999)
    assert pd.isna(cleaned.loc[0, "tank_level_morning_pct"])
    assert cleaned.loc[1, "tank_level_morning_pct"] == 65


def test_fix_unit_mixup(sample_raw_df):
    cleaned = fix_unit_mixup(sample_raw_df, col="garden_area_m2", factor=10.764)
    # 10764 / 10.764 = 1000.0
    assert np.isclose(cleaned.loc[0, "garden_area_m2"], 1000.0)
    # Tower B value untouched
    assert cleaned.loc[1, "garden_area_m2"] == 850.0


def test_fix_stuck_sensor_window(sample_raw_df):
    cleaned = fix_stuck_sensor(sample_raw_df, col="flats_occupied", stuck_value=184)
    # Row 0 is 2026-04-25 (inside window) -> NaN
    assert pd.isna(cleaned.loc[0, "flats_occupied"])
    # Row 1 is 2026-01-15 (outside window) -> genuine 184 preserved
    assert cleaned.loc[1, "flats_occupied"] == 184


def test_clean_train_data_pipeline(sample_raw_df):
    cleaned = clean_train_data(sample_raw_df)
    assert pd.isna(cleaned.loc[0, "flats_occupied"])
    assert pd.isna(cleaned.loc[0, "tank_level_morning_pct"])
    assert np.isclose(cleaned.loc[0, "garden_area_m2"], 1000.0)
    assert cleaned.loc[1, "flats_occupied"] == 184
