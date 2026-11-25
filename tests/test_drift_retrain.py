"""Unit tests for drift monitoring (PSI) and model promotion logic.

All tests operate on synthetic data in memory with zero CSV reads.
"""

import numpy as np
import pytest

from mp226.drift import psi
from mp226.retrain import should_promote


def test_psi_identical_data_is_zero():
    """Identical distributions must produce a PSI very close to 0."""
    rng = np.random.default_rng(42)
    data = rng.normal(loc=50.0, scale=10.0, size=500)

    psi_q, bins_q = psi(data, data.copy(), bins=10, method="quantile")
    psi_ew, bins_ew = psi(data, data.copy(), bins=10, method="equal_width")

    assert np.isclose(psi_q, 0.0, atol=1e-3), f"Expected PSI ~ 0 for quantile, got {psi_q}"
    assert np.isclose(psi_ew, 0.0, atol=1e-3), f"Expected PSI ~ 0 for equal_width, got {psi_ew}"
    assert bins_q > 0
    assert bins_ew > 0


def test_psi_shifted_data_is_large():
    """A distribution shifted significantly must produce a much higher PSI than identical data."""
    rng = np.random.default_rng(42)
    expected = rng.normal(loc=50.0, scale=10.0, size=500)
    shifted = expected + 25.0  # substantial shift

    psi_val, _ = psi(expected, shifted, bins=10, method="quantile")
    assert psi_val > 1.0, f"Expected large PSI for shifted distribution, got {psi_val}"


def test_should_promote_at_boundary():
    """Exactly at the 15% improvement threshold, promotion should succeed."""
    mae_v1 = 100.0
    mae_v2 = 85.0  # (100 - 85) / 100 = 0.1500 (15%)
    assert should_promote(mae_v1, mae_v2, threshold=0.15) is True


def test_should_promote_just_below_boundary():
    """Just below the 15% threshold (e.g. 14.99%), promotion must be rejected."""
    mae_v1 = 100.0
    mae_v2 = 85.01  # (100 - 85.01) / 100 = 0.1499 (< 15%)
    assert should_promote(mae_v1, mae_v2, threshold=0.15) is False


def test_should_promote_just_above_boundary():
    """Just above the 15% threshold (e.g. 15.01%), promotion must succeed."""
    mae_v1 = 100.0
    mae_v2 = 84.99  # (100 - 84.99) / 100 = 0.1501 (> 15%)
    assert should_promote(mae_v1, mae_v2, threshold=0.15) is True


def test_should_promote_worse_v2():
    """When v2 performs worse or equal to v1, promotion must be rejected."""
    assert should_promote(100.0, 105.0, threshold=0.15) is False
    assert should_promote(100.0, 100.0, threshold=0.15) is False


def test_weighted_fit_moves_predictions_toward_later_regime():
    """Synthetic test verifying sample weights shift regressor predictions toward the later regime.

    Constructs two regimes in memory without reading CSV files:
      - Regime 1 ('train'): 100 observations with target centered around 1000.
      - Regime 2 ('later'): 20 observations with target centered around 5000.
    With uniform weights (1.0 each), the fit is dominated by the 100 observations of Regime 1.
    With sample weighting (e.g. weight 40 on Regime 2), the prediction on a test point shifts
    strongly toward the Regime 2 target value (5000).
    """
    import pandas as pd
    from sklearn.linear_model import Ridge
    from mp226.model import build_pipeline

    n_train = 100
    n_later = 20

    # Synthetic features matching schema
    cols = {
        "flats_occupied": np.full(n_train + n_later, 100.0),
        "temp_c": np.full(n_train + n_later, 28.0),
        "garden_area_m2": np.full(n_train + n_later, 500.0),
        "tank_level_morning_pct": np.full(n_train + n_later, 75.0),
        "leak_complaints": np.full(n_train + n_later, 2.0),
        "supply_hours": np.full(n_train + n_later, 8.0),
        "supply_source": ["Borewell"] * (n_train + n_later),
        "tower": ["Tower A"] * (n_train + n_later),
    }
    df = pd.DataFrame(cols)
    y = np.concatenate([np.full(n_train, 1000.0), np.full(n_later, 5000.0)])

    # Test point with identical feature values
    test_df = pd.DataFrame({k: [v[0]] for k, v in cols.items()})

    # Unweighted fit
    pipe_unweighted = build_pipeline(Ridge(alpha=1.0), scale_numeric=True)
    pipe_unweighted.fit(df, y, regressor__sample_weight=np.ones(len(df)))
    pred_unweighted = pipe_unweighted.predict(test_df)[0]

    # Weighted fit (weight 40 for later regime)
    weights = np.concatenate([np.ones(n_train), np.full(n_later, 40.0)])
    pipe_weighted = build_pipeline(Ridge(alpha=1.0), scale_numeric=True)
    pipe_weighted.fit(df, y, regressor__sample_weight=weights)
    pred_weighted = pipe_weighted.predict(test_df)[0]

    # The weighted prediction must shift substantially towards Regime 2's target (5000)
    assert pred_weighted > pred_unweighted, (
        f"Expected weighted prediction ({pred_weighted}) > unweighted ({pred_unweighted})"
    )
    assert abs(pred_weighted - 5000.0) < abs(pred_unweighted - 5000.0), (
        "Weighted prediction must be closer to the later regime target (5000) than unweighted"
    )
