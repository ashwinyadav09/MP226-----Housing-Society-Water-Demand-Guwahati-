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
