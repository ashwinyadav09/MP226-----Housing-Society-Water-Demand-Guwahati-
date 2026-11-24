"""Data cleaning module for MP226.

Implements all data-cleaning decisions identified in Phase 1:
- Sentinels in tank_level_morning_pct (999) replaced with NaN
- Stuck sensor in flats_occupied during 2026-04-23 to 2026-05-11 replaced with NaN
- Unit mixup in garden_area_m2 for Tower A divided by 10.764 (sq ft -> m^2)
- Deduplication and typo verification rules (handled safely if present)

Imputation of NaNs is intentionally deferred to the scikit-learn Pipeline so
that the same imputation statistics are consistently applied during both training
and inference serving.
"""

from typing import Optional
import numpy as np
import pandas as pd

from .config import load_params


def fix_sentinels(
    df: pd.DataFrame, col: str, sentinel: float
) -> pd.DataFrame:
    """Replace placeholder sentinel values with NaN."""
    out = df.copy()
    if col in out.columns:
        out.loc[out[col] == sentinel, col] = np.nan
    return out


def fix_stuck_sensor(
    df: pd.DataFrame,
    col: str,
    start_date: str,
    end_date: str,
    stuck_value: float,
) -> pd.DataFrame:
    """Blank stuck sensor readings inside the specific failure window only.

    Readings equal to stuck_value outside this date window are genuine and preserved.
    """
    out = df.copy()
    if col in out.columns and "date" in out.columns:
        mask = (
            (out["date"] >= start_date)
            & (out["date"] <= end_date)
            & (out[col] == stuck_value)
        )
        out.loc[mask, col] = np.nan
    return out


def fix_unit_mixup(
    df: pd.DataFrame,
    col: str,
    group_col: str,
    group_val: str,
    factor: float,
) -> pd.DataFrame:
    """Convert Tower A garden area from square feet to square metres."""
    out = df.copy()
    if col in out.columns and group_col in out.columns:
        mask = out[group_col] == group_val
        out.loc[mask, col] = out.loc[mask, col] / factor
    return out


def drop_duplicates_custom(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate records ignoring record_id."""
    cols_no_id = [c for c in df.columns if c != "record_id"]
    return df.drop_duplicates(subset=cols_no_id).copy()


def fix_target_typos(df: pd.DataFrame, target: str = "litres_per_day") -> pd.DataFrame:
    """Adjust target typos (divided by 10) only if confirmed, otherwise preserve."""
    out = df.copy()
    if target in out.columns and "meter_litres_next_morning" in out.columns:
        ratio = out[target] / out["meter_litres_next_morning"].replace(0, np.nan)
        # 10x typo condition: ratio around 10
        typo_mask = ratio > 5.0
        if typo_mask.any():
            out.loc[typo_mask, target] = out.loc[typo_mask, target] / 10.0
    return out


def fix_spellings(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize text column casing and strip extra whitespace."""
    out = df.copy()
    for col in ["supply_source", "tower"]:
        if col in out.columns and pd.api.types.is_string_dtype(out[col]):
            out[col] = out[col].str.strip()
    return out


def clean_train_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full sequence of data cleaning operations to training data.

    Constants are loaded directly from params.yaml.
    """
    params = load_params()
    clean_cfg = params.get("clean", {})

    cleaned = df.copy()
    cleaned = fix_spellings(cleaned)
    cleaned = drop_duplicates_custom(cleaned)
    cleaned = fix_target_typos(cleaned)

    if "unit_mixup" in clean_cfg:
        u = clean_cfg["unit_mixup"]
        cleaned = fix_unit_mixup(
            cleaned,
            col=u["column"],
            group_col=u["group_col"],
            group_val=u["group_value"],
            factor=float(u["factor"]),
        )

    if "sentinel" in clean_cfg:
        s = clean_cfg["sentinel"]
        cleaned = fix_sentinels(
            cleaned,
            col=s["column"],
            sentinel=float(s["value"]),
        )

    if "stuck_sensor" in clean_cfg:
        st = clean_cfg["stuck_sensor"]
        cleaned = fix_stuck_sensor(
            cleaned,
            col=st["column"],
            start_date=st["start_date"],
            end_date=st["end_date"],
            stuck_value=float(st["stuck_value"]),
        )

    return cleaned
