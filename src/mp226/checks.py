"""The seven data checks for MP226.

Each function analyzes the dataset and detects one specific kind of data defect
described in the Student Handbook (section 7).
Returns a dictionary matching the schema in reports/findings.yaml.

Run all checks with:
    python -m mp226.checks
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from .data import load_raw


def check_leaking_column(
    train: pd.DataFrame, holdout: pd.DataFrame, target: str
) -> Dict[str, Any]:
    """Identify columns known only AFTER the event.

    A leaking column exists in training data but is missing from holdout_features,
    and exhibits near-perfect correlation with the target variable.
    """
    candidates = [
        col for col in train.columns
        if col not in holdout.columns and col != target and col != "record_id"
    ]
    for col in candidates:
        if pd.api.types.is_numeric_dtype(train[col]):
            corr = abs(train[col].corr(train[target]))
            if corr > 0.95:
                return {"present": True, "column": col}
    return {"present": False, "column": None}


def check_unit_mixup(
    train: pd.DataFrame,
    later: pd.DataFrame,
    numeric_cols: List[str],
    group_col: str,
) -> Dict[str, Any]:
    """Detect if one group value recorded a feature in a different unit.

    Compares the mean and median per group against other groups in train
    and against clean later data. Common unit factors include ~10.764 (sq ft to sq m),
    1000 (m to km), 12 (years to months), or 60 (minutes to hours).
    """
    groups = train[group_col].dropna().unique()
    for col in numeric_cols:
        means = train.groupby(group_col)[col].mean()
        overall_mean = means.median()
        for g in groups:
            ratio = means[g] / overall_mean if overall_mean > 0 else 1.0
            # Check for significant scaling factor (~10x or common unit conversion)
            if ratio > 5.0 or ratio < 0.2:
                # Find exact factor by comparing group values against cleanly scaled integers
                vals = train[train[group_col] == g][col]
                # Test standard unit conversion factors:
                # 10.764 is the exact conversion from sq ft to sq m
                for candidate_factor in [10.764, 10.0, 1000.0, 100.0, 12.0, 60.0]:
                    unscaled = vals / candidate_factor
                    diff = np.abs(unscaled - np.round(unscaled))
                    if (diff < 0.01).mean() > 0.95:
                        return {
                            "present": True,
                            "column": col,
                            "group_value": g,
                            "factor": candidate_factor,
                        }
                return {
                    "present": True,
                    "column": col,
                    "group_value": g,
                    "factor": round(float(ratio), 3),
                }
    return {"present": False, "column": None, "group_value": None, "factor": None}


def check_sentinel_value(
    train: pd.DataFrame, numeric_cols: List[str]
) -> Dict[str, Any]:
    """Detect sentinel codes (e.g., 999, -999, -1) used instead of NaN.

    Looks for impossible numbers repeated frequently outside normal distribution range.
    """
    common_sentinels = [999, 9999, -999, -1, 999.0, 9999.0, -999.0, -1.0]
    for col in numeric_cols:
        vc = train[col].value_counts()
        for val, count in vc.items():
            if val in common_sentinels and count >= 5:
                return {
                    "present": True,
                    "column": col,
                    "value": int(val) if float(val).is_integer() else val,
                    "count": int(count),
                }
            # For percentages (0-100), any repeated value > 100 is an obvious sentinel
            if "pct" in col and val > 100 and count >= 5:
                return {
                    "present": True,
                    "column": col,
                    "value": int(val) if float(val).is_integer() else val,
                    "count": int(count),
                }
    return {"present": False, "column": None, "value": None, "count": None}


def check_duplicate_rows(train: pd.DataFrame) -> Dict[str, Any]:
    """Count duplicate rows, checking both full rows and without record_id."""
    cols_no_id = [c for c in train.columns if c != "record_id"]
    dup_count = int(train.duplicated(subset=cols_no_id).sum())
    return {
        "present": bool(dup_count > 0),
        "count": dup_count if dup_count > 0 else None,
    }


def check_stuck_sensor(
    train: pd.DataFrame, numeric_cols: List[str]
) -> Dict[str, Any]:
    """Detect a stuck sensor showing identical readings over consecutive dates."""
    df = train.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    for col in numeric_cols:
        # Check for multiple dates where every reading on that date is identical
        daily_nunique = df.groupby("date")[col].nunique()
        daily_val = df.groupby("date")[col].first()
        dates_with_single_val = daily_nunique[daily_nunique == 1].index

        # Find longest streak of dates with the exact same constant value
        streak_val = None
        streak_start = None
        streak_end = None
        max_streak_len = 0
        best_start = None
        best_end = None

        curr_val = None
        curr_start = None
        curr_end = None
        curr_len = 0

        for d in dates_with_single_val:
            val = daily_val[d]
            if curr_val is None or val != curr_val or (d - curr_end).days > 1:
                curr_val = val
                curr_start = d
                curr_end = d
                curr_len = 1
            else:
                curr_end = d
                curr_len += 1

            if curr_len > max_streak_len:
                max_streak_len = curr_len
                best_start = curr_start
                best_end = curr_end
                streak_val = curr_val

        # A stuck sensor persisting across more than a week (>= 7 days)
        if max_streak_len >= 7:
            return {
                "present": True,
                "column": col,
                "start_date": best_start.strftime("%Y-%m-%d"),
                "end_date": best_end.strftime("%Y-%m-%d"),
            }

    return {"present": False, "column": None, "start_date": None, "end_date": None}


def check_target_typos(train: pd.DataFrame, target: str) -> Dict[str, Any]:
    """Detect typo entries in target (e.g. 10x too big with an extra zero).

    Cross-checked using the target distribution gap and ratio to meter_litres_next_morning.
    """
    # 1. Ratio check against meter_litres_next_morning if available
    if "meter_litres_next_morning" in train.columns:
        ratio = train[target] / train["meter_litres_next_morning"].replace(0, np.nan)
        typos = (ratio > 5.0) | (ratio < 0.2)
        count = int(typos.sum())
        if count > 0:
            return {"present": True, "count": count}

    # 2. Distribution gap check: sort target and look for orders-of-magnitude jump
    sorted_targets = train[target].dropna().sort_values()
    if len(sorted_targets) > 10:
        ratio_to_prev = sorted_targets.iloc[1:] / sorted_targets.iloc[:-1].values
        huge_jumps = ratio_to_prev > 5.0
        if huge_jumps.any():
            jump_idx = np.where(huge_jumps)[0][0]
            count = int(len(sorted_targets) - (jump_idx + 1))
            return {"present": True, "count": count}

    return {"present": False, "count": None}


def check_spelling_variants(
    train: pd.DataFrame, later: pd.DataFrame, text_cols: List[str]
) -> Dict[str, Any]:
    """Detect inconsistent category spellings (whitespace, casing, typos)."""
    for col in text_cols:
        train_vals = set(train[col].dropna().astype(str).unique())
        later_vals = set(later[col].dropna().astype(str).unique())

        # Clean normalized categories
        norm_train = {v.strip().lower() for v in train_vals}
        if len(norm_train) < len(train_vals):
            return {
                "present": True,
                "column": col,
                "true_levels": len(later_vals),
            }

        # Any unseen corrupted category variant in train not present in clean later
        diff = train_vals - later_vals
        if diff:
            return {
                "present": True,
                "column": col,
                "true_levels": len(later_vals),
            }

    return {"present": False, "column": None, "true_levels": None}


def main():
    from .config import load_params

    cols = load_params()["columns"]
    train, later, holdout = (load_raw(n) for n in ("train", "later", "holdout"))

    jobs = [
        ("spelling_variants", lambda: check_spelling_variants(train, later, cols["text"])),
        ("leaking_column", lambda: check_leaking_column(train, holdout, cols["target"])),
        ("unit_mixup", lambda: check_unit_mixup(train, later, cols["numeric"], cols["group"])),
        ("sentinel_value", lambda: check_sentinel_value(train, cols["numeric"])),
        ("duplicate_rows", lambda: check_duplicate_rows(train)),
        ("stuck_sensor", lambda: check_stuck_sensor(train, cols["numeric"])),
        ("target_typos", lambda: check_target_typos(train, cols["target"])),
    ]
    for name, job in jobs:
        try:
            print(f"{name:18s}", job())
        except NotImplementedError:
            print(f"{name:18s} TODO")


if __name__ == "__main__":
    main()
