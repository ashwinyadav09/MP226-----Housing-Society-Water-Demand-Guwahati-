"""Data drift monitoring and subgroup performance evaluation for MP226.

Computes Population Stability Index (PSI) between cleaned train.csv and later.csv
using both quantile and equal-width binning methods.
Evaluates v1 champion model performance per tower on later.csv.

Run with:
    python -m mp226.drift
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import yaml

from .clean import clean_train_data
from .config import ROOT, load_params
from .data import load_raw


def psi(
    expected: Union[pd.Series, np.ndarray],
    actual: Union[pd.Series, np.ndarray],
    bins: int = 10,
    method: str = "quantile",
) -> Tuple[float, int]:
    """Compute Population Stability Index (PSI) for a continuous or discrete variable.

    Rules:
    - Edges always come from expected (train) with NaNs dropped.
    - Outer edges extended to -inf and +inf so out-of-range values in actual land properly.
    - Duplicate quantile edges dropped for low-cardinality discrete columns.
    - Zero bin shares replaced with 1e-4 before computing the logarithm.
    Returns:
        (psi_value, num_bins_used)
    """
    exp = np.asarray(pd.Series(expected).dropna())
    act = np.asarray(pd.Series(actual).dropna())

    if len(exp) == 0 or len(act) == 0:
        return 0.0, 0

    if method == "equal_width":
        edges = np.linspace(exp.min(), exp.max(), bins + 1)
    elif method == "quantile":
        quantiles = np.linspace(0, 1, bins + 1)
        edges = np.unique(np.quantile(exp, quantiles))
    else:
        raise ValueError(f"Unknown binning method '{method}'. Choose 'quantile' or 'equal_width'.")

    # Extend outer edges to -inf and +inf
    edges[0] = -np.inf
    edges[-1] = np.inf

    actual_bins = len(edges) - 1

    exp_counts, _ = np.histogram(exp, bins=edges)
    act_counts, _ = np.histogram(act, bins=edges)

    exp_share = exp_counts / len(exp)
    act_share = act_counts / len(act)

    # Replace zero shares with 1e-4 before log
    exp_share = np.where(exp_share == 0, 1e-4, exp_share)
    act_share = np.where(act_share == 0, 1e-4, act_share)

    psi_val = np.sum((act_share - exp_share) * np.log(act_share / exp_share))
    return float(psi_val), actual_bins


def evaluate_drift_and_subgroups() -> Dict[str, Any]:
    """Compute PSI across both methods, evaluate v1 by tower, and update findings.yaml."""
    cfg = load_params()
    num_cols = cfg["columns"]["numeric"]
    features = cfg["columns"]["features"]
    target = cfg["columns"]["target"]
    configured_method = cfg.get("drift", {}).get("binning", "quantile")
    num_bins = cfg.get("drift", {}).get("bins", 10)

    # 1. Load data: clean train data, load later.csv AS GIVEN
    train_clean = clean_train_data(load_raw("train"))
    later_raw = load_raw("later")

    # 2. Compute PSI with BOTH methods
    results = []
    psi_dict_selected = {}

    print(f"{'Feature':24s} | {'Quantile PSI':14s} | {'Quantile Bins':13s} | {'EqualWidth PSI':14s} | {'EqualWidth Bins':15s}")
    print("-" * 90)

    for col in num_cols:
        psi_q, bins_q = psi(train_clean[col], later_raw[col], bins=num_bins, method="quantile")
        psi_ew, bins_ew = psi(train_clean[col], later_raw[col], bins=num_bins, method="equal_width")

        results.append({
            "feature": col,
            "psi_quantile": round(psi_q, 4),
            "bins_quantile": bins_q,
            "psi_equal_width": round(psi_ew, 4),
            "bins_equal_width": bins_ew,
        })

        selected_val = psi_q if configured_method == "quantile" else psi_ew
        psi_dict_selected[col] = round(selected_val, 4)

        print(f"{col:24s} | {psi_q:14.4f} | {bins_q:13d} | {psi_ew:14.4f} | {bins_ew:15d}")

    df_both = pd.DataFrame(results)
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / "psi_both_methods.csv"
    df_both.to_csv(csv_path, index=False)
    print(f"\nSaved comparison to {csv_path}")

    most_drifted = max(psi_dict_selected, key=psi_dict_selected.get)
    print(f"Configured binning method '{configured_method}': most drifted feature is '{most_drifted}' (PSI = {psi_dict_selected[most_drifted]:.4f})")

    # 3. Compute v1's MAE on later.csv per tower
    tracking_uri = cfg["mlflow"]["tracking_uri"]
    reg_model_name = cfg["mlflow"]["registered_model_name"]
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    # Load version 1 explicitly
    v1_model_uri = f"models:/{reg_model_name}/1"
    model_v1 = mlflow.sklearn.load_model(v1_model_uri)

    mae_by_tower = {}
    print("\n--- v1 MAE on later.csv by Tower ---")
    for tower_name, grp in later_raw.groupby("tower"):
        y_true = grp[target]
        preds = model_v1.predict(grp[features])
        tower_mae = float(mean_absolute_error(y_true, preds))
        mae_by_tower[str(tower_name)] = round(tower_mae, 2)
        print(f"Tower '{tower_name}': MAE = {tower_mae:.2f} litres (n={len(grp)})")

    worst_served = max(mae_by_tower, key=mae_by_tower.get)
    print(f"Worst-served group: {worst_served} (MAE = {mae_by_tower[worst_served]:.2f} litres)")

    # 4. Update reports/findings.yaml (monitoring section only; does not touch registry or model.joblib)
    findings_path = ROOT / "reports" / "findings.yaml"
    with open(findings_path, "r", encoding="utf-8") as f:
        findings = yaml.safe_load(f)

    if "monitoring" not in findings or findings["monitoring"] is None:
        findings["monitoring"] = {}

    findings["monitoring"]["psi"] = psi_dict_selected
    findings["monitoring"]["most_drifted_feature"] = most_drifted
    findings["monitoring"]["mae_by_group"] = {
        "Tower A": mae_by_tower.get("Tower A"),
        "Tower B": mae_by_tower.get("Tower B"),
        "Tower C": mae_by_tower.get("Tower C"),
    }
    findings["monitoring"]["worst_served_group"] = worst_served

    with open(findings_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(findings, f, sort_keys=False)

    print(f"Updated monitoring section in {findings_path}")
    return {
        "psi_selected": psi_dict_selected,
        "most_drifted": most_drifted,
        "mae_by_tower": mae_by_tower,
        "worst_served": worst_served,
    }


def main():
    evaluate_drift_and_subgroups()


if __name__ == "__main__":
    main()
