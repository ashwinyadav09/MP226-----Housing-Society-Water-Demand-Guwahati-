"""Retraining pipeline and model promotion decision for MP226.

Retrains v2 on clean train.csv + first 70% of later.csv (sorted by date).
Evaluates v1 vs v2 on the last 30% check set.
Promotes v2 as champion only if its MAE is at least 15% lower than v1's.

Run with:
    python -m mp226.retrain
"""

from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import yaml

from .clean import clean_train_data
from .config import ROOT, load_params
from .data import load_raw, raw_path, sha256_of_file
from .model import build_pipeline, evaluate_predictions


def should_promote(mae_v1: float, mae_v2: float, threshold: float = 0.15) -> bool:
    """Pure function deciding if v2 should replace v1.

    Promotes only if relative MAE improvement >= threshold (e.g., 15%).
    """
    if mae_v1 <= 0:
        return False
    relative_improvement = (mae_v1 - mae_v2) / mae_v1
    return bool(relative_improvement >= threshold)


def retrain_and_evaluate() -> Dict[str, Any]:
    """Execute v2 retraining, evaluate on last 30% of later.csv, and apply promotion rule."""
    cfg = load_params()
    features = cfg["columns"]["features"]
    target = cfg["columns"]["target"]
    later_train_frac = cfg.get("retrain", {}).get("later_train_fraction", 0.7)
    threshold = cfg.get("retrain", {}).get("promotion_improvement", 0.15)
    tracking_uri = cfg["mlflow"]["tracking_uri"]
    exp_name = cfg["mlflow"]["experiment_name"]
    reg_model_name = cfg["mlflow"]["registered_model_name"]
    ridge_alpha = float(cfg.get("train", {}).get("models", {}).get("ridge", {}).get("alpha", 10.0))

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    # Guard: check if version 2 already exists
    try:
        reg_model = client.get_registered_model(reg_model_name)
        existing_versions = [int(v.version) for v in reg_model.latest_versions]
        if max(existing_versions, default=1) >= 2:
            print(
                f"Registered model '{reg_model_name}' already has version 2 or higher. "
                "Exiting without changing anything."
            )
            return {"already_retrained": True}
    except Exception:
        pass

    mlflow.set_experiment(exp_name)

    # 1. SHA-256 of original raw train.csv
    data_sha = sha256_of_file(raw_path("train"))

    # 2. Prepare data
    train_clean = clean_train_data(load_raw("train"))
    later_raw = load_raw("later").sort_values("date").reset_index(drop=True)

    n_later_train = int(len(later_raw) * later_train_frac)
    later_train = later_raw.iloc[:n_later_train]
    later_check = later_raw.iloc[n_later_train:]

    print(f"Retraining split: clean train ({len(train_clean)} rows) + later first 70% ({len(later_train)} rows)")
    print(f"Check evaluation set: later last 30% ({len(later_check)} rows)")

    combined_train = pd.concat([train_clean, later_train], ignore_index=True)
    X_train_v2 = combined_train[features]
    y_train_v2 = combined_train[target]

    X_check = later_check[features]
    y_check = later_check[target]

    # 3. Load v1 champion model for baseline check
    v1_uri = f"models:/{reg_model_name}/1"
    model_v1 = mlflow.sklearn.load_model(v1_uri)
    v1_preds = model_v1.predict(X_check)
    mae_v1 = float(mean_absolute_error(y_check, v1_preds))

    # 4. Fit v2 pipeline (same architecture as v1)
    pipe_v2 = build_pipeline(Ridge(alpha=ridge_alpha), scale_numeric=True)

    with mlflow.start_run(run_name="Retrain_v2_with_later") as run:
        v2_run_id = run.info.run_id
        pipe_v2.fit(X_train_v2, y_train_v2)

        v2_preds = pipe_v2.predict(X_check)
        mae_v2 = float(mean_absolute_error(y_check, v2_preds))
        check_metrics = evaluate_predictions(y_check, v2_preds)

        # Log parameters and metrics
        mlflow.log_param("data_sha256", data_sha)
        mlflow.log_param("model_type", "Ridge")
        mlflow.log_param("alpha", ridge_alpha)
        mlflow.log_param("rows_used", len(combined_train))
        mlflow.log_param("train_rows_clean", len(train_clean))
        mlflow.log_param("later_train_rows", len(later_train))
        mlflow.log_param("check_rows", len(later_check))

        mlflow.log_metric("mae_last30", mae_v2)
        mlflow.log_metric("mae", mae_v2)
        mlflow.log_metric("check_rmse", check_metrics["rmse"])
        mlflow.log_metric("check_r2", check_metrics["r2"])

        # Log model artifact with cloudpickle
        mlflow.sklearn.log_model(
            sk_model=pipe_v2,
            artifact_path="model",
            serialization_format="cloudpickle",
        )

    print(f"Run v2 (ID: {v2_run_id}) -> Check Set MAE: {mae_v2:.2f} litres")

    # 5. ALWAYS register as version 2
    model_uri = f"runs:/{v2_run_id}/model"
    reg_v2 = mlflow.register_model(model_uri=model_uri, name=reg_model_name)
    v2_num = str(reg_v2.version)
    print(f"Registered version {v2_num} of '{reg_model_name}'.")

    # 6. Apply promotion rule
    promote = should_promote(mae_v1, mae_v2, threshold)
    decision = "promote" if promote else "keep"
    improvement = (mae_v1 - mae_v2) / mae_v1

    print(f"\n--- Promotion Decision ---")
    print(f"v1 MAE on check set: {mae_v1:.2f} litres")
    print(f"v2 MAE on check set: {mae_v2:.2f} litres")
    print(f"Relative improvement: {improvement * 100:.2f}% (Threshold: {threshold * 100:.1f}%)")
    print(f"Decision: '{decision}'")

    findings_path = ROOT / "reports" / "findings.yaml"
    with open(findings_path, "r", encoding="utf-8") as f:
        findings = yaml.safe_load(f)

    if promote:
        # Move @champion alias to version 2
        client.set_registered_model_alias(name=reg_model_name, alias="champion", version=v2_num)
        print(f"Moved alias 'champion' to version {v2_num}.")

        # Export v2 to models/model.joblib
        model_path = ROOT / "models" / "model.joblib"
        joblib.dump(pipe_v2, model_path)
        print(f"Exported promoted v2 champion to {model_path}")

        # Update champion_version to 2
        findings["tracking"]["champion_version"] = 2
    else:
        print("Decision is 'keep': champion remains version 1.")

    # Record check MAEs and decision in findings.yaml
    findings["monitoring"]["mae_v1_on_check"] = round(mae_v1, 2)
    findings["monitoring"]["mae_v2_on_check"] = round(mae_v2, 2)
    findings["monitoring"]["decision"] = decision

    with open(findings_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(findings, f, sort_keys=False)

    print(f"Updated findings at {findings_path}")
    return {
        "mae_v1": mae_v1,
        "mae_v2": mae_v2,
        "improvement": improvement,
        "decision": decision,
        "champion_version": 2 if promote else 1,
    }


def main():
    retrain_and_evaluate()


if __name__ == "__main__":
    main()
