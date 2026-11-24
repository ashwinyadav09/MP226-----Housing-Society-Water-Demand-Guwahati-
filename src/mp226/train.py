"""MLflow model training and registry management for MP226.

Tracks all experiment runs to sqlite:///mlflow.db, logging:
- data_sha256 parameter on every run
- mae validation metrics
- full pipeline artifacts (imputer + encoder + regressor)

Registers the best model as version 1 with the 'champion' alias and exports
an identical copy to models/model.joblib.

Run with:
    python -m mp226.train
"""

import hashlib
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

from .clean import clean_train_data
from .config import ROOT, load_params
from .data import load_raw, raw_path, sha256_of_file
from .model import build_pipeline, evaluate_predictions


def get_data_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, str]:
    """Load raw train data, apply cleaning, and split using a time-ordered validation split."""
    params = load_params()
    features = params["columns"]["features"]
    target = params["columns"]["target"]
    val_fraction = params.get("train", {}).get("val_split_fraction", 0.2)

    # 1. Compute SHA-256 of raw training data file
    train_file = raw_path("train")
    data_sha = sha256_of_file(train_file)

    # 2. Clean data and sort by date for time-ordered split
    raw_df = load_raw("train")
    cleaned = clean_train_data(raw_df).sort_values("date").reset_index(drop=True)

    # 3. Time-ordered split: train on earlier data, validate on recent data
    val_size = int(len(cleaned) * val_fraction)
    train_split = cleaned.iloc[:-val_size]
    val_split = cleaned.iloc[-val_size:]

    X_train = train_split[features]
    y_train = train_split[target]
    X_val = val_split[features]
    y_val = val_split[target]

    return X_train, X_val, y_train, y_val, data_sha


def train_and_track() -> Dict[str, Any]:
    """Execute at least 3 MLflow runs, log metrics, register champion, and export model."""
    params = load_params()
    tracking_uri = params["mlflow"]["tracking_uri"]
    exp_name = params["mlflow"]["experiment_name"]
    reg_model_name = params["mlflow"]["registered_model_name"]

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    try:
        reg_model = client.get_registered_model(reg_model_name)
        if reg_model and reg_model.latest_versions:
            print(
                f"Registered model '{reg_model_name}' already has version(s): "
                f"{[v.version for v in reg_model.latest_versions]}. "
                "Exiting without touching the registry, models/model.joblib, or reports/findings.yaml."
            )
            return {"already_registered": True}
    except Exception:
        pass

    mlflow.set_experiment(exp_name)

    X_train, X_val, y_train, y_val, data_sha = get_data_splits()

    models_cfg = params.get("train", {}).get("models", {})
    ridge_cfg = models_cfg.get("ridge", {})
    rf_cfg = models_cfg.get("random_forest", {})
    gb_cfg = models_cfg.get("gradient_boosting", {})

    ridge_alpha = float(ridge_cfg.get("alpha", 10.0))
    rf_n_est = int(rf_cfg.get("n_estimators", 100))
    rf_depth = int(rf_cfg.get("max_depth", 10))
    rf_seed = int(rf_cfg.get("random_state", 42))
    gb_n_est = int(gb_cfg.get("n_estimators", 150))
    gb_lr = float(gb_cfg.get("learning_rate", 0.05))
    gb_depth = int(gb_cfg.get("max_depth", 4))
    gb_seed = int(gb_cfg.get("random_state", 42))

    candidates = [
        {
            "name": "Ridge_Regression",
            "model_type": "Ridge",
            "params": {"alpha": ridge_alpha, "scale_numeric": True},
            "pipeline": build_pipeline(Ridge(alpha=ridge_alpha), scale_numeric=True),
        },
        {
            "name": "Random_Forest",
            "model_type": "RandomForestRegressor",
            "params": {"n_estimators": rf_n_est, "max_depth": rf_depth, "random_state": rf_seed},
            "pipeline": build_pipeline(
                RandomForestRegressor(n_estimators=rf_n_est, max_depth=rf_depth, random_state=rf_seed)
            ),
        },
        {
            "name": "Gradient_Boosting",
            "model_type": "GradientBoostingRegressor",
            "params": {
                "n_estimators": gb_n_est,
                "learning_rate": gb_lr,
                "max_depth": gb_depth,
                "random_state": gb_seed,
            },
            "pipeline": build_pipeline(
                GradientBoostingRegressor(
                    n_estimators=gb_n_est,
                    learning_rate=gb_lr,
                    max_depth=gb_depth,
                    random_state=gb_seed,
                )
            ),
        },
    ]

    best_run_id = None
    best_mae = float("inf")
    best_pipe = None
    best_run_name = None

    for cand in candidates:
        run_name = cand["name"]
        pipe = cand["pipeline"]

        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id

            # Fit pipeline
            pipe.fit(X_train, y_train)

            # Evaluate
            preds = pipe.predict(X_val)
            metrics = evaluate_predictions(y_val, preds)

            # Mandatory logging requirements
            mlflow.log_param("data_sha256", data_sha)
            mlflow.log_param("model_type", cand["model_type"])
            for p_k, p_v in cand["params"].items():
                mlflow.log_param(p_k, p_v)
            mlflow.log_param("train_rows", len(X_train))
            mlflow.log_param("val_rows", len(X_val))

            # Metric name must contain 'mae'
            mlflow.log_metric("val_mae", metrics["mae"])
            mlflow.log_metric("mae", metrics["mae"])
            mlflow.log_metric("val_rmse", metrics["rmse"])
            mlflow.log_metric("val_r2", metrics["r2"])

            # Log model with cloudpickle serialization for MLflow 3 compatibility
            mlflow.sklearn.log_model(
                sk_model=pipe,
                artifact_path="model",
                serialization_format="cloudpickle",
            )

            print(
                f"Run '{run_name}' (ID: {run_id}) -> "
                f"val_mae: {metrics['mae']:.2f}, R2: {metrics['r2']:.4f}"
            )

            if metrics["mae"] < best_mae:
                best_mae = metrics["mae"]
                best_run_id = run_id
                best_pipe = pipe
                best_run_name = run_name

    print(f"\nBest Run: {best_run_name} with MAE: {best_mae:.2f} (ID: {best_run_id})")

    # Register best model
    client = MlflowClient()
    model_uri = f"runs:/{best_run_id}/model"
    reg_version = mlflow.register_model(model_uri=model_uri, name=reg_model_name)
    v_num = str(reg_version.version)

    # Set alias champion
    client.set_registered_model_alias(name=reg_model_name, alias="champion", version=v_num)
    print(f"Registered model '{reg_model_name}' version {v_num} with alias 'champion'.")

    # Export to models/model.joblib
    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "model.joblib"
    joblib.dump(best_pipe, model_path)
    print(f"Saved champion pipeline to {model_path}")

    # Update reports/findings.yaml tracking section
    findings_path = ROOT / "reports" / "findings.yaml"
    import yaml

    with open(findings_path, "r", encoding="utf-8") as f:
        findings = yaml.safe_load(f)

    findings["tracking"] = {
        "experiment_name": exp_name,
        "best_run_id": best_run_id,
        "registered_model_name": reg_model_name,
        "champion_version": int(v_num),
    }

    with open(findings_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(findings, f, sort_keys=False)
    print(f"Updated tracking section in {findings_path}")

    return {
        "best_run_id": best_run_id,
        "best_mae": best_mae,
        "champion_version": v_num,
    }


def main():
    train_and_track()


if __name__ == "__main__":
    main()
