"""Pipeline definition, preprocessing, and model creation for MP226.

Ensures imputation and feature transformations live inside one unified scikit-learn
Pipeline so that serving receives identical preprocessing without data leakage.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import load_params


def get_feature_columns() -> Dict[str, List[str]]:
    """Return configured feature column names."""
    cfg = load_params()["columns"]
    return {
        "features": cfg["features"],
        "numeric": cfg["numeric"],
        "text": cfg["text"],
        "target": cfg["target"],
    }


def build_pipeline(regressor=None, scale_numeric: bool = False) -> Pipeline:
    """Build scikit-learn pipeline with median imputation and one-hot encoding."""
    cols = get_feature_columns()
    numeric_features = cols["numeric"]
    categorical_features = cols["text"]

    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    numeric_transformer = Pipeline(steps=num_steps)

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore", sparse_output=False
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    if regressor is None:
        regressor = RandomForestRegressor(n_estimators=100, random_state=42)

    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", regressor)])


def evaluate_predictions(y_true, y_pred) -> Dict[str, float]:
    """Compute standard regression metrics."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2}
