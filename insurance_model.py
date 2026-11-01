"""Training and inference helpers for the insurance CatBoost model."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


FEATURE_COLUMNS = ["age", "sex", "bmi", "children", "smoker", "region"]
CATEGORICAL_COLUMNS = ["sex", "region"]
MODEL_PARAMETERS = {
    "iterations": 500,
    "learning_rate": 0.03,
    "depth": 3,
    "l2_leaf_reg": 8,
    "loss_function": "RMSE",
    "random_seed": 42,
    "verbose": False,
    "thread_count": -1,
}


def train_models(data: pd.DataFrame) -> dict[str, Any]:
    """Train one CatBoost model per smoker group using log-transformed charges."""
    required = FEATURE_COLUMNS + ["charges"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

    training_data = data[required].dropna().drop_duplicates().copy()
    if training_data.empty:
        raise ValueError("No complete, unique records are available for training.")

    for column in ["age", "bmi", "children", "charges"]:
        training_data[column] = pd.to_numeric(training_data[column], errors="raise")
    if (training_data["charges"] < 0).any():
        raise ValueError("Charges must be nonnegative for the log1p target transform.")
    if not np.isfinite(training_data[["age", "bmi", "children", "charges"]]).all().all():
        raise ValueError("Numeric model inputs and charges must be finite.")

    training_data["smoker"] = training_data["smoker"].astype(str).str.strip().str.lower()
    training_data["sex"] = training_data["sex"].astype(str).str.strip().str.lower()
    training_data["region"] = training_data["region"].astype(str).str.strip().str.lower()

    models: dict[str, CatBoostRegressor] = {}
    for smoker_status in ["no", "yes"]:
        group = training_data.loc[training_data["smoker"] == smoker_status]
        if group.empty:
            raise ValueError(f"Training data contains no records for smoker='{smoker_status}'.")
        model = CatBoostRegressor(**MODEL_PARAMETERS)
        model.fit(
            group[["age", "sex", "bmi", "children", "region"]],
            np.log1p(group["charges"]),
            cat_features=CATEGORICAL_COLUMNS,
        )
        models[smoker_status] = model

    return {
        "format_version": 1,
        "model_name": "smoker-segmented CatBoost",
        "feature_columns": FEATURE_COLUMNS,
        "target_transform": "log1p",
        "models": models,
    }


def predict_charges(bundle: dict[str, Any], features: pd.DataFrame) -> np.ndarray:
    """Predict charges from raw UI fields; CatBoost performs categorical handling."""
    required = bundle.get("feature_columns")
    if required != FEATURE_COLUMNS:
        raise ValueError("Model artifact has an unexpected feature schema.")
    missing = [column for column in FEATURE_COLUMNS if column not in features.columns]
    if missing:
        raise ValueError(f"Prediction data is missing columns: {', '.join(missing)}")

    inputs = features[FEATURE_COLUMNS].copy()
    if inputs.isna().any().any():
        raise ValueError("Prediction inputs cannot contain missing values.")
    for column in ["age", "bmi", "children"]:
        inputs[column] = pd.to_numeric(inputs[column], errors="raise")
    if not np.isfinite(inputs[["age", "bmi", "children"]]).all().all():
        raise ValueError("Numeric prediction inputs must be finite.")

    inputs["smoker"] = inputs["smoker"].astype(str).str.strip().str.lower()
    inputs["sex"] = inputs["sex"].astype(str).str.strip().str.lower()
    inputs["region"] = inputs["region"].astype(str).str.strip().str.lower()
    invalid_smoker_values = sorted(set(inputs["smoker"]) - {"no", "yes"})
    if invalid_smoker_values:
        raise ValueError(
            "Smoker values must be 'no' or 'yes'; got: "
            + ", ".join(invalid_smoker_values)
        )

    models = bundle.get("models")
    if not isinstance(models, dict) or set(models) != {"no", "yes"}:
        raise ValueError("Model artifact must contain models for smokers and non-smokers.")

    predictions = np.empty(len(inputs), dtype=float)
    for smoker_status in ["no", "yes"]:
        mask = inputs["smoker"].eq(smoker_status)
        if mask.any():
            group_features = inputs.loc[mask, ["age", "sex", "bmi", "children", "region"]]
            predictions[mask.to_numpy()] = np.maximum(
                0.0, np.expm1(models[smoker_status].predict(group_features))
            )
    if not np.isfinite(predictions).all():
        raise ValueError("The model returned a non-finite prediction.")
    return predictions
