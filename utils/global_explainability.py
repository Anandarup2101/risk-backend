import shap
import numpy as np
import pandas as pd
from functools import lru_cache

from services.data_loader import load_data
from services.feature_engineering import engineer_features
from services.prediction import model, features, run_prediction


rf = model


# -----------------------------
# BUILD FULL GLOBAL FEATURE DATA
# -----------------------------
@lru_cache(maxsize=1)
def get_global_feature_matrix():
    """
    Loads the full pipeline data once, applies feature engineering
    and prediction, and returns only the model feature matrix X.
    """

    df = load_data()
    df = engineer_features(df)
    df = run_prediction(df)

    X = df[features].copy()

    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X


# -----------------------------
# SHARED GLOBAL SHAP CACHE
# -----------------------------
@lru_cache(maxsize=1)
def get_global_shap_data():
    """
    Computes SHAP once for the full global dataset and caches:
    - X
    - explainer
    - raw SHAP array
    - explanation object (for waterfall)
    """

    X = get_global_feature_matrix()

    explainer = shap.TreeExplainer(rf)

    # For summary/bar style outputs
    shap_vals = explainer.shap_values(X)

    if isinstance(shap_vals, list):
        shap_array = shap_vals[1]
    else:
        shap_array = shap_vals[:, :, 1]

    # For waterfall-style outputs
    shap_exp = explainer(X)

    return {
        "X": X,
        "explainer": explainer,
        "shap_array": shap_array,
        "shap_exp": shap_exp
    }


# -----------------------------
# INTERNAL VALIDATION
# -----------------------------
def _validate_input_X(X):
    """
    Optional compatibility validation.
    Your app.py currently passes X into get_shap_summary/get_shap_bar.
    We don't recompute SHAP for that X; we use cached global SHAP.
    This just checks that the columns are aligned.
    """

    if X is None:
        return

    missing = set(features) - set(X.columns)
    if missing:
        raise ValueError(f"Missing features in input X: {missing}")


# -----------------------------
# SHAP SUMMARY (returns plot data)
# -----------------------------
def get_shap_summary(X=None):
    """
    Returns frontend-ready SHAP summary data using cached global SHAP.
    Compatible with current app.py usage: get_shap_summary(X)
    """

    _validate_input_X(X)

    cached = get_global_shap_data()
    global_X = cached["X"]
    shap_array = cached["shap_array"]

    points = []

    for feature_idx, feature in enumerate(global_X.columns):
        for row_idx in range(len(global_X)):
            feature_value = global_X.iloc[row_idx, feature_idx]

            try:
                feature_value = float(feature_value)
            except Exception:
                feature_value = 0.0

            points.append({
                "feature": feature,
                "shap_value": float(shap_array[row_idx, feature_idx]),
                "feature_value": feature_value
            })

    return {
        "features": list(global_X.columns),
        "points": points
    }


# -----------------------------
# SHAP BAR (returns plot data)
# -----------------------------
def get_shap_bar(X=None):
    """
    Returns frontend-ready SHAP global importance bar data
    using cached global SHAP.
    Compatible with current app.py usage: get_shap_bar(X)
    """

    _validate_input_X(X)

    cached = get_global_shap_data()
    global_X = cached["X"]
    shap_array = cached["shap_array"]

    importance = np.abs(shap_array).mean(axis=0)

    results = [
        {
            "feature": feature,
            "importance": float(value)
        }
        for feature, value in zip(global_X.columns, importance)
    ]

    results = sorted(
        results,
        key=lambda item: item["importance"],
        reverse=True
    )

    return results
