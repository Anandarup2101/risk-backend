import numpy as np
import pandas as pd
from functools import lru_cache

from services.prediction import model, features
from utils.global_explainability import (
    get_global_shap_data,
    get_global_feature_matrix
)

rf = model


# -----------------------------
# HELPER: linear interpolation
# -----------------------------
def _interp_y(x_grid, y_grid, x_value):
    x_grid = np.asarray(x_grid, dtype=float)
    y_grid = np.asarray(y_grid, dtype=float)
    x_value = float(x_value)

    if len(x_grid) == 1:
        return float(y_grid[0])

    if x_value <= x_grid.min():
        return float(y_grid[0])

    if x_value >= x_grid.max():
        return float(y_grid[-1])

    return float(np.interp(x_value, x_grid, y_grid))


# -----------------------------
# WATERFALL (returns plot data)
# -----------------------------

def get_waterfall_from_cache(hospital_name, WF_DATA):
    if not hospital_name:
        return {"features": []}

    key = str(hospital_name).strip().lower()
    return WF_DATA.get(key, {"features": []})

def get_waterfall(idx, max_display=10):
    """
    Returns frontend-ready waterfall data for one hospital
    using cached global SHAP data.
    """

    cached = get_global_shap_data()
    global_X = cached["X"]
    shap_array = cached["shap_array"]

    idx = int(idx)

    if idx < 0 or idx >= len(global_X):
        raise IndexError(f"idx {idx} out of range for global dataset")

    row = global_X.iloc[idx]

    contributions = []

    for feature_idx, feature in enumerate(global_X.columns):
        raw_value = row.iloc[feature_idx]

        try:
            feature_value = float(raw_value)
        except Exception:
            feature_value = 0.0

        shap_value = float(shap_array[idx, feature_idx])

        contributions.append({
            "feature": str(feature),
            "feature_value": float(feature_value),
            "shap_value": float(shap_value),
            "abs_shap_value": float(abs(shap_value)),
            "direction": "increase" if shap_value >= 0 else "decrease"
        })

    contributions = sorted(
        contributions,
        key=lambda item: item["abs_shap_value"],
        reverse=True
    )

    top_contributions = contributions[:max_display]
    others = contributions[max_display:]

    others_total_shap = float(
        sum(item["shap_value"] for item in others)
    ) if others else 0.0

    return {
        "hospital_index": int(idx),
        "base_value": None,
        "prediction_value": float(np.sum(shap_array[idx])),
        "features": top_contributions,
        "others": {
            "count": int(len(others)),
            "combined_shap_value": float(others_total_shap)
        }
    }


# -----------------------------
# TREE VOTE ONLY SELECTED HOSPITAL
# -----------------------------
def get_tree_vote(X_hospital):
    """
    X_hospital must be a single-row DataFrame containing model features.
    """

    if not isinstance(X_hospital, pd.DataFrame):
        raise ValueError("X_hospital must be a pandas DataFrame")

    if len(X_hospital) != 1:
        raise ValueError("X_hospital must contain exactly one row")

    missing = set(features) - set(X_hospital.columns)
    if missing:
        raise ValueError(f"Missing features in X_hospital: {missing}")

    X_hospital = X_hospital[features].copy()
    X_hospital = X_hospital.replace([np.inf, -np.inf], 0)
    X_hospital = X_hospital.fillna(0)

    tree_preds = np.array([
        tree.predict(X_hospital.values)[0]
        for tree in rf.estimators_
    ])

    yes_votes = int(np.sum(tree_preds == 1))
    total_trees = int(len(rf.estimators_))
    no_votes = int(total_trees - yes_votes)

    risk_percent = float(
        round(yes_votes / total_trees * 100, 2)
    )

    return {
        "risk_percent": risk_percent,
        "yes_votes": yes_votes,
        "no_votes": no_votes,
        "total_trees": total_trees
    }


# -----------------------------
# GLOBAL PDP CACHE
# -----------------------------
@lru_cache(maxsize=1)
def get_global_pdp_data(n_grid=25):
    """
    Computes PDP curves once for all features and caches them.
    Uses the global feature matrix as reference data.
    """

    X_ref = get_global_feature_matrix().copy()
    X_ref = X_ref[features].copy()
    X_ref = X_ref.replace([np.inf, -np.inf], 0)
    X_ref = X_ref.fillna(0)

    pdp_data = {}

    for feature in features:
        col = pd.to_numeric(X_ref[feature], errors="coerce").fillna(0)

        lo = float(np.percentile(col, 5))
        hi = float(np.percentile(col, 95))

        if lo == hi:
            unique_vals = np.sort(col.unique())
            if len(unique_vals) == 1:
                grid = np.array([float(unique_vals[0])], dtype=float)
            else:
                grid = unique_vals.astype(float)
        else:
            grid = np.linspace(lo, hi, n_grid)

        curve = []

        for val in grid:
            X_tmp = X_ref.copy()
            X_tmp[feature] = float(val)

            avg_risk = float(
                rf.predict_proba(X_tmp)[:, 1].mean() * 100
            )

            curve.append({
                "x": float(val),
                "y": float(avg_risk)
            })

        y_values = [pt["y"] for pt in curve]
        min_idx = int(np.argmin(y_values))

        pdp_data[str(feature)] = {
            "feature": str(feature),
            "curve": curve,
            "optimal_value": float(curve[min_idx]["x"]),
            "optimal_risk": float(curve[min_idx]["y"])
        }

    return pdp_data


# -----------------------------
# PDP PLOTS + HOSPITAL POSITION
# -----------------------------

def get_pdp_from_cache(row, PDP_DATA, SHAP_BAR, top_n=13):
    output = []

    top_features = [
        item["feature"]
        for item in SHAP_BAR[:top_n]
        if item["feature"] in PDP_DATA and item["feature"] in row.index
    ]

    for feature in top_features:
        current = float(row.get(feature, 0))
        feature_pdp = PDP_DATA[feature]
        curve = feature_pdp.get("curve", [])

        if not curve:
            continue

        x_grid = [float(pt["x"]) for pt in curve]
        y_grid = [float(pt["y"]) for pt in curve]

        if len(x_grid) == 1:
            hospital_y = y_grid[0]
        elif current <= x_grid[0]:
            hospital_y = y_grid[0]
        elif current >= x_grid[-1]:
            hospital_y = y_grid[-1]
        else:
            hospital_y = y_grid[0]
            for i in range(len(x_grid) - 1):
                if x_grid[i] <= current <= x_grid[i + 1]:
                    x1, x2 = x_grid[i], x_grid[i + 1]
                    y1, y2 = y_grid[i], y_grid[i + 1]
                    ratio = 0 if x2 == x1 else (current - x1) / (x2 - x1)
                    hospital_y = y1 + ratio * (y2 - y1)
                    break

        optimal_value = float(feature_pdp.get("optimal_value", 0))
        optimal_risk = float(feature_pdp.get("optimal_risk", 0))

        x_min = min(x_grid)
        x_max = max(x_grid)
        x_span = max(x_max - x_min, 1e-6)

        value_gap_ratio = abs(current - optimal_value) / x_span
        risk_gap = max(0.0, hospital_y - optimal_risk)

        if value_gap_ratio <= 0.10 and risk_gap <= 2:
            status = "Good"
        elif value_gap_ratio <= 0.30 and risk_gap <= 8:
            status = "Needs Attention"
        else:
            status = "Critical"

        if current > optimal_value:
            suggestion = f"Reduce {feature} toward {round(optimal_value, 4)}"
        elif current < optimal_value:
            suggestion = f"Increase {feature} toward {round(optimal_value, 4)}"
        else:
            suggestion = f"Keep {feature} near {round(optimal_value, 4)}"

        output.append({
            "feature": str(feature),
            "curve": [{"x": float(pt["x"]), "y": float(pt["y"])} for pt in curve],
            "hospital_point": {"x": float(current), "y": float(hospital_y)},
            "current_value": float(current),
            "optimal_value": float(optimal_value),
            "optimal_risk": float(optimal_risk),
            "suggestion": str(suggestion),
            "status": str(status)
        })

    return output


def get_pdp_from_cache(row, PDP_DATA):
    output = []

    for feature, feature_pdp in PDP_DATA.items():
        current = float(row[feature])

        curve = feature_pdp["curve"]

        x_grid = [pt["x"] for pt in curve]
        y_grid = [pt["y"] for pt in curve]

        # simple interpolation
        hospital_y = y_grid[0]
        for i in range(len(x_grid) - 1):
            if x_grid[i] <= current <= x_grid[i + 1]:
                hospital_y = y_grid[i]
                break

        optimal_value = float(feature_pdp["optimal_value"])
        optimal_risk = float(feature_pdp["optimal_risk"])

        output.append({
            "feature": feature,
            "curve": curve,
            "hospital_point": {
                "x": current,
                "y": hospital_y
            },
            "current_value": current,
            "optimal_value": optimal_value,
            "optimal_risk": optimal_risk,
            "suggestion": f"Move toward {round(optimal_value, 2)}",
            "status": "Needs Attention"
        })

    return output

def get_pdp_actions(row):
    """
    Returns full PDP plot data for each feature
    + selected hospital marker on each PDP curve.
    """

    pdp_cache = get_global_pdp_data()

    output = []

    for feature in features:
        current = float(row[feature])

        feature_pdp = pdp_cache[str(feature)]
        curve = feature_pdp["curve"]

        x_grid = [pt["x"] for pt in curve]
        y_grid = [pt["y"] for pt in curve]

        hospital_y = _interp_y(
            x_grid=x_grid,
            y_grid=y_grid,
            x_value=current
        )

        optimal_value = float(feature_pdp["optimal_value"])
        optimal_risk = float(feature_pdp["optimal_risk"])

        diff_ratio = abs(current - optimal_value) / (abs(optimal_value) + 1e-6)

        if diff_ratio <= 0.2:
            status = "Good"
        elif diff_ratio <= 0.5:
            status = "Needs Attention"
        else:
            status = "Critical"

        if current > optimal_value:
            suggestion = f"Reduce {feature} toward {round(optimal_value, 4)}"
        elif current < optimal_value:
            suggestion = f"Increase {feature} toward {round(optimal_value, 4)}"
        else:
            suggestion = f"Keep {feature} near {round(optimal_value, 4)}"

        output.append({
            "feature": str(feature),
            "curve": [
                {
                    "x": float(pt["x"]),
                    "y": float(pt["y"])
                }
                for pt in curve
            ],
            "hospital_point": {
                "x": float(current),
                "y": float(hospital_y)
            },
            "current_value": float(current),
            "optimal_value": float(optimal_value),
            "optimal_risk": float(optimal_risk),
            "suggestion": str(suggestion),
            "status": str(status)
        })

    return output