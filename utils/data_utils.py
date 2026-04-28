from functools import lru_cache
from typing import List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel

from services.data_loader import load_data
from services.feature_engineering import engineer_features
from services.prediction import run_prediction, features

from utils.charts import (
    get_donut_data,
    get_bubble_data,
    get_line_data,
    get_speciality_donut_data,
    format_number,
)

from utils.individual_explainability import (
    get_waterfall_from_cache,
    get_tree_vote,
    get_pdp_from_cache,
)
# ---------------- FILTER MODELS ----------------

class LocationFilter(BaseModel):
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float


class ExposureRange(BaseModel):
    min: float
    max: float


class DashboardFilters(BaseModel):
    location: Optional[LocationFilter] = None
    specialty: Optional[List[str]] = None
    risk_tier: Optional[List[str]] = None
    exposure_range: Optional[ExposureRange] = None


# ---------------- SAFE JSON ----------------

def to_python_type(obj):
    if isinstance(obj, dict):
        return {str(k): to_python_type(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [to_python_type(v) for v in obj]

    if isinstance(obj, tuple):
        return [to_python_type(v) for v in obj]

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.bool_):
        return bool(obj)

    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass

    return obj


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], 0).fillna(0)


# ---------------- USERS ----------------

@lru_cache(maxsize=1)
def load_users(users_path):
    df = pd.read_excel(users_path)
    return df.fillna("")


# ---------------- DATA PIPELINE ----------------

@lru_cache(maxsize=1)
def get_processed_data():
    df = load_data()
    df = engineer_features(df)
    df = run_prediction(df)
    return df


# ---------------- FILTERS ----------------

def apply_dashboard_filters(df: pd.DataFrame, filters: DashboardFilters) -> pd.DataFrame:
    df = df.copy()

    if filters.location:
        df = df[
            (df["latitude"] >= filters.location.min_lat)
            & (df["latitude"] <= filters.location.max_lat)
            & (df["longitude"] >= filters.location.min_lon)
            & (df["longitude"] <= filters.location.max_lon)
        ]

    if filters.specialty:
        df = df[df["specialty"].isin(filters.specialty)]

    if filters.risk_tier:
        df = df[df["risk_tier"].isin(filters.risk_tier)]

    if filters.exposure_range:
        df = df[
            (df["ar_exposure"] >= filters.exposure_range.min)
            & (df["ar_exposure"] <= filters.exposure_range.max)
        ]

    return clean_df(df)


def apply_dict_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    df = df.copy()

    location = filters.get("location", {}) or {}

    min_lat = location.get("min_lat")
    max_lat = location.get("max_lat")
    min_lon = location.get("min_lon")
    max_lon = location.get("max_lon")

    if all(v is not None for v in [min_lat, max_lat, min_lon, max_lon]):
        df = df[
            (df["latitude"] >= float(min_lat))
            & (df["latitude"] <= float(max_lat))
            & (df["longitude"] >= float(min_lon))
            & (df["longitude"] <= float(max_lon))
        ]

    specialty = filters.get("specialty")
    if specialty:
        df = df[df["specialty"].isin(specialty)]

    risk_tier = filters.get("risk_tier")
    if risk_tier:
        df = df[df["risk_tier"].isin(risk_tier)]

    exposure_range = filters.get("exposure_range", {}) or {}
    min_exp = exposure_range.get("min")
    max_exp = exposure_range.get("max")

    if min_exp is not None and max_exp is not None:
        df = df[
            (df["ar_exposure"] >= float(min_exp))
            & (df["ar_exposure"] <= float(max_exp))
        ]

    return clean_df(df)


# ---------------- DASHBOARD RESPONSE ----------------

def build_dashboard_response(df: pd.DataFrame):
    df = clean_df(df)

    total_exposure = (
        float(df["ar_exposure"].sum())
        if "ar_exposure" in df.columns
        else 0.0
    )

    risky_exposure = (
        float(df[df["risk_flag"] == 1]["ar_exposure"].sum())
        if len(df) > 0 and "risk_flag" in df.columns and "ar_exposure" in df.columns
        else 0.0
    )

    exposure_at_risk = (
        round((risky_exposure / total_exposure) * 100, 2)
        if total_exposure > 0
        else 0.0
    )

    cards = {
        "total_hospitals": int(len(df)),
        "total_at_risk": int(df["risk_flag"].sum()) if "risk_flag" in df.columns else 0,
        "total_exposure": format_number(total_exposure),
        "total_exposure_raw": total_exposure,
        "exposure_at_risk": exposure_at_risk,
    }

    charts = {
        "donut": get_donut_data(df),
        "bubble": get_bubble_data(df),
        "line": get_line_data(df),
        "specialty_donut": get_speciality_donut_data(df),
    }

    table = df.to_dict(orient="records")

    return to_python_type(
        {
            "cards": cards,
            "charts": charts,
            "table": table,
        }
    )


# ---------------- HOSPITAL CONTEXT ----------------

def build_individual_hospital_context(
    hospital_name: str,
    wf_data: dict,
    pdp_data: dict,
    shap_bar: list,
):
    df = get_processed_data().copy()
    df = clean_df(df)

    selected = df[
        df["hospital_name"].astype(str).str.strip().str.lower()
        == str(hospital_name).strip().lower()
    ]

    if selected.empty:
        return None

    row = selected.iloc[0]
    x_hospital = selected[features].iloc[[0]].copy().fillna(0)

    return to_python_type(
        {
            "hospital": {
                "hospital_name": str(row.get("hospital_name", "")),
                "risk_score": float(row.get("risk_score", 0)),
                "risk_tier": str(row.get("risk_tier", "Unknown")),
                "trend_indicator": int(row["dso_trend"])
                if pd.notna(row.get("dso_trend"))
                else None,
                "location": str(row.get("location"))
                if pd.notna(row.get("location"))
                else None,
                "specialty": str(row.get("specialty"))
                if pd.notna(row.get("specialty"))
                else None,
                "ar_exposure": float(row.get("ar_exposure", 0) or 0),
                "dso_30d": float(row.get("dso_30d", 0) or 0),
                "delay_ratio": float(row.get("delay_ratio", 0) or 0),
                "ops_stress": float(row.get("ops_stress", 0) or 0),
            },
            "waterfall": get_waterfall_from_cache(hospital_name, wf_data),
            "tree_vote": get_tree_vote(x_hospital),
            "pdp_plots": get_pdp_from_cache(row, pdp_data, shap_bar),
        }
    )