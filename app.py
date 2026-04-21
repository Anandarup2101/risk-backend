from fastapi import FastAPI, HTTPException, Request  # type:ignore
from functools import lru_cache
from pathlib import Path
# from db import get_connection
from services.data_loader import load_data
from services.feature_engineering import engineer_features
from services.prediction import run_prediction, features
from utils.charts import (
    get_donut_data,
    get_bubble_data,
    get_line_data,
    get_geo_heatmap_data,
    get_cluster_scatter_data
)
from utils.global_explainability import (
    get_shap_summary,
    get_shap_bar,
    get_global_shap_data,
    get_global_feature_matrix
)
from utils.indivudual_explainability import (
    get_waterfall,
    get_tree_vote,
    get_pdp_actions,
    get_global_pdp_data
)

from fastapi.middleware.cors import CORSMiddleware  # type:ignore

from pydantic import BaseModel
from typing import List, Optional

import numpy as np
import pandas as pd

app = FastAPI()




# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


# ---------------- SAFE JSON CONVERTER ----------------

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

    if pd.isna(obj):
        return None

    return obj

BASE_DIR = Path(__file__).resolve().parent
USERS_PATH = BASE_DIR / "services" / "inputs" / "users.xlsx"


@lru_cache(maxsize=1)
def load_users():
    df = pd.read_excel(USERS_PATH)
    return df.fillna("")

# ---------------- LOGIN ----------------

@app.post("/login")
async def login(request: Request):

    data = await request.json()

    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        raise HTTPException(
            status_code=400,
            detail="Username and password required"
        )

    try:
        users_df = load_users()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load users file: {str(e)}"
        )

    # Clean
    users_df = users_df.fillna("")

    # Find user
    user_row = users_df[
        users_df["username"] == username
    ]
    print("user_row is:", user_row)
    if user_row.empty:
        raise HTTPException(
            status_code=400,
            detail="Invalid username"
        )

    # Get values
    db_user = str(user_row.iloc[0]["username"])
    print("db_user", db_user)
    db_pass = str(user_row.iloc[0]["password_hash"])
    print("db_pass", db_pass)
    role = str(user_row.iloc[0]["role"])

    # Password check (plain text as per your requirement)
    if password != db_pass:
        raise HTTPException(
            status_code=400,
            detail="Invalid password"
        )

    return {
        "username": db_user,
        "role": role
    }

# ---------------- CACHED PIPELINE ----------------

@lru_cache(maxsize=1)
def get_processed_data():

    df = load_data()

    df = engineer_features(df)

    df = run_prediction(df)

    return df


@app.get("/health")
def health():
    return {"status": "ok"}
# ---------------- DASHBOARD (FILTERED POST API) ----------------

@app.post("/dashboard")
def dashboard(filters: DashboardFilters):

    df = get_processed_data().copy()

    # -------- LOCATION FILTER --------
    if filters.location:
        df = df[
            (df["latitude"] >= filters.location.min_lat) &
            (df["latitude"] <= filters.location.max_lat) &
            (df["longitude"] >= filters.location.min_lon) &
            (df["longitude"] <= filters.location.max_lon)
        ]

    # -------- SPECIALTY FILTER --------
    if filters.specialty:
        df = df[
            df["specialty"].isin(filters.specialty)
        ]

    # -------- RISK TIER FILTER --------
    if filters.risk_tier:
        df = df[
            df["risk_tier"].isin(filters.risk_tier)
        ]

    # -------- EXPOSURE RANGE FILTER --------
    if filters.exposure_range:
        df = df[
            (df["ar_exposure"] >= filters.exposure_range.min) &
            (df["ar_exposure"] <= filters.exposure_range.max)
        ]

    # -------- CLEAN BAD VALUES --------
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)

    # -------- CARDS --------
    total_exposure = float(df["ar_exposure"].sum())

    exposure_at_risk = (
        round(
            float(
                df[df["risk_flag"] == 1]["ar_exposure"].sum()
            ) / total_exposure * 100,
            2
        )
        if total_exposure > 0
        else 0
    )

    cards = {
        "total_hospitals": int(len(df)),
        "total_at_risk": int(df["risk_flag"].sum()),
        "total_exposure": total_exposure,
        "exposure_at_risk": exposure_at_risk
    }

    # -------- CHARTS --------
    charts = {
        "donut": get_donut_data(df),
        "bubble": get_bubble_data(df),
        "line": get_line_data(df)
    }

    # -------- TABLE --------
    table = df.to_dict(orient="records")

    return to_python_type({
        "cards": cards,
        "charts": charts,
        "table": table
    })


# ---------------- GLOBAL SHAP EXPLAINABILITY ----------------

@app.get("/explainability/global")
def global_explainability():

    try:
        df = get_processed_data().copy()

        X = df[features].copy()
        X = X.replace([np.inf, -np.inf], 0)
        X = X.fillna(0)

        shap_summary = get_shap_summary(X)
        shap_bar = get_shap_bar(X)

        return to_python_type({
            "shap_summary": shap_summary,
            "shap_bar": shap_bar
        })

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Explainability failed: {str(e)}"
        )


# ---------------- INDIVIDUAL SHAP EXPLAINABILITY ----------------

@app.post("/explainability/individual")
async def individual_explainability(request: Request):

    try:
        data = await request.json()

        hospital_name = data.get("hospital_name")

        if not hospital_name:
            raise HTTPException(
                status_code=400,
                detail="hospital_name is required"
            )

        df = get_processed_data().copy()

        df = df.replace([np.inf, -np.inf], 0)
        df = df.fillna(0)

        selected = df[
            df["hospital_name"] == hospital_name
        ]

        if selected.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Hospital not found: {hospital_name}"
            )

        idx = int(selected.index[0])
        row = selected.iloc[0]

        X_hospital = selected[features].iloc[[0]].copy()
        X_hospital = X_hospital.replace([np.inf, -np.inf], 0)
        X_hospital = X_hospital.fillna(0)

        waterfall = get_waterfall(idx)
        tree_vote = get_tree_vote(X_hospital)
        pdp_plots = get_pdp_actions(row)

        hospital = {
            "hospital_name": str(row.get("hospital_name")),
            "risk_score": float(row.get("risk_score", 0)),
            "risk_tier": str(row.get("risk_tier")),
            "trend_indicator": (
                int(row.get("dso_trend"))
                if row.get("dso_trend", None) is not None and pd.notna(row.get("dso_trend"))
                else None
            ),
            "location": (
                str(row.get("location"))
                if row.get("location", None) is not None and pd.notna(row.get("location"))
                else None
            ),
            "specialty": (
                str(row.get("specialty"))
                if row.get("specialty", None) is not None and pd.notna(row.get("specialty"))
                else None
            ),
            "latitude": (
                float(row["latitude"])
                if "latitude" in row and pd.notna(row["latitude"])
                else None
            ),
            "longitude": (
                float(row["longitude"])
                if "longitude" in row and pd.notna(row["longitude"])
                else None
            )
        }

        return to_python_type({
            "hospital": hospital,
            "waterfall": waterfall,
            "tree_vote": tree_vote,
            "pdp_plots": pdp_plots
        })

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Individual explainability failed: {str(e)}"
        )


# ---------------- CACHE REFRESH ----------------

@app.post("/refresh")
def refresh():

    get_processed_data.cache_clear()
    get_global_shap_data.cache_clear()
    get_global_feature_matrix.cache_clear()
    get_global_pdp_data.cache_clear()

    return {
        "message": "Cache cleared successfully"
    }

# ---------------- GEO HEATMAP CHART ----------------

@app.post("/charts/geo-heatmap")
async def geo_heatmap_chart(request: Request):

    try:
        filters = await request.json()
        filters = filters or {}

        df = get_processed_data().copy()

        # -------- LOCATION FILTER --------
        min_lat = filters.get("min_lat")
        max_lat = filters.get("max_lat")
        min_lon = filters.get("min_lon")
        max_lon = filters.get("max_lon")

        if all(v is not None for v in [min_lat, max_lat, min_lon, max_lon]):
            df = df[
                (df["latitude"] >= float(min_lat)) &
                (df["latitude"] <= float(max_lat)) &
                (df["longitude"] >= float(min_lon)) &
                (df["longitude"] <= float(max_lon))
            ]

        # -------- SPECIALTY FILTER --------
        specialties = filters.get("specialty", filters.get("speciality"))
        if specialties:
            df = df[
                df["specialty"].isin(specialties)
            ]

        # -------- RISK TIER FILTER --------
        risk_tiers = filters.get("risk_tier")
        if risk_tiers:
            df = df[
                df["risk_tier"].isin(risk_tiers)
            ]

        # -------- EXPOSURE FILTER --------
        min_exposure = filters.get("min_exposure")
        max_exposure = filters.get("max_exposure")

        if min_exposure is not None:
            df = df[
                df["ar_exposure"] >= float(min_exposure)
            ]

        if max_exposure is not None:
            df = df[
                df["ar_exposure"] <= float(max_exposure)
            ]

        df = df.replace([np.inf, -np.inf], 0)
        df = df.fillna(0)

        chart_data = get_geo_heatmap_data(df)

        return chart_data

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Geo heatmap generation failed: {str(e)}"
        )
    
# ---------------- CLUSTER SCATTER CHART ----------------


    try:
        data = await request.json()
        data = data or {}

        filters = data.get("filters", data)
        n_clusters = data.get("n_clusters", 3)

        df = get_processed_data().copy()

        # -------- LOCATION FILTER --------
        min_lat = filters.get("min_lat")
        max_lat = filters.get("max_lat")
        min_lon = filters.get("min_lon")
        max_lon = filters.get("max_lon")

        if all(v is not None for v in [min_lat, max_lat, min_lon, max_lon]):
            df = df[
                (df["latitude"] >= float(min_lat)) &
                (df["latitude"] <= float(max_lat)) &
                (df["longitude"] >= float(min_lon)) &
                (df["longitude"] <= float(max_lon))
            ]

        # -------- SPECIALTY FILTER --------
        specialties = filters.get("specialty", filters.get("speciality"))
        if specialties:
            df = df[
                df["specialty"].isin(specialties)
            ]

        # -------- RISK TIER FILTER --------
        risk_tiers = filters.get("risk_tier")
        if risk_tiers:
            df = df[
                df["risk_tier"].isin(risk_tiers)
            ]

        # -------- EXPOSURE FILTER --------
        min_exposure = filters.get("min_exposure")
        max_exposure = filters.get("max_exposure")

        if min_exposure is not None:
            df = df[
                df["ar_exposure"] >= float(min_exposure)
            ]

        if max_exposure is not None:
            df = df[
                df["ar_exposure"] <= float(max_exposure)
            ]

        df = df.replace([np.inf, -np.inf], 0)
        df = df.fillna(0)

        chart_data = get_cluster_scatter_data(
            df=df,
            n_clusters=n_clusters
        )

        return chart_data

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Cluster scatter generation failed: {str(e)}"
        )
    
    
# ---------------- MAIN ----------------

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )