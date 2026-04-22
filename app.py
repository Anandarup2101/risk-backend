from fastapi import FastAPI, HTTPException, Request
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Uncomment these only when serving React build from FastAPI
# from fastapi.responses import FileResponse
# from fastapi.staticfiles import StaticFiles

from services.data_loader import load_data
from services.feature_engineering import engineer_features
from services.prediction import run_prediction, features

from utils.charts import (
    get_donut_data,
    get_bubble_data,
    get_line_data,
    get_geo_heatmap_data,
    get_cluster_scatter_data,
)

from utils.global_explainability import (
    get_shap_summary,
    get_shap_bar,
    get_global_shap_data,
    get_global_feature_matrix,
)

from utils.indivudual_explainability import (
    get_waterfall,
    get_tree_vote,
    get_pdp_actions,
    get_global_pdp_data,
)

app = FastAPI()

# ---------------- PATHS ----------------

BASE_DIR = Path(__file__).resolve().parent
USERS_PATH = BASE_DIR / "services" / "inputs" / "users.xlsx"

# Uncomment these only when using React build inside backend
# FRONTEND_DIR = BASE_DIR / "frontend_build"
# FRONTEND_STATIC_DIR = FRONTEND_DIR / "static"
# FRONTEND_INDEX = FRONTEND_DIR / "index.html"

# ---------------- CORS ----------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",   # local React frontend
        "http://127.0.0.1:3000",
        "http://localhost:8000",   # if serving build from backend later
        "http://127.0.0.1:8000",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- STATIC FRONTEND ----------------

# Uncomment only when serving React build from FastAPI
# if FRONTEND_STATIC_DIR.exists():
#     app.mount("/static", StaticFiles(directory=FRONTEND_STATIC_DIR), name="static")

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
    if pd.isna(obj):
        return None
    return obj


# ---------------- USERS ----------------

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
        raise HTTPException(status_code=400, detail="Username and password required")

    users_df = load_users()
    user_row = users_df[users_df["username"] == username]

    if user_row.empty:
        raise HTTPException(status_code=400, detail="Invalid username")

    db_user = str(user_row.iloc[0]["username"])
    db_pass = str(user_row.iloc[0]["password_hash"])
    role = str(user_row.iloc[0]["role"])

    if password != db_pass:
        raise HTTPException(status_code=400, detail="Invalid password")

    return {
        "username": db_user,
        "role": role,
        "message": "Login successful"
    }


# ---------------- DATA PIPELINE ----------------

@lru_cache(maxsize=1)
def get_processed_data():
    df = load_data()
    df = engineer_features(df)
    df = run_prediction(df)
    return df


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------- DASHBOARD ----------------

@app.post("/dashboard")
def dashboard(filters: DashboardFilters):
    df = get_processed_data().copy()

    bars = get_line_data(df)

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

    df = df.replace([np.inf, -np.inf], 0).fillna(0)

    total_exposure = float(df["ar_exposure"].sum()) if "ar_exposure" in df.columns else 0.0
    risky_exposure = float(df[df["risk_flag"] == 1]["ar_exposure"].sum()) if len(df) > 0 else 0.0

    exposure_at_risk = round((risky_exposure / total_exposure) * 100, 2) if total_exposure > 0 else 0.0

    cards = {
        "total_hospitals": int(len(df)),
        "total_at_risk": int(df["risk_flag"].sum()) if "risk_flag" in df.columns else 0,
        "total_exposure": total_exposure,
        "exposure_at_risk": exposure_at_risk,
    }

    charts = {
        "donut": get_donut_data(df),
        "bubble": get_bubble_data(df),
        "line": bars,
    }

    table = df.to_dict(orient="records")

    return to_python_type({
        "cards": cards,
        "charts": charts,
        "table": table,
    })


# ---------------- GLOBAL SHAP ----------------

@app.get("/explainability/global")
def global_explainability():
    df = get_processed_data().copy()

    X = df[features].copy()
    X = X.replace([np.inf, -np.inf], 0).fillna(0)

    return to_python_type({
        "shap_summary": get_shap_summary(X),
        "shap_bar": get_shap_bar(X),
    })


# ---------------- INDIVIDUAL SHAP ----------------

@app.post("/explainability/individual")
async def individual_explainability(request: Request):
    data = await request.json()
    hospital_name = data.get("hospital_name")

    if not hospital_name:
        raise HTTPException(status_code=400, detail="hospital_name required")

    df = get_processed_data().copy()
    df = df.replace([np.inf, -np.inf], 0).fillna(0)

    selected = df[df["hospital_name"] == hospital_name]

    if selected.empty:
        raise HTTPException(status_code=404, detail="Hospital not found")

    idx = int(selected.index[0])
    row = selected.iloc[0]
    X_hospital = selected[features].iloc[[0]].copy().fillna(0)

    return to_python_type({
        "hospital": {
            "hospital_name": str(row["hospital_name"]),
            "risk_score": float(row["risk_score"]),
            "risk_tier": str(row["risk_tier"]),
            "trend_indicator": int(row["dso_trend"]) if pd.notna(row.get("dso_trend")) else None,
            "location": str(row["location"]) if pd.notna(row.get("location")) else None,
            "specialty": str(row["specialty"]) if pd.notna(row.get("specialty")) else None,
            "latitude": float(row["latitude"]) if pd.notna(row.get("latitude")) else None,
            "longitude": float(row["longitude"]) if pd.notna(row.get("longitude")) else None,
        },
        "waterfall": get_waterfall(idx),
        "tree_vote": get_tree_vote(X_hospital),
        "pdp_plots": get_pdp_actions(row),
    })


# ---------------- CHARTS ----------------

@app.post("/charts/geo-heatmap")
async def geo_heatmap_chart(request: Request):
    filters = await request.json() or {}
    df = get_processed_data().copy()

    # -----------------------------
    # LOCATION FILTER
    # Expected shape:
    # {
    #   "location": {
    #       "min_lat": 47,
    #       "max_lat": 55,
    #       "min_lon": 5.5,
    #       "max_lon": 15.5
    #   },
    #   "exposure_range": {
    #       "min": 100000,
    #       "max": 920000
    #   },
    #   "specialty": [...],
    #   "risk_tier": [...]
    # }
    # -----------------------------
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

    # -----------------------------
    # SPECIALTY FILTER
    # -----------------------------
    specialty = filters.get("specialty")
    if specialty:
        df = df[df["specialty"].isin(specialty)]

    # -----------------------------
    # RISK TIER FILTER
    # -----------------------------
    risk_tier = filters.get("risk_tier")
    if risk_tier:
        df = df[df["risk_tier"].isin(risk_tier)]

    # -----------------------------
    # EXPOSURE RANGE FILTER
    # -----------------------------
    exposure_range = filters.get("exposure_range", {}) or {}
    min_exp = exposure_range.get("min")
    max_exp = exposure_range.get("max")

    if min_exp is not None and max_exp is not None:
        df = df[
            (df["ar_exposure"] >= float(min_exp))
            & (df["ar_exposure"] <= float(max_exp))
        ]

    df = df.replace([np.inf, -np.inf], 0).fillna(0)

    return to_python_type(get_geo_heatmap_data(df))

# @app.post("/charts/geo-heatmap")
# async def geo_heatmap_chart(request: Request):
#     filters = await request.json() or {}
#     df = get_processed_data().copy()

#     min_lat = filters.get("min_lat")
#     max_lat = filters.get("max_lat")
#     min_lon = filters.get("min_lon")
#     max_lon = filters.get("max_lon")

#     if all(v is not None for v in [min_lat, max_lat, min_lon, max_lon]):
#         df = df[
#             (df["latitude"] >= float(min_lat))
#             & (df["latitude"] <= float(max_lat))
#             & (df["longitude"] >= float(min_lon))
#             & (df["longitude"] <= float(max_lon))
#         ]

#     return to_python_type(get_geo_heatmap_data(df))


@app.post("/charts/cluster-scatter")
async def cluster_scatter_chart(request: Request):
    data = await request.json() or {}
    filters = data.get("filters", {}) or {}
    n_clusters = data.get("n_clusters", 3)

    df = get_processed_data().copy()

    # -----------------------------
    # LOCATION FILTER
    # Expected inside filters:
    # {
    #   "location": {
    #       "min_lat": 47,
    #       "max_lat": 55,
    #       "min_lon": 5.5,
    #       "max_lon": 15.5
    #   },
    #   "exposure_range": {
    #       "min": 100000,
    #       "max": 920000
    #   },
    #   "specialty": [...],
    #   "risk_tier": [...]
    # }
    # -----------------------------
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

    # -----------------------------
    # SPECIALTY FILTER
    # -----------------------------
    specialty = filters.get("specialty")
    if specialty:
        df = df[df["specialty"].isin(specialty)]

    # -----------------------------
    # RISK TIER FILTER
    # -----------------------------
    risk_tier = filters.get("risk_tier")
    if risk_tier:
        df = df[df["risk_tier"].isin(risk_tier)]

    # -----------------------------
    # EXPOSURE RANGE FILTER
    # -----------------------------
    exposure_range = filters.get("exposure_range", {}) or {}
    min_exp = exposure_range.get("min")
    max_exp = exposure_range.get("max")

    if min_exp is not None and max_exp is not None:
        df = df[
            (df["ar_exposure"] >= float(min_exp))
            & (df["ar_exposure"] <= float(max_exp))
        ]

    df = df.replace([np.inf, -np.inf], 0).fillna(0)

    return to_python_type(
        get_cluster_scatter_data(df=df, n_clusters=n_clusters)
    )



# @app.post("/charts/cluster-scatter")
# async def cluster_scatter_chart(request: Request):
#     data = await request.json() or {}
#     n_clusters = data.get("n_clusters", 3)

#     df = get_processed_data().copy()

#     return to_python_type(
#         get_cluster_scatter_data(df=df, n_clusters=n_clusters)
#     )


# ---------------- CACHE ----------------

@app.post("/refresh")
def refresh():
    get_processed_data.cache_clear()
    get_global_shap_data.cache_clear()
    get_global_feature_matrix.cache_clear()
    get_global_pdp_data.cache_clear()
    return {"message": "Cache cleared"}


# ---------------- FRONTEND ----------------

# Uncomment these only when using React build inside backend
# @app.get("/")
# def serve_frontend():
#     if FRONTEND_INDEX.exists():
#         return FileResponse(FRONTEND_INDEX)
#     raise HTTPException(status_code=404, detail="Frontend not found")


# @app.get("/{path:path}")
# def serve_react(path: str):
#     if path.startswith(("api", "docs", "openapi", "charts", "explainability")):
#         raise HTTPException(status_code=404)
#
#     if FRONTEND_INDEX.exists():
#         return FileResponse(FRONTEND_INDEX)
#
#     raise HTTPException(status_code=404)


# ---------------- MAIN ----------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)