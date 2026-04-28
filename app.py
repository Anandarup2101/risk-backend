from fastapi import FastAPI, HTTPException, Request
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import json
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
    get_speciality_donut_data,
    format_number,
)

from utils.global_explainability import (
    get_global_shap_data,
    get_global_feature_matrix,
)

from utils.individual_explainability import (
    get_waterfall_from_cache,
    get_tree_vote,
    get_pdp_from_cache,
    get_global_pdp_data,
)

from utils.llm_utils import summarize_shap_for_llm

from utils.llm_graph import (
    run_llm_task,
    run_global_shap_explanations,
    clear_chat_memory,
    CHAT_MEMORY,
)

app = FastAPI()

# ---------------- PATHS ----------------

BASE_DIR = Path(__file__).resolve().parent
USERS_PATH = BASE_DIR / "services" / "inputs" / "users.xlsx"

# Uncomment these only when using React build inside backend
# FRONTEND_DIR = BASE_DIR / "frontend_build"
# FRONTEND_STATIC_DIR = FRONTEND_DIR / "static"
# FRONTEND_INDEX = FRONTEND_DIR / "index.html"

# ---------------- CACHE FILES ----------------

CACHE_DIR = BASE_DIR / "data_cache"

with open(CACHE_DIR / "shap_summary.json") as f:
    SHAP_SUMMARY = json.load(f)

with open(CACHE_DIR / "shap_bar.json") as f:
    SHAP_BAR = json.load(f)

with open(CACHE_DIR / "pdp_data.json") as f:
    PDP_DATA = json.load(f)

with open(CACHE_DIR / "waterfall_data.json") as f:
    WF_DATA = json.load(f)

# ---------------- CORS ----------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
        "message": "Login successful",
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


# ---------------- FILTER HELPERS ----------------

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


# ---------------- DASHBOARD DATA BUILDER ----------------

def build_dashboard_response(df: pd.DataFrame):
    df = clean_df(df)

    total_exposure = float(df["ar_exposure"].sum()) if "ar_exposure" in df.columns else 0.0

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


# ---------------- DASHBOARD ----------------

@app.post("/dashboard")
def dashboard(filters: DashboardFilters):
    df = get_processed_data().copy()
    df = apply_dashboard_filters(df, filters)
    return build_dashboard_response(df)


# ---------------- LLM CONTEXT HELPERS ----------------

def build_dashboard_overview_payload(cards: dict, charts: dict, table: list) -> dict:
    donut = charts.get("donut", {})
    exposure_by_tier = charts.get("line", [])
    specialty_donut = charts.get("specialty_donut", {})
    bubble = charts.get("bubble", [])

    top_risky_hospitals = sorted(
        table,
        key=lambda x: float(x.get("risk_score", 0) or 0),
        reverse=True,
    )[:5]

    top_exposure_hospitals = sorted(
        table,
        key=lambda x: float(x.get("ar_exposure", 0) or 0),
        reverse=True,
    )[:5]

    return {
        "cards": cards,
        "risk_distribution": {
            "labels": donut.get("labels", []),
            "values": donut.get("values", []),
        },
        "exposure_by_tier": exposure_by_tier,
        "specialty_distribution": {
            "labels": specialty_donut.get("labels", []),
            "values": specialty_donut.get("values", []),
        },
        "top_risky_hospitals": [
            {
                "hospital_name": h.get("hospital_name"),
                "risk_score": h.get("risk_score"),
                "risk_tier": h.get("risk_tier"),
                "ar_exposure": h.get("ar_exposure"),
                "specialty": h.get("specialty"),
                "dso_30d": h.get("dso_30d"),
                "delay_ratio": h.get("delay_ratio"),
                "ops_stress": h.get("ops_stress"),
            }
            for h in top_risky_hospitals
        ],
        "top_exposure_hospitals": [
            {
                "hospital_name": h.get("hospital_name"),
                "risk_score": h.get("risk_score"),
                "risk_tier": h.get("risk_tier"),
                "ar_exposure": h.get("ar_exposure"),
                "specialty": h.get("specialty"),
            }
            for h in top_exposure_hospitals
        ],
        "bubble_summary": {
            "total_points": len(bubble),
            "high_risk_points": len(
                [b for b in bubble if b.get("risk_tier") == "High"]
            ),
            "critical_points": len(
                [b for b in bubble if b.get("risk_tier") == "Critical"]
            ),
        },
    }


def find_hospital_from_text(text: str, df: pd.DataFrame):
    text_clean = str(text or "").strip().lower()

    if not text_clean or "hospital_name" not in df.columns:
        return None

    hospital_names = df["hospital_name"].dropna().astype(str).unique().tolist()

    hospital_names = sorted(hospital_names, key=len, reverse=True)

    for name in hospital_names:
        if str(name).strip().lower() in text_clean:
            return str(name)

    return None


def extract_hospital_from_memory(session_id: str, df: pd.DataFrame):
    memory = CHAT_MEMORY.get(session_id, [])

    for msg in reversed(memory):
        content = str(msg.get("content", ""))
        hospital_name = find_hospital_from_text(content, df)

        if hospital_name:
            return hospital_name

    return None


def build_individual_hospital_context(hospital_name: str):
    df = get_processed_data().copy()
    df = clean_df(df)

    selected = df[
        df["hospital_name"].astype(str).str.strip().str.lower()
        == str(hospital_name).strip().lower()
    ]

    if selected.empty:
        return None

    row = selected.iloc[0]
    X_hospital = selected[features].iloc[[0]].copy().fillna(0)

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
            "waterfall": get_waterfall_from_cache(hospital_name, WF_DATA),
            "tree_vote": get_tree_vote(X_hospital),
            "pdp_plots": get_pdp_from_cache(row, PDP_DATA, SHAP_BAR),
        }
    )


def build_smart_ask_context(prompt: str, session_id: str = "default") -> dict:
    df = get_processed_data().copy()
    df = clean_df(df)

    dashboard_payload = build_dashboard_response(df)

    compact_shap_payload = summarize_shap_for_llm(SHAP_SUMMARY)

    context = {
        "full_dashboard_data": dashboard_payload,
        "global_model_context": {
            "shap_summary_compact": compact_shap_payload,
            "shap_bar": SHAP_BAR,
        },
        "available_columns": list(df.columns),
        "instruction_to_llm": (
            "Use full_dashboard_data.table when the user asks for all hospitals, "
            "lists, filtered groups, critical hospitals, specialty-specific hospitals, "
            "or hospital comparisons. Do not rely only on top_risky_hospitals."
        ),
    }

    hospital_name = find_hospital_from_text(prompt, df)

    if not hospital_name:
        hospital_name = extract_hospital_from_memory(session_id, df)

    if hospital_name:
        context["matched_hospital_name"] = hospital_name
        context["individual_hospital_context"] = build_individual_hospital_context(
            hospital_name
        )

    return to_python_type(context)

# ---------------- LLM: DASHBOARD OVERVIEW ----------------

@app.post("/llm/dashboard-overview")
async def dashboard_overview(request: Request):
    try:
        data = await request.json()

        cards = data.get("cards", {})
        charts = data.get("charts", {})
        table = data.get("table", [])

        overview_payload = build_dashboard_overview_payload(cards, charts, table)

        result = run_llm_task(
            "dashboard_overview",
            {
                "overview_payload": overview_payload,
            },
        )

        if result.get("error"):
            return {
                "overview": "Unable to generate dashboard overview.",
                "error": result.get("error"),
            }

        return {
            "overview": result.get("overview", "Unable to generate dashboard overview.")
        }

    except Exception as e:
        return {
            "overview": "Unable to generate dashboard overview.",
            "error": str(e),
        }


# ---------------- GLOBAL SHAP ----------------

@app.get("/global-shap")
def global_explainability():
    compact_payload = summarize_shap_for_llm(SHAP_SUMMARY)

    llm_result = run_global_shap_explanations(
        compact_payload=compact_payload,
        shap_bar=SHAP_BAR,
    )

    return to_python_type(
        {
            "shap_summary": SHAP_SUMMARY,
            "shap_bar": SHAP_BAR,
            "summary_explanation": llm_result.get(
                "summary_explanation",
                "Unable to generate beeswarm explanation.",
            ),
            "bar_explanation": llm_result.get(
                "bar_explanation",
                "Unable to generate bar plot explanation.",
            ),
            "explanation_payload": compact_payload,
        }
    )


# ---------------- INDIVIDUAL HOSPITAL ----------------

@app.post("/individual-hospital")
async def individual_explainability(request: Request):
    data = await request.json()
    hospital_name = data.get("hospital_name")

    if not hospital_name:
        raise HTTPException(status_code=400, detail="hospital_name required")

    hospital_context = build_individual_hospital_context(hospital_name)

    if not hospital_context:
        raise HTTPException(status_code=404, detail="Hospital not found")

    return to_python_type(hospital_context)


# ---------------- WATERFALL LLM EXPLANATION ----------------

@app.post("/llm/waterfall-explanation")
async def waterfall_explanation(request: Request):
    try:
        data = await request.json()

        hospital = data.get("hospital", {}) or {}
        waterfall = data.get("waterfall", {}) or {}
        waterfall_features = waterfall.get("features", []) or []

        if not waterfall_features:
            return {
                "explanation": "No waterfall features available for explanation."
            }

        top_features = sorted(
            waterfall_features,
            key=lambda x: float(x.get("abs_shap_value", 0) or 0),
            reverse=True,
        )

        increasing_features = [
            {
                "feature": f.get("feature"),
                "feature_value": f.get("feature_value"),
                "impact": f.get("shap_value"),
            }
            for f in top_features
            if f.get("direction") == "increase"
        ]

        decreasing_features = [
            {
                "feature": f.get("feature"),
                "feature_value": f.get("feature_value"),
                "impact": f.get("shap_value"),
            }
            for f in top_features
            if f.get("direction") == "decrease"
        ]

        llm_payload = {
            "hospital": {
                "hospital_name": hospital.get("hospital_name"),
                "risk_score": hospital.get("risk_score"),
                "risk_tier": hospital.get("risk_tier"),
                "specialty": hospital.get("specialty"),
                "trend_indicator": hospital.get("trend_indicator"),
            },
            "top_risk_increasing_factors": increasing_features,
            "top_risk_reducing_factors": decreasing_features,
        }

        result = run_llm_task(
            "waterfall_explanation",
            {
                "llm_payload": llm_payload,
            },
        )

        if result.get("error"):
            return {
                "explanation": "Unable to generate waterfall explanation.",
                "error": result.get("error"),
            }

        return {
            "explanation": result.get(
                "explanation",
                "Unable to generate waterfall explanation.",
            ),
            "explanation_payload": llm_payload,
        }

    except Exception as e:
        return {
            "explanation": "Unable to generate waterfall explanation.",
            "error": str(e),
        }


# ---------------- CHARTS ----------------

@app.post("/charts/geo-heatmap")
async def geo_heatmap_chart(request: Request):
    filters = await request.json() or {}
    df = get_processed_data().copy()
    df = apply_dict_filters(df, filters)

    return to_python_type(get_geo_heatmap_data(df))


@app.post("/charts/cluster-scatter")
async def cluster_scatter_chart(request: Request):
    data = await request.json() or {}
    filters = data.get("filters", {}) or {}
    n_clusters = data.get("n_clusters", 3)

    df = get_processed_data().copy()
    df = apply_dict_filters(df, filters)

    return to_python_type(
        get_cluster_scatter_data(df=df, n_clusters=n_clusters)
    )


# ---------------- CONTEXT-AWARE LLM ASK ----------------

@app.post("/llm/ask")
async def ask_llm(request: Request):
    try:
        data = await request.json()

        prompt = data.get("prompt")
        session_id = data.get("session_id", "default")

        if not prompt:
            return {"answer": "No prompt provided"}

        context = build_smart_ask_context(prompt, session_id=session_id)

        result = run_llm_task(
            "smart_ask",
            {
                "prompt": prompt,
                "context": context,
                "session_id": session_id,
            },
        )

        if result.get("error"):
            return {
                "answer": "Error generating response",
                "error": result.get("error"),
            }

        return {
            "answer": result.get("answer", "Error generating response")
        }

    except Exception as e:
        return {
            "answer": "Error generating response",
            "error": str(e),
        }


# ---------------- CACHE ----------------

@app.post("/refresh")
def refresh():
    get_processed_data.cache_clear()
    get_global_shap_data.cache_clear()
    get_global_feature_matrix.cache_clear()
    get_global_pdp_data.cache_clear()
    clear_chat_memory()
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

#     if FRONTEND_INDEX.exists():
#         return FileResponse(FRONTEND_INDEX)

#     raise HTTPException(status_code=404)


# ---------------- MAIN ----------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)