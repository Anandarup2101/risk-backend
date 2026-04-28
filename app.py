from pathlib import Path
import json

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

# ================= FRONTEND SERVING IMPORTS - OPTIONAL =================
# Uncomment ONLY if you want FastAPI to serve the React production build.
# Keep commented if frontend runs separately with npm start / Azure Static Web Apps.

# from fastapi.responses import FileResponse
# from fastapi.staticfiles import StaticFiles


# ================= DATA UTILS =================

from utils.data_utils import (
    DashboardFilters,
    load_users,
    get_processed_data,
    apply_dashboard_filters,
    apply_dict_filters,
    build_dashboard_response,
    build_individual_hospital_context,
    to_python_type,
)


# ================= LLM UTILS =================

from utils.llm_utils import (
    summarize_shap_for_llm,
    build_dashboard_overview_payload,
    build_smart_ask_context,
    build_waterfall_explanation_payload,
)


# ================= CHARTS =================

from utils.charts import (
    get_geo_heatmap_data,
    get_cluster_scatter_data,
)


# ================= EXPLAINABILITY =================

from utils.global_explainability import (
    get_global_shap_data,
    get_global_feature_matrix,
)

from utils.individual_explainability import (
    get_global_pdp_data,
)


# ================= LLM GRAPH =================

from utils.llm_graph import (
    run_llm_task,
    run_global_shap_explanations,
    clear_chat_memory,
    CHAT_MEMORY,
)


# ================= APP =================

app = FastAPI()


# ================= GLOBAL VARIABLES =================

BASE_DIR = Path(__file__).resolve().parent
USERS_PATH = BASE_DIR / "services" / "inputs" / "users.xlsx"
CACHE_DIR = BASE_DIR / "data_cache"


# ================= FRONTEND PATHS - OPTIONAL =================
# Uncomment ONLY when using React build inside backend.
#
# Expected structure:
#
# risk-backend/
# ├── app.py
# ├── frontend_build/
# │   ├── index.html
# │   └── static/
#
# Steps:
# 1. Go to frontend folder
# 2. Run: npm run build
# 3. Copy frontend/build folder into backend as frontend_build
# 4. Uncomment imports above + paths below + static mount + frontend routes

# FRONTEND_DIR = BASE_DIR / "frontend_build"
# FRONTEND_STATIC_DIR = FRONTEND_DIR / "static"
# FRONTEND_INDEX = FRONTEND_DIR / "index.html"


# ================= CACHE FILES =================

with open(CACHE_DIR / "shap_summary.json") as f:
    SHAP_SUMMARY = json.load(f)

with open(CACHE_DIR / "shap_bar.json") as f:
    SHAP_BAR = json.load(f)

with open(CACHE_DIR / "pdp_data.json") as f:
    PDP_DATA = json.load(f)

with open(CACHE_DIR / "waterfall_data.json") as f:
    WF_DATA = json.load(f)


# ================= CORS =================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace "*" with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================= STATIC FRONTEND - OPTIONAL =================
# Uncomment ONLY when serving React build from FastAPI.
# Keep commented when running frontend separately at localhost:3000.

# if FRONTEND_STATIC_DIR.exists():
#     app.mount(
#         "/static",
#         StaticFiles(directory=FRONTEND_STATIC_DIR),
#         name="static",
#     )


# ================= HEALTH =================

@app.get("/health")
def health():
    return {"status": "ok"}


# ================= LOGIN =================

@app.post("/login")
async def login(request: Request):
    try:
        data = await request.json()

        username = data.get("username")
        password = data.get("password")

        if not username or not password:
            raise HTTPException(
                status_code=400,
                detail="Username and password required",
            )

        users_df = load_users(USERS_PATH)
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

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Login failed: {str(e)}",
        )


# ================= DASHBOARD =================

@app.post("/dashboard")
def dashboard(filters: DashboardFilters):
    try:
        df = get_processed_data().copy()
        df = apply_dashboard_filters(df, filters)

        return build_dashboard_response(df)

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Dashboard data generation failed: {str(e)}",
        )


# ================= LLM DASHBOARD OVERVIEW =================

@app.post("/llm/dashboard-overview")
async def dashboard_overview(request: Request):
    try:
        data = await request.json()

        overview_payload = build_dashboard_overview_payload(
            cards=data.get("cards", {}),
            charts=data.get("charts", {}),
            table=data.get("table", []),
        )

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
            "overview": result.get(
                "overview",
                "Unable to generate dashboard overview.",
            )
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Dashboard overview generation failed: {str(e)}",
        )


# ================= GLOBAL SHAP =================

@app.get("/global-shap")
def global_explainability():
    try:
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

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Global SHAP explanation failed: {str(e)}",
        )


# ================= INDIVIDUAL HOSPITAL =================

@app.post("/individual-hospital")
async def individual_explainability(request: Request):
    try:
        data = await request.json()
        hospital_name = data.get("hospital_name")

        if not hospital_name:
            raise HTTPException(
                status_code=400,
                detail="hospital_name required",
            )

        hospital_context = build_individual_hospital_context(
            hospital_name=hospital_name,
            wf_data=WF_DATA,
            pdp_data=PDP_DATA,
            shap_bar=SHAP_BAR,
        )

        if not hospital_context:
            raise HTTPException(status_code=404, detail="Hospital not found")

        return to_python_type(hospital_context)

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Individual hospital explanation failed: {str(e)}",
        )


# ================= WATERFALL LLM EXPLANATION =================

@app.post("/llm/waterfall-explanation")
async def waterfall_explanation(request: Request):
    try:
        data = await request.json()

        llm_payload = build_waterfall_explanation_payload(
            hospital=data.get("hospital", {}) or {},
            waterfall=data.get("waterfall", {}) or {},
        )

        if not llm_payload:
            return {
                "explanation": "No waterfall features available for explanation."
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

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Waterfall explanation failed: {str(e)}",
        )


# ================= GEO HEATMAP =================

@app.post("/charts/geo-heatmap")
async def geo_heatmap_chart(request: Request):
    try:
        filters = await request.json() or {}

        df = get_processed_data().copy()
        df = apply_dict_filters(df, filters)

        return to_python_type(get_geo_heatmap_data(df))

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Geo heatmap generation failed: {str(e)}",
        )


# ================= CLUSTER SCATTER =================

@app.post("/charts/cluster-scatter")
async def cluster_scatter_chart(request: Request):
    try:
        data = await request.json() or {}

        filters = data.get("filters", {}) or {}
        n_clusters = data.get("n_clusters", 3)

        df = get_processed_data().copy()
        df = apply_dict_filters(df, filters)

        return to_python_type(
            get_cluster_scatter_data(
                df=df,
                n_clusters=n_clusters,
            )
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Cluster scatter generation failed: {str(e)}",
        )


# ================= CHATBOT / SMART ASK =================

@app.post("/llm/ask")
async def ask_llm(request: Request):
    try:
        data = await request.json()

        prompt = data.get("prompt")
        session_id = data.get("session_id", "default")

        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt required")

        context = build_smart_ask_context(
            prompt=prompt,
            session_id=session_id,
            chat_memory=CHAT_MEMORY,
            shap_summary=SHAP_SUMMARY,
            shap_bar=SHAP_BAR,
            wf_data=WF_DATA,
            pdp_data=PDP_DATA,
        )

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

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM response generation failed: {str(e)}",
        )


# ================= REFRESH CACHE =================

@app.post("/refresh")
def refresh():
    try:
        get_processed_data.cache_clear()
        get_global_shap_data.cache_clear()
        get_global_feature_matrix.cache_clear()
        get_global_pdp_data.cache_clear()
        clear_chat_memory()

        return {"message": "Cache cleared"}

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Cache refresh failed: {str(e)}",
        )


# ================= FRONTEND ROUTING - OPTIONAL =================
# Uncomment ONLY when serving React build from FastAPI.
#
# Important:
# - Keep this section BELOW all API routes.
# - This prevents React routing from hijacking backend APIs.
# - Do NOT enable this while running frontend separately with npm start.

# @app.get("/")
# def serve_frontend():
#     if FRONTEND_INDEX.exists():
#         return FileResponse(FRONTEND_INDEX)
#
#     raise HTTPException(
#         status_code=404,
#         detail="Frontend build not found. Run npm run build and copy build folder.",
#     )


# @app.get("/{path:path}")
# def serve_react(path: str):
#     # Do not let React catch backend/API/docs routes
#     if path.startswith(
#         (
#             "health",
#             "login",
#             "dashboard",
#             "llm",
#             "charts",
#             "global-shap",
#             "individual-hospital",
#             "refresh",
#             "docs",
#             "redoc",
#             "openapi.json",
#         )
#     ):
#         raise HTTPException(status_code=404)
#
#     if FRONTEND_INDEX.exists():
#         return FileResponse(FRONTEND_INDEX)
#
#     raise HTTPException(
#         status_code=404,
#         detail="Frontend build not found.",
#     )


# ================= MAIN =================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )