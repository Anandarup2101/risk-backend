from pathlib import Path
import json

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

# DATA UTILS
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

# LLM UTILS
from utils.llm_utils import (
    summarize_shap_for_llm,
    build_dashboard_overview_payload,
    build_smart_ask_context,
    build_waterfall_explanation_payload,
)

# CHARTS
from utils.charts import (
    get_geo_heatmap_data,
    get_cluster_scatter_data,
)

# EXPLAINABILITY
from utils.global_explainability import (
    get_global_shap_data,
    get_global_feature_matrix,
)

from utils.individual_explainability import (
    get_global_pdp_data,
)

# LLM GRAPH
from utils.llm_graph import (
    run_llm_task,
    run_global_shap_explanations,
    clear_chat_memory,
    CHAT_MEMORY,
)

app = FastAPI()

# ---------------- GLOBAL VARIABLES ----------------

BASE_DIR = Path(__file__).resolve().parent
USERS_PATH = BASE_DIR / "services" / "inputs" / "users.xlsx"
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

# ---------------- API ROUTES ----------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/login")
async def login(request: Request):
    data = await request.json()

    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password required")

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


@app.post("/dashboard")
def dashboard(filters: DashboardFilters):
    df = get_processed_data().copy()
    df = apply_dashboard_filters(df, filters)
    return build_dashboard_response(df)


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

    except Exception as e:
        return {
            "overview": "Unable to generate dashboard overview.",
            "error": str(e),
        }


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


@app.post("/individual-hospital")
async def individual_explainability(request: Request):
    data = await request.json()
    hospital_name = data.get("hospital_name")

    if not hospital_name:
        raise HTTPException(status_code=400, detail="hospital_name required")

    hospital_context = build_individual_hospital_context(
        hospital_name=hospital_name,
        wf_data=WF_DATA,
        pdp_data=PDP_DATA,
        shap_bar=SHAP_BAR,
    )

    if not hospital_context:
        raise HTTPException(status_code=404, detail="Hospital not found")

    return to_python_type(hospital_context)


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

    except Exception as e:
        return {
            "explanation": "Unable to generate waterfall explanation.",
            "error": str(e),
        }


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
        get_cluster_scatter_data(
            df=df,
            n_clusters=n_clusters,
        )
    )


@app.post("/llm/ask")
async def ask_llm(request: Request):
    try:
        data = await request.json()

        prompt = data.get("prompt")
        session_id = data.get("session_id", "default")

        if not prompt:
            return {"answer": "No prompt provided"}

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

    except Exception as e:
        return {
            "answer": "Error generating response",
            "error": str(e),
        }


@app.post("/refresh")
def refresh():
    get_processed_data.cache_clear()
    get_global_shap_data.cache_clear()
    get_global_feature_matrix.cache_clear()
    get_global_pdp_data.cache_clear()
    clear_chat_memory()

    return {"message": "Cache cleared"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)