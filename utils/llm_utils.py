from utils.data_utils import (
    get_processed_data,
    clean_df,
    build_dashboard_response,
    build_individual_hospital_context,
    to_python_type,
)


def summarize_shap_for_llm(shap_summary):
    points = shap_summary.get("points", [])

    feature_stats = {}

    for p in points:
        feature = p.get("feature")
        shap_value = float(p.get("shap_value", 0) or 0)
        feature_value = float(p.get("feature_value", 0) or 0)

        if not feature:
            continue

        if feature not in feature_stats:
            feature_stats[feature] = {
                "count": 0,
                "avg_shap": 0,
                "avg_abs_shap": 0,
                "min_shap": shap_value,
                "max_shap": shap_value,
                "avg_feature_value": 0,
                "positive_impact_count": 0,
                "negative_impact_count": 0,
            }

        s = feature_stats[feature]
        s["count"] += 1
        s["avg_shap"] += shap_value
        s["avg_abs_shap"] += abs(shap_value)
        s["avg_feature_value"] += feature_value
        s["min_shap"] = min(s["min_shap"], shap_value)
        s["max_shap"] = max(s["max_shap"], shap_value)

        if shap_value > 0:
            s["positive_impact_count"] += 1
        elif shap_value < 0:
            s["negative_impact_count"] += 1

    compact_features = []

    for feature, s in feature_stats.items():
        count = max(s["count"], 1)

        compact_features.append(
            {
                "feature": feature,
                "avg_shap": round(s["avg_shap"] / count, 4),
                "avg_abs_shap": round(s["avg_abs_shap"] / count, 4),
                "min_shap": round(s["min_shap"], 4),
                "max_shap": round(s["max_shap"], 4),
                "avg_feature_value": round(s["avg_feature_value"] / count, 2),
                "positive_impact_count": s["positive_impact_count"],
                "negative_impact_count": s["negative_impact_count"],
            }
        )

    compact_features = sorted(
        compact_features,
        key=lambda x: x["avg_abs_shap"],
        reverse=True,
    )

    return {
        "summary_plot_interpretation_data": compact_features
    }


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


def find_hospital_from_text(text: str, df):
    text_clean = str(text or "").strip().lower()

    if not text_clean or "hospital_name" not in df.columns:
        return None

    hospital_names = df["hospital_name"].dropna().astype(str).unique().tolist()
    hospital_names = sorted(hospital_names, key=len, reverse=True)

    for name in hospital_names:
        if str(name).strip().lower() in text_clean:
            return str(name)

    return None


def extract_hospital_from_memory(session_id: str, df, chat_memory: dict):
    memory = chat_memory.get(session_id, [])

    for msg in reversed(memory):
        content = str(msg.get("content", ""))
        hospital_name = find_hospital_from_text(content, df)

        if hospital_name:
            return hospital_name

    return None


def build_smart_ask_context(
    prompt: str,
    session_id: str,
    chat_memory: dict,
    shap_summary: dict,
    shap_bar: list,
    wf_data: dict,
    pdp_data: dict,
) -> dict:
    df = get_processed_data().copy()
    df = clean_df(df)

    dashboard_payload = build_dashboard_response(df)
    compact_shap_payload = summarize_shap_for_llm(shap_summary)

    context = {
        "full_dashboard_data": dashboard_payload,
        "global_model_context": {
            "shap_summary_compact": compact_shap_payload,
            "shap_bar": shap_bar,
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
        hospital_name = extract_hospital_from_memory(
            session_id=session_id,
            df=df,
            chat_memory=chat_memory,
        )

    if hospital_name:
        context["matched_hospital_name"] = hospital_name
        context["individual_hospital_context"] = build_individual_hospital_context(
            hospital_name=hospital_name,
            wf_data=wf_data,
            pdp_data=pdp_data,
            shap_bar=shap_bar,
        )

    return to_python_type(context)


def build_waterfall_explanation_payload(hospital: dict, waterfall: dict):
    waterfall_features = waterfall.get("features", []) or []

    if not waterfall_features:
        return None

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

    return {
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