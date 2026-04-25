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
                "negative_impact_count": 0
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

        compact_features.append({
            "feature": feature,
            "avg_shap": round(s["avg_shap"] / count, 4),
            "avg_abs_shap": round(s["avg_abs_shap"] / count, 4),
            "min_shap": round(s["min_shap"], 4),
            "max_shap": round(s["max_shap"], 4),
            "avg_feature_value": round(s["avg_feature_value"] / count, 2),
            "positive_impact_count": s["positive_impact_count"],
            "negative_impact_count": s["negative_impact_count"]
        })

    compact_features = sorted(
        compact_features,
        key=lambda x: x["avg_abs_shap"],
        reverse=True
    )

    return {
        "summary_plot_interpretation_data": compact_features
    }