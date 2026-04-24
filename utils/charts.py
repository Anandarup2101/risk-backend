import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import math


def format_number(n):
    try:
        if n is None:
            return "0"

        n = float(n)

        if math.isnan(n) or math.isinf(n):
            return "0"

        sign = "-" if n < 0 else ""
        n = abs(n)

        def fmt(value):
            return str(int(value)) if value.is_integer() else f"{value:.1f}".rstrip("0").rstrip(".")

        if n >= 1_000_000_000:
            return f"{sign}{fmt(n / 1_000_000_000)}B"
        elif n >= 1_000_000:
            return f"{sign}{fmt(n / 1_000_000)}M"
        elif n >= 1_000:
            return f"{sign}{fmt(n / 1_000)}K"
        else:
            return f"{sign}{int(n):,}"

    except Exception:
        return "0"


def safe_float(value, default=0.0):
    try:
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    except Exception:
        return default


def safe_int(value, default=0):
    try:
        value = int(value)
        return value
    except Exception:
        return default


def get_donut_data(df):
    risk_labels = ["Low Risk", "Medium Risk", "High Risk", "Critical"]

    if df.empty or "risk_tier" not in df.columns:
        return {
            "labels": risk_labels,
            "values": [0, 0, 0, 0]
        }

    return {
        "labels": risk_labels,
        "values": [
            int((df["risk_tier"] == "Low").sum()),
            int((df["risk_tier"] == "Medium").sum()),
            int((df["risk_tier"] == "High").sum()),
            int((df["risk_tier"] == "Critical").sum())
        ]
    }


def get_speciality_donut_data(df):
    specialties = [
        "Cardiology",
        "Oncology",
        "Neurology",
        "Orthopedics",
        "Surgery",
        "General Medicine",
        "Dermatology",
        "Gastroenterology",
        "Pulmonology",
        "Endocrinology"
    ]

    if df.empty or "specialty" not in df.columns:
        return {
            "labels": specialties,
            "values": [0 for _ in specialties]
        }

    df_1nf = (
        df.assign(specialty=df["specialty"].astype(str).str.split(","))
        .explode("specialty")
        .reset_index(drop=True)
    )

    df_1nf["specialty"] = df_1nf["specialty"].astype(str).str.strip()

    return {
        "labels": specialties,
        "values": [
            int((df_1nf["specialty"] == specialty).sum())
            for specialty in specialties
        ]
    }


def get_bubble_data(df):
    required_cols = [
        "dso_30d",
        "credit_used",
        "total_payments",
        "hospital_name",
        "risk_flag"
    ]

    if df.empty:
        return []

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        return []

    plot_df = df.copy()
    plot_df = plot_df.replace([np.inf, -np.inf], 0).fillna(0)

    return plot_df.apply(
        lambda row: {
            "x": safe_float(row.get("dso_30d", 0)),
            "y": safe_float(row.get("credit_used", 0)),
            "size": safe_float(row.get("total_payments", 0)),
            "label": str(row.get("hospital_name", "")),
            "risk": safe_int(row.get("risk_flag", 0)),
            "risk_tier": str(row.get("risk_tier", "Low"))
        },
        axis=1
    ).to_list()


def get_line_data(df):
    risk_order = ["Low", "Medium", "High", "Critical"]

    if df.empty:
        return []

    required_cols = ["risk_tier", "ar_exposure", "hospital_name", "risk_score"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        return []

    plot_df = df.copy()
    plot_df = plot_df.replace([np.inf, -np.inf], 0).fillna(0)

    grouped = (
        plot_df.groupby("risk_tier", as_index=False)
        .agg(
            total_exposure_raw=("ar_exposure", "sum"),
            hospital_count=("hospital_name", "count"),
            avg_risk_score=("risk_score", "mean")
        )
    )

    grouped["risk_tier"] = pd.Categorical(
        grouped["risk_tier"],
        categories=risk_order,
        ordered=True
    )

    grouped = grouped.sort_values("risk_tier")

    return [
        {
            "tier": str(row["risk_tier"]),
            "total_exposure": format_number(row["total_exposure_raw"]),
            "total_exposure_raw": safe_float(row["total_exposure_raw"]),
            "hospital_count": safe_int(row["hospital_count"]),
            "avg_risk_score": safe_float(row["avg_risk_score"])
        }
        for _, row in grouped.iterrows()
    ]


def get_geo_heatmap_data(df):
    required_cols = ["latitude", "longitude", "risk_score", "hospital_name"]
    missing = [col for col in required_cols if col not in df.columns]

    if missing or df.empty:
        return {
            "title": "Hospital Risk Heatmap (Geographic Density)",
            "x_label": "Longitude",
            "y_label": "Latitude",
            "color_label": "Risk % (Intensity)",
            "points": []
        }

    plot_df = df.copy()
    plot_df = plot_df.dropna(subset=["latitude", "longitude", "risk_score"])
    plot_df = plot_df.replace([np.inf, -np.inf], 0).fillna(0)

    points = plot_df.apply(
        lambda row: {
            "hospital_name": str(row.get("hospital_name", "")),
            "latitude": safe_float(row.get("latitude", 0)),
            "longitude": safe_float(row.get("longitude", 0)),
            "risk_percent": safe_float(row.get("risk_score", 0)),
            "risk_tier": str(row.get("risk_tier", "")),
            "specialty": str(row.get("specialty", "")),
            "ar_exposure": safe_float(row.get("ar_exposure", 0)),
            "ar_exposure_display": format_number(row.get("ar_exposure", 0)),
            "dso_30d": safe_float(row.get("dso_30d", 0))
        },
        axis=1
    ).to_list()

    return {
        "title": "Hospital Risk Heatmap (Geographic Density)",
        "x_label": "Longitude",
        "y_label": "Latitude",
        "color_label": "Risk % (Intensity)",
        "points": points
    }


def get_cluster_scatter_data(df, n_clusters=3):
    required_cols = [
        "hospital_name",
        "dso_30d",
        "credit_used",
        "risk_score"
    ]

    missing = [col for col in required_cols if col not in df.columns]

    if missing or df.empty:
        return {
            "title": "Clustered Hospital Scatter Plot",
            "x_label": "DSO (30D)",
            "y_label": "Exposure (Credit Used)",
            "size_label": "Risk %",
            "clusters": []
        }

    plot_df = df.copy()
    plot_df = plot_df.replace([np.inf, -np.inf], 0).fillna(0)

    cluster_features = ["dso_30d", "credit_used"]
    cluster_data = plot_df[cluster_features].copy()

    n_clusters = min(int(n_clusters), len(plot_df))
    n_clusters = max(n_clusters, 1)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    plot_df["cluster"] = kmeans.fit_predict(cluster_data)

    clusters = []

    for cluster_id in sorted(plot_df["cluster"].unique()):
        cluster_df = plot_df[plot_df["cluster"] == cluster_id]

        cluster_points = cluster_df.apply(
            lambda row: {
                "hospital_name": str(row.get("hospital_name", "")),
                "x": safe_float(row.get("dso_30d", 0)),
                "y": safe_float(row.get("credit_used", 0)),
                "risk_percent": safe_float(row.get("risk_score", 0)),
                "bubble_size": safe_float(row.get("risk_score", 0)),
                "cluster": safe_int(row.get("cluster", 0)),
                "risk_tier": str(row.get("risk_tier", "")),
                "specialty": str(row.get("specialty", "")),
                "ar_exposure": safe_float(row.get("ar_exposure", 0)),
                "ar_exposure_display": format_number(row.get("ar_exposure", 0))
            },
            axis=1
        ).to_list()

        center = kmeans.cluster_centers_[int(cluster_id)]

        clusters.append({
            "cluster_id": int(cluster_id),
            "center": {
                "x": safe_float(center[0]),
                "y": safe_float(center[1])
            },
            "points": cluster_points
        })

    return {
        "title": "Clustered Hospital Scatter Plot",
        "x_label": "DSO (30D)",
        "y_label": "Exposure (Credit Used)",
        "size_label": "Risk %",
        "clusters": clusters
    }