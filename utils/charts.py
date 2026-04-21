import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def get_donut_data(df):
    return {
        "labels": ["Safe Hospitals", "Risky Hospitals"],
        "values": [
            int((df["risk_flag"] == 0).sum()),
            int((df["risk_flag"] == 1).sum())
        ]
    }


def get_bubble_data(df):
    return df.apply(lambda row: {
        "x": float(row["dso_30d"]),
        "y": float(row["credit_used"]),
        "size": float(row["total_payments"]),
        "label": row["hospital_name"],
        "risk": int(row["risk_flag"])
    }, axis=1).tolist()


# def get_line_data(df):
#     df = df.sort_values(by="risk_score")

#     safe = df[df["risk_flag"] == 0]
#     risky = df[df["risk_flag"] == 1]

#     return {
#         "safe": safe["risk_score"].tolist(),
#         "risky": risky["risk_score"].tolist()
#     }

def get_line_data(df):
    # Ensure you have a date column or create one
    # This is a dummy example - adjust based on your actual 'date' column
    if 'date' not in df.columns:
        # Fallback if no date column: create monthly buckets
        df = df.copy()
        df['date'] = pd.to_datetime(df.get('timestamp', pd.Timestamp.now()))
        df.set_index('date', inplace=True)
        monthly_risk = df.resample('M')['risk_score'].mean().reset_index()
        monthly_risk['date'] = monthly_risk['date'].dt.strftime('%b')
    else:
        monthly_risk = df.groupby('date')['risk_score'].mean().reset_index()

    # Return list of dicts for JSON serialization
    return monthly_risk.rename(columns={'date': 'month', 'risk_score': 'value'}).to_dict(orient='records')


# -----------------------------
# GEO RISK HEATMAP DATA
# -----------------------------
def get_geo_heatmap_data(df):
    """
    Returns plot-ready data for geographic risk heatmap.
    Frontend should render this as a scatter/heat style map.
    """

    required_cols = ["latitude", "longitude", "risk_score", "hospital_name"]
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns for geo heatmap: {missing}")

    plot_df = df.copy()

    plot_df = plot_df.dropna(
        subset=["latitude", "longitude", "risk_score"]
    )

    plot_df = plot_df.replace([np.inf, -np.inf], 0)
    plot_df = plot_df.fillna(0)

    points = plot_df.apply(
        lambda row: {
            "hospital_name": str(row["hospital_name"]),
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "risk_percent": float(row["risk_score"]),
            "risk_tier": str(row["risk_tier"]) if "risk_tier" in plot_df.columns else None,
            "specialty": str(row["specialty"]) if "specialty" in plot_df.columns else None
        },
        axis=1
    ).tolist()

    return {
        "title": "Hospital Risk Heatmap (Geographic Density)",
        "x_label": "Longitude",
        "y_label": "Latitude",
        "color_label": "Risk % (Intensity)",
        "points": points
    }


# -----------------------------
# CLUSTERED RISK SCATTER DATA
# -----------------------------
def get_cluster_scatter_data(df, n_clusters=3):
    """
    Returns plot-ready clustered scatter data.
    X-axis = dso_30d
    Y-axis = credit_used
    Bubble size = risk_score
    """

    required_cols = [
        "hospital_name",
        "dso_30d",
        "credit_used",
        "risk_score"
    ]
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns for cluster scatter: {missing}")

    plot_df = df.copy()

    plot_df = plot_df.replace([np.inf, -np.inf], 0)
    plot_df = plot_df.fillna(0)

    cluster_features = ["dso_30d", "credit_used"]
    cluster_data = plot_df[cluster_features].copy()

    if len(plot_df) == 0:
        return {
            "title": "Clustered Hospital Scatter Plot",
            "x_label": "DSO (30D)",
            "y_label": "Exposure (Credit Used)",
            "clusters": []
        }

    # Prevent invalid cluster count
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
        cluster_df = plot_df[
            plot_df["cluster"] == cluster_id
        ]

        cluster_points = cluster_df.apply(
            lambda row: {
                "hospital_name": str(row["hospital_name"]),
                "x": float(row["dso_30d"]),
                "y": float(row["credit_used"]),
                "risk_percent": float(row["risk_score"]),
                "bubble_size": float(row["risk_score"]),
                "cluster": int(row["cluster"]),
                "risk_tier": str(row["risk_tier"]) if "risk_tier" in plot_df.columns else None,
                "specialty": str(row["specialty"]) if "specialty" in plot_df.columns else None
            },
            axis=1
        ).tolist()

        center = kmeans.cluster_centers_[int(cluster_id)]

        clusters.append({
            "cluster_id": int(cluster_id),
            "center": {
                "x": float(center[0]),
                "y": float(center[1])
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