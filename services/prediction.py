import pickle
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "risk_model_bundle.pkl"

with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
threshold = bundle["threshold"]
features = bundle["features"]


def run_prediction(df):
    missing = set(features) - set(df.columns)
    if missing:
        raise ValueError(f"Missing features: {missing}")

    X = df[features]

    probs = model.predict_proba(X)[:, 1]

    df["risk_score"] = probs * 100
    df["risk_flag"] = (probs >= threshold).astype(int)

    df["risk_tier"] = pd.cut(
        df["risk_score"],
        bins=[0, 40, 55, 70, 100],
        labels=["Low", "Medium", "High", "Critical"],
        include_lowest=True
    ).astype(str)

    return df