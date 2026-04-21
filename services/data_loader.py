import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "inputs"


def load_data():
    payment = pd.read_excel(INPUT_DIR / "payment_behavior.xlsx")
    financial = pd.read_excel(INPUT_DIR / "financial_signal.xlsx")
    operational = pd.read_excel(INPUT_DIR / "operational_change.xlsx")
    metadata = pd.read_excel(INPUT_DIR / "target.xlsx")

    df = (
        payment.merge(financial, on="hospital_name")
               .merge(operational, on="hospital_name")
               .merge(metadata, on="hospital_name")
    )

    if "target" in df.columns:
        df = df.drop("target", axis=1)

    print("DF Columns:", df.columns.to_list())
    return df