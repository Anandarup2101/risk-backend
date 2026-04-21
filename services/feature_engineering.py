def engineer_features(df):

    df["delay_ratio"] = df["delayed_payments"] / (df["total_payments"] + 1)
    df["dso_trend"] = df["dso_90d"] - df["dso_30d"]
    df["credit_stress"] = df["credit_used"] / (df["credit_limit"] + 1)
    df["ops_stress"] = df["order_drop_pct"] + df["billing_disputes"]

    return df