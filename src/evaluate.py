import pandas as pd
import numpy as np


def run_error_analysis(df: pd.DataFrame, model, features) -> pd.DataFrame:
    """Isolates validation predictions and calculates the biggest error groups."""
    print("🔎 Extracting validation slices for error analysis...")

    # Isolate validation set exactly like our training loop
    val_cutoff = pd.to_datetime("2017-07-26")
    val_df = df[df["date"] >= val_cutoff].copy()

    # Generate predictions in log space and invert them back to real sales units
    X_val = val_df[features]
    preds_log = model.predict(X_val)
    val_df["predicted_sales"] = np.expm1(preds_log)
    val_df["predicted_sales"] = np.clip(val_df["predicted_sales"], 0, None)

    # Calculate the absolute difference error per row
    val_df["absolute_error"] = np.abs(val_df["sales"] - val_df["predicted_sales"])

    print("\n⚠️ --- Top 5 Product Families with Highest Average Error ---")
    family_errors = (
        val_df.groupby("family")["absolute_error"]
        .mean()
        .sort_values(ascending=False)
        .head(5)
    )
    print(family_errors)

    print("\n🏢 --- Top 5 Stores with Highest Average Error ---")
    store_errors = (
        val_df.groupby("store_nbr")["absolute_error"]
        .mean()
        .sort_values(ascending=False)
        .head(5)
    )
    print(store_errors)

    return val_df
