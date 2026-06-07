import pandas as pd
import numpy as np


def run_adaptive_error_analysis(
    df: pd.DataFrame, family_models: dict, family_scale_types: dict, features: list
) -> pd.DataFrame:
    """Isolates validation predictions across mixed-scale family models for precise error profiling."""
    print("🔎 Extracting validation slices for adaptive-scale error analysis...")

    # Isolate validation set exactly like our training loop (final 16 days of training data)
    val_cutoff = pd.to_datetime("2017-07-26")
    val_df = df[df["date"] >= val_cutoff].copy()

    # Pre-cast category columns to prevent type crashes
    cat_cols = ["city", "state", "type", "cluster", "store_family"]
    for col in cat_cols:
        val_df[col] = val_df[col].astype("category")

    # Create blank placeholder column for loop predictions in raw units
    val_df["predicted_sales"] = 0.0

    # Loop through each family and collect its validation predictions
    for family, model in family_models.items():
        fam_mask = val_df["family"] == family

        if fam_mask.sum() > 0:
            X_val_fam = val_df.loc[fam_mask, features]
            preds_trans = model.predict(X_val_fam)

            # Look up the specific scale type used during the training loop for this family
            scale_type = family_scale_types[family]

            # Invert predictions back to raw units
            if scale_type == "sqrt":
                final_preds = np.square(preds_trans)
            else:
                final_preds = np.expm1(preds_trans)

            # Assign raw predictions back to the matching family rows
            val_df.loc[fam_mask, "predicted_sales"] = np.clip(final_preds, 0, None)

    # Calculate the raw absolute difference error per row
    val_df["absolute_error"] = np.abs(val_df["sales"] - val_df["predicted_sales"])

    print("\n⚠️ --- Top 5 Product Families with Highest Average Error ---")
    family_errors = (
        val_df.groupby("family", observed=True)["absolute_error"]
        .mean()
        .sort_values(ascending=False)
        .head(5)
    )
    print(family_errors)

    print("\n🏢 --- Top 5 Stores with Highest Average Error ---")
    store_errors = (
        val_df.groupby("store_nbr", observed=True)["absolute_error"]
        .mean()
        .sort_values(ascending=False)
        .head(5)
    )
    print(store_errors)

    return val_df
