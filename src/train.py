import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from src import config


def run_per_family_training(df: pd.DataFrame, experiment_name: str = "per_family"):
    """Trains a completely independent LightGBM model for each product family

    to handle scale variances cleanly.
    """
    print("✂️ Preparing Train and Validation horizons...")

    # Define features (Drop 'family' from cat_cols since we split the data by family now)
    cat_cols = ["city", "state", "type", "cluster", "store_family"]
    num_cols = [
        "onpromotion",
        "oil_price",
        "oil_roll_mean_7",
        "oil_daily_diff",
        "year",
        "month",
        "day",
        "day_of_week",
        "is_weekend",
        "is_payday",
        "is_nye",
        "is_nyd",
        "sales_lag_16",
        "sales_lag_21",
        "sales_lag_28",
        "sales_roll_mean_16_7",
        "sales_roll_std_16_7",
        "family_mean_sales",
        "store_type_mean_sales",
        "promo_lag_1",
        "promo_lag_7",
        "promo_roll_mean_7",
        "type_family_mean_sales",
        "day_of_week_sin",
        "day_of_week_cos",
        "month_sin",
        "month_cos",
        "is_national_holiday",
        "holiday_lead_1",
        "holiday_lag_1",
        "promo_intensity_ratio",
        "promo_vs_sales_trend",
    ]
    features = cat_cols + num_cols

    for col in cat_cols:
        df[col] = df[col].astype("category")

    val_cutoff = pd.to_datetime("2017-07-26")
    # train_mask = df["date"] < val_cutoff
    val_mask = df["date"] >= val_cutoff

    # Dictionaries to store our independent models
    family_models = {}

    # Create empty arrays to collect our validation predictions
    all_val_preds_log = np.zeros(df[val_mask].shape[0])
    all_val_targets_log = np.zeros(df[val_mask].shape[0])

    # Extract all unique product families
    unique_families = df["family"].unique()
    print(
        f"🚀 Commencing isolated loop training across {len(unique_families)} product families..."
    )

    # Track row positions across the validation array chunks
    current_idx = 0

    for family in unique_families:
        # Isolate the data rows for just this specific family
        fam_df = df[df["family"] == family]

        X_train = fam_df.loc[fam_df["date"] < val_cutoff, features]
        y_train = fam_df.loc[fam_df["date"] < val_cutoff, config.TARGET_COL]

        X_val = fam_df.loc[fam_df["date"] >= val_cutoff, features]
        y_val = fam_df.loc[fam_df["date"] >= val_cutoff, config.TARGET_COL]

        # Log transform
        y_train_log = np.log1p(y_train)
        y_val_log = np.log1p(y_val)

        # Train a dedicated model for this family
        model = lgb.LGBMRegressor(
            n_estimators=120,
            learning_rate=0.08,
            random_state=42,
            n_jobs=-1,
            verbose=-1,  # Keep the console output clean during the loop
        )

        model.fit(
            X_train,
            y_train_log,
            eval_set=[(X_val, y_val_log)],
            callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)],
        )

        # Generate predictions for this family
        preds_log = model.predict(X_val)

        # Save model to our dictionary
        family_models[family] = model

        # Append predictions to our master validation arrays
        num_rows = len(preds_log)
        all_val_preds_log[current_idx : current_idx + num_rows] = preds_log
        all_val_targets_log[current_idx : current_idx + num_rows] = y_val_log.values
        current_idx += num_rows

    # Calculate global validation score across all families combined
    global_rmsle = root_mean_squared_error(all_val_targets_log, all_val_preds_log)
    print(f"\n🎉 Isolated Loop Validation RMSLE Score: {global_rmsle:.4f}")

    # Save to our ledger file
    ledger_path = os.path.join(config.BASE_DIR, "metrics_ledger.txt")
    with open(ledger_path, "a") as f:
        f.write(
            f"Run: {experiment_name} | Per-Family Loop | Validation RMSLE: {global_rmsle:.4f}\n"
        )

    return family_models, features


def generate_per_family_submission(
    df: pd.DataFrame, test_row_count: int, family_models, features
):
    """Generates a submission file by routing rows to their specialized family models."""
    print("🔮 Running multi-model loop inference on future test grid...")

    # Isolate rows that belong strictly to the future test timeline
    test_mask = df["date"] >= pd.to_datetime("2017-08-16")
    test_df = df[test_mask].copy()

    # Remove any injected holiday placeholder rows (IDs that we set to -1)
    test_df = test_df[test_df["id"] != -1].copy()

    # Ensure categories match our training setup
    cat_cols = ["city", "state", "type", "cluster", "store_family"]
    for col in cat_cols:
        test_df[col] = test_df[col].astype("category")

    # Create an empty array to collect final predictions
    test_df["sales"] = 0.0

    # Loop through each product family and use its dedicated model to predict
    for family, model in family_models.items():
        fam_mask = test_df["family"] == family

        if fam_mask.sum() > 0:
            X_test_fam = test_df.loc[fam_mask, features]

            # Predict in log-space, then transform back to raw sales units
            preds_log = model.predict(X_test_fam)
            final_preds = np.expm1(preds_log)
            final_preds = np.clip(final_preds, 0, None)

            # Assign predictions back to the matching family rows
            test_df.loc[fam_mask, "sales"] = final_preds

    print("📋 Re-aligning loop predictions with raw Kaggle test format...")
    # Load the original raw test file to use as our layout template
    raw_test = pd.read_csv(config.TEST_PATH)

    # Merge our loop predictions onto the raw template using the unique ID column
    submission = pd.merge(
        raw_test[["id"]], test_df[["id", "sales"]], on="id", how="left"
    )

    # Safety Check: Fill any missing rows with 0 if an ID skipped calculation
    submission["sales"] = submission["sales"].fillna(0.0)

    # Verify formatting bounds before exporting
    print(f"📊 Submission Row Count: {len(submission)} | Expected: {len(raw_test)}")

    sub_path = os.path.join(config.BASE_DIR, "data", "processed", "submission.csv")
    submission.to_csv(sub_path, index=False)
    print(f"🎉 Pristine multi-model submission file saved successfully to: {sub_path}")
