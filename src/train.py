import os
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error
from src import config


def run_ensemble_training(df: pd.DataFrame, experiment_name: str = "ensemble_run"):
    """Trains both LightGBM and XGBoost models, averages their predictions,

    and logs the blended score to the local ledger file.
    """
    print("✂️ Splitting data into Train and Validation sets...")

    cat_cols = ["family", "city", "state", "type", "cluster", "store_family"]
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
    ]
    features = cat_cols + num_cols

    # Prepare category types cleanly for both tree backends
    for col in cat_cols:
        df[col] = df[col].astype("category")

    val_cutoff = pd.to_datetime("2017-07-26")
    train_mask = df["date"] < val_cutoff
    val_mask = df["date"] >= val_cutoff

    X_train, y_train = (
        df.loc[train_mask, features],
        df.loc[train_mask, config.TARGET_COL],
    )
    X_val, y_val = (
        df.loc[val_mask, features],
        df.loc[val_mask, config.TARGET_COL],
    )

    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)

    # 1. Train LightGBM
    print("🚀 Training LightGBM model component...")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=150, learning_rate=0.08, random_state=42, n_jobs=-1
    )
    lgb_model.fit(
        X_train,
        y_train_log,
        eval_set=[(X_val, y_val_log)],
        callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)],
    )
    lgb_preds = lgb_model.predict(X_val)

    # 2. Train XGBoost (Using experimental high-performance category handling)
    print("🚀 Training XGBoost model component...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=150,
        learning_rate=0.08,
        random_state=42,
        n_jobs=-1,
        enable_categorical=True,  # Tells XGBoost to read category columns natively
        early_stopping_rounds=15,
    )
    xgb_model.fit(
        X_train,
        y_train_log,
        eval_set=[(X_val, y_val_log)],
        verbose=False,
    )
    xgb_preds = xgb_model.predict(X_val)

    # 3. Blended Ensemble (Simple 50/50 average in log space)
    print("⚖️ Blending model predictions into unified ensemble...")
    ensemble_preds = (lgb_preds * 0.5) + (xgb_preds * 0.5)

    # Calculate metrics
    lgb_rmsle = root_mean_squared_error(y_val_log, lgb_preds)
    xgb_rmsle = root_mean_squared_error(y_val_log, xgb_preds)
    ensemble_rmsle = root_mean_squared_error(y_val_log, ensemble_preds)

    print(f"\n💡 LightGBM Component RMSLE: {lgb_rmsle:.4f}")
    print(f"💡 XGBoost Component RMSLE: {xgb_rmsle:.4f}")
    print(f"🎉 Final Blended Ensemble RMSLE Score: {ensemble_rmsle:.4f}")

    # --- Write Results to Local Ledger ---
    ledger_path = os.path.join(config.BASE_DIR, "metrics_ledger.txt")
    with open(ledger_path, "a") as f:
        f.write(
            f"Run: {experiment_name} | LGB: {lgb_rmsle:.4f} | XGB: {xgb_rmsle:.4f} | Ensemble Blend: {ensemble_rmsle:.4f}\n"
        )
    print(f"✓ Ensemble scores saved securely to ledger: {ledger_path}")

    return lgb_model, xgb_model, features


def generate_kaggle_submission(
    df: pd.DataFrame, test_row_count: int, xgb_model, features
):
    """Generates a submission file aligned with Kaggle's row order guidelines."""
    print("🔮 Running inference on future test grid...")

    # Isolate rows that belong strictly to the future test timeline
    # The true test data starts exactly on August 16, 2017
    test_mask = df["date"] >= pd.to_datetime("2017-08-16")
    test_df = df[test_mask].copy()

    # Remove any injected holiday placeholder rows (IDs that we set to -1)
    test_df = test_df[test_df["id"] != -1].copy()

    # Ensure categories match our training setup
    cat_cols = ["family", "city", "state", "type", "cluster", "store_family"]
    for col in cat_cols:
        test_df[col] = test_df[col].astype("category")

    X_test = test_df[features]

    # Predict and transform out of log space
    preds_log = xgb_model.predict(X_test)
    final_preds = np.expm1(preds_log)
    final_preds = np.clip(final_preds, 0, None)

    # Attach predictions back onto our sliced dataframe
    test_df["sales"] = final_preds

    print("📋 Re-aligning predictions with raw Kaggle test format...")
    # Load the original raw test file to use as our layout template
    raw_test = pd.read_csv(config.TEST_PATH)

    # Merge our predictions onto the raw template using the unique ID column
    submission = pd.merge(
        raw_test[["id"]], test_df[["id", "sales"]], on="id", how="left"
    )

    # Safety Check: Fill any missing rows with 0 if an ID skipped calculation
    submission["sales"] = submission["sales"].fillna(0.0)

    # Verify formatting bounds before exporting
    print(f"📊 Submission Row Count: {len(submission)}")
    print(f"📊 Expected Row Count: {len(raw_test)}")

    sub_path = os.path.join(config.BASE_DIR, "data", "processed", "submission.csv")
    submission.to_csv(sub_path, index=False)
    print(f"🎉 Pristine submission file saved successfully to: {sub_path}")
