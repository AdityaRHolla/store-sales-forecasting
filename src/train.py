import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from src import config


def run_adaptive_scale_training(
    df: pd.DataFrame, experiment_name: str = "adaptive_scaling"
):
    """Trains independent models with specialized target scaling per family to beat the log bottleneck."""
    print("✂️ Preparing Train and Validation horizons...")

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
        "sales_lag_21",
        "sales_lag_28",
        "sales_roll_mean_21_7",
        "sales_roll_std_21_7",
        "family_mean_sales",
        "store_type_mean_sales",
        "type_family_mean_sales",
        "day_of_week_sin",
        "day_of_week_cos",
        "month_sin",
        "month_cos",
        "is_national_holiday",
        "holiday_lead_1",
        "holiday_lag_1",
        "promo_intensity_ratio",
    ]
    features = cat_cols + num_cols

    for col in cat_cols:
        df[col] = df[col].astype("category")

    val_cutoff = pd.to_datetime("2017-07-26")
    val_mask = df["date"] >= val_cutoff

    family_models = {}
    family_scale_types = {}  # Tracks which transformation each family used

    # Arrays to collect our final global predictions in true raw units
    val_row_count = df[val_mask].shape[0]
    all_val_preds_raw = np.zeros(val_row_count)
    all_val_targets_raw = np.zeros(val_row_count)

    unique_families = df["family"].unique()
    print(
        f"🚀 Commencing adaptive scale loop training across {len(unique_families)} families..."
    )

    current_idx = 0

    for family in unique_families:
        fam_df = df[df["family"] == family]

        X_train = fam_df.loc[fam_df["date"] < val_cutoff, features]
        y_train = fam_df.loc[fam_df["date"] < val_cutoff, config.TARGET_COL]

        X_val = fam_df.loc[fam_df["date"] >= val_cutoff, features]
        y_val = fam_df.loc[fam_df["date"] >= val_cutoff, config.TARGET_COL]

        # --- ADAPTIVE TARGET SCALING LOGIC ---
        # If the category peaks over 5,000 units, use Square Root to protect high-end variance
        if y_train.max() > 5000:
            scale_type = "sqrt"
            y_train_trans = np.sqrt(y_train)
            y_val_trans = np.sqrt(y_val)
        else:
            scale_type = "log1p"
            y_train_trans = np.log1p(y_train)
            y_val_trans = np.log1p(y_val)

        family_scale_types[family] = scale_type
        # --------------------------------------

        model = lgb.LGBMRegressor(
            n_estimators=120, learning_rate=0.08, random_state=42, n_jobs=-1, verbose=-1
        )
        model.fit(
            X_train,
            y_train_trans,
            eval_set=[(X_val, y_val_trans)],
            callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)],
        )

        preds_trans = model.predict(X_val)

        # Invert predictions back to raw sales units based on the scale used
        if scale_type == "sqrt":
            preds_raw = np.square(preds_trans)
        else:
            preds_raw = np.expm1(preds_trans)

        preds_raw = np.clip(preds_raw, 0, None)

        # Save to master raw arrays
        num_rows = len(preds_raw)
        all_val_preds_raw[current_idx : current_idx + num_rows] = preds_raw
        all_val_targets_raw[current_idx : current_idx + num_rows] = y_val.values

        family_models[family] = model
        current_idx += num_rows

    # Calculate global competitive score in log space (RMSLE) for honest tracking
    global_rmsle = root_mean_squared_error(
        np.log1p(all_val_targets_raw), np.log1p(all_val_preds_raw)
    )
    print(f"\n🎉 Adaptive Scale Validation RMSLE Score: {global_rmsle:.4f}")

    ledger_path = os.path.join(config.BASE_DIR, "metrics_ledger.txt")
    with open(ledger_path, "a") as f:
        f.write(
            f"Run: {experiment_name} | Adaptive Scaling Loop | Validation RMSLE: {global_rmsle:.4f}\n"
        )

    return family_models, family_scale_types, features


def generate_adaptive_scale_submission(
    df: pd.DataFrame, test_row_count: int, family_models, family_scale_types, features
):
    """Generates a submission file by handling mixed sqrt and log1p scale inversions per family."""
    print("🔮 Running adaptive-scale inference loop on future test grid...")

    test_mask = df["date"] >= pd.to_datetime("2017-08-16")
    test_df = df[test_mask].copy()
    test_df = test_df[test_df["id"] != -1].copy()

    cat_cols = ["city", "state", "type", "cluster", "store_family"]
    for col in cat_cols:
        test_df[col] = test_df[col].astype("category")

    test_df["sales"] = 0.0

    for family, model in family_models.items():
        fam_mask = test_df["family"] == family

        if fam_mask.sum() > 0:
            X_test_fam = test_df.loc[fam_mask, features]
            preds_trans = model.predict(X_test_fam)

            # Look up the specific scale type used for this product family
            scale_type = family_scale_types[family]

            if scale_type == "sqrt":
                final_preds = np.square(preds_trans)
            else:
                final_preds = np.expm1(preds_trans)

            test_df.loc[fam_mask, "sales"] = np.clip(final_preds, 0, None)

    print("📋 Re-aligning adaptive predictions with raw Kaggle test format...")
    raw_test = pd.read_csv(config.TEST_PATH)
    submission = pd.merge(
        raw_test[["id"]], test_df[["id", "sales"]], on="id", how="left"
    )
    submission["sales"] = submission["sales"].fillna(0.0)

    print(f"📊 Submission Row Count: {len(submission)} | Expected: {len(raw_test)}")

    sub_path = os.path.join(config.BASE_DIR, "data", "processed", "submission.csv")
    submission.to_csv(sub_path, index=False)
    print(
        f"🎉 Pristine adaptive-scale submission file saved successfully to: {sub_path}"
    )
