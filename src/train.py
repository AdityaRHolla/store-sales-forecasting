import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from src import config


def run_baseline_training(df: pd.DataFrame, experiment_name: str = "lgb_baseline"):
    """Splits data, trains LightGBM, and saves all metrics automatically to MLflow."""
    print("✂️ Splitting data into Train and Validation sets...")

    cat_cols = ["family", "city", "state", "type", "cluster"]
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
        "sales_lag_1",
        "sales_lag_7",
        "sales_lag_14",
        "sales_roll_mean_7",
    ]
    features = cat_cols + num_cols

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

    # --- NEW: Initialize MLflow Experiment Dashboard ---
    mlflow.set_experiment("Favorita_Store_Sales")

    with mlflow.start_run(run_name=experiment_name):
        print(f"📊 Tracking execution under run: '{experiment_name}'")
        print("🚀 Training LightGBM model...")

        model = lgb.LGBMRegressor(
            n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1
        )

        model.fit(
            X_train,
            y_train_log,
            eval_set=[(X_val, y_val_log)],
            callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)],
        )

        preds_log = model.predict(X_val)
        rmsle = root_mean_squared_error(y_val_log, preds_log)

        # Log parameters and metrics explicitly into our dashboard backend
        mlflow.log_param("num_features", len(features))
        mlflow.log_param("model_type", "LightGBM")
        mlflow.log_metric("val_rmsle", rmsle)

        print(f"\n🎉 Validation RMSLE Score: {rmsle:.4f}")

    return model, features
