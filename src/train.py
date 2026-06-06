import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from src import config


def run_baseline_training(df: pd.DataFrame):
    """Splits data chronologically, prepares features, and trains a baseline LightGBM model."""
    print("✂️ Splitting data into Train and Validation sets...")

    # Define categorical and numerical features for the baseline
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
    ]

    features = cat_cols + num_cols

    # Convert string columns to categorical type so LightGBM knows how to handle them
    for col in cat_cols:
        df[col] = df[col].astype("category")

    # Chronological Cutoff (Keep final 16 days of train data for validation)
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

    # Kaggle competition metrics use RMSLE (Root Mean Squared Logarithmic Error)
    # The simplest way to optimize for RMSLE is to train directly on log1p(y)
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)

    print(f"📊 Train records: {len(X_train)} | Validation records: {len(X_val)}")
    print("🚀 Training baseline LightGBM model...")

    # Initialize a fast baseline model
    model = lgb.LGBMRegressor(
        n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1
    )

    model.fit(
        X_train,
        y_train_log,
        eval_set=[(X_val, y_val_log)],
        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)],
    )

    # Predict and transform predictions back from log space
    preds_log = model.predict(X_val)
    preds = np.expm1(preds_log)
    preds = np.clip(preds, 0, None)  # Sales cannot be negative

    # Calculate final baseline metric
    # Using modern scikit-learn root_mean_squared_error on log values equals RMSLE
    rmsle = root_mean_squared_error(y_val_log, preds_log)
    print(f"\n🎉 Baseline Validation RMSLE Score: {rmsle:.4f}")

    return model, features
