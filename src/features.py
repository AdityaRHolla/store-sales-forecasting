import pandas as pd
import numpy as np


def build_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts high-value calendar features from the date column based on EDA insights."""
    print("🗓️ Engineering calendar and payday features...")

    # Core calendar features
    df["year"] = df["date"].dt.year.astype(np.int16)
    df["month"] = df["date"].dt.month.astype(np.int8)
    df["day"] = df["date"].dt.day.astype(np.int8)
    df["day_of_week"] = df["date"].dt.dayofweek.astype(np.int8)

    # Weekend Flag (Sat=5, Sun=6)
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(np.int8)

    # Payday Flag (15th and Last day of the month)
    df["is_payday"] = ((df["day"] == 15) | (df["date"].dt.is_month_end)).astype(np.int8)

    # New Year's Eve (Dec 31) and New Year's Day (Jan 1) flags
    df["is_nye"] = ((df["month"] == 12) & (df["day"] == 31)).astype(np.int8)
    df["is_nyd"] = ((df["month"] == 1) & (df["day"] == 1)).astype(np.int8)

    return df


def build_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transforms raw oil prices to prevent long-term trend overfitting."""
    print("🛢️ Engineering advanced oil price trends...")

    # Group by date to extract unique daily oil prices
    daily_oil = df.groupby("date")["oil_price"].first().reset_index()

    # Calculate 7-day rolling average and daily price changes
    daily_oil["oil_roll_mean_7"] = (
        daily_oil["oil_price"].rolling(window=7, min_periods=1).mean()
    )
    daily_oil["oil_daily_diff"] = daily_oil["oil_price"].diff().fillna(0)

    # Merge these smoothed macroeconomic features back into the main dataset
    df = pd.merge(
        df,
        daily_oil[["date", "oil_roll_mean_7", "oil_daily_diff"]],
        on="date",
        how="left",
    )

    # Optimize column sizes
    df["oil_roll_mean_7"] = df["oil_roll_mean_7"].astype(np.float32)
    df["oil_daily_diff"] = df["oil_daily_diff"].astype(np.float32)

    return df


def build_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates leak-free historical lag features using a strict 16-day safety gap."""
    print("⏳ Engineering leak-free historical lags (16+ days out)...")

    # Ensure chronological sort order
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    # Store-Family grouping configuration
    sf_group = df.groupby(["store_nbr", "family"])["sales"]

    # 1. Safe Individual Lags (Starts at 16 to match the length of the test window)
    df["sales_lag_16"] = sf_group.shift(16).astype(np.float32)
    df["sales_lag_21"] = sf_group.shift(21).astype(np.float32)
    df["sales_lag_28"] = sf_group.shift(28).astype(np.float32)

    # 2. Safe Moving Averages
    # We shift by 16 first, then calculate a 7-day rolling window
    df["sales_roll_mean_16_7"] = (
        sf_group.shift(16)
        .transform(lambda x: x.rolling(7, min_periods=1).mean())
        .astype(np.float32)
    )

    df["sales_roll_std_16_7"] = (
        sf_group.shift(16)
        .transform(lambda x: x.rolling(7, min_periods=1).std())
        .fillna(0.0)
        .astype(np.float32)
    )

    # 3. Target Scales (Already safe since they use shift(16))
    df["family_mean_sales"] = (
        df.groupby("family")["sales"]
        .transform(lambda x: x.shift(16).rolling(30, min_periods=1).mean())
        .fillna(0.0)
        .astype(np.float32)
    )

    df["store_type_mean_sales"] = (
        df.groupby("type")["sales"]
        .transform(lambda x: x.shift(16).rolling(30, min_periods=1).mean())
        .fillna(0.0)
        .astype(np.float32)
    )

    # Clean up empty spaces at the beginning of the timeline
    fill_cols = [
        "sales_lag_16",
        "sales_lag_21",
        "sales_lag_28",
        "sales_roll_mean_16_7",
        "sales_roll_std_16_7",
        "family_mean_sales",
        "store_type_mean_sales",
    ]
    df[fill_cols] = df[fill_cols].fillna(0.0)

    print("✓ Leak-free features generated cleanly.")
    return df
