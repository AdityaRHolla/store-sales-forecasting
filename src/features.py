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
    """Creates historical lag features for sales.

    Must sort chronologically first to ensure correct historical shifts!
    """
    print("⏳ Engineering structural historical lags...")

    # Crucial step: Sort the entire dataframe by date so lags pull consecutive records
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    # Group by store and product family so lags don't blend different store sales together
    grouped = df.groupby(["store_nbr", "family"])["sales"]

    # Create yesterday's sales lag, 7 days ago lag (weekly match), and 14 days ago lag
    df["sales_lag_1"] = grouped.shift(1).astype(np.float32)
    df["sales_lag_7"] = grouped.shift(7).astype(np.float32)
    df["sales_lag_14"] = grouped.shift(14).astype(np.float32)

    # 7-day Moving Average of historical sales to capture the rolling local momentum
    # We shift by 1 first so we don't accidentally include today's true sales target in the feature
    df["sales_roll_mean_7"] = (
        grouped.shift(1)
        .transform(lambda x: x.rolling(7, min_periods=1).mean())
        .astype(np.float32)
    )

    # For rows at the absolute beginning of 2013, lags won't exist. Fill them with a baseline 0.
    lag_cols = ["sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_roll_mean_7"]
    df[lag_cols] = df[lag_cols].fillna(0.0)

    print("✓ Time-series memory features generated cleanly.")
    return df
