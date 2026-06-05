import pandas as pd
import numpy as np
from src import config


def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Iterate through all columns of a dataframe and modify the data type

    to reduce memory usage without losing information.
    """
    for col in df.columns:
        col_type = df[col].dtype

        if col_type is not object and not isinstance(col_type, pd.DatetimeTZDtype):
            c_min = df[col].min()
            c_max = df[col].max()

            # Handle Integers
            if str(col_type)[:3] == "int" or str(col_type)[:4] == "uint":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)

            # Handle Floats
            elif str(col_type)[:5] == "float":
                if (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    return df


def load_and_merge_data() -> pd.DataFrame:
    """Loads raw train data, fixes chronological gaps, optimizes memory, and merges metadata."""
    print("⏳ Loading train data...")
    train = pd.read_csv(config.TRAIN_PATH, parse_dates=["date"])

    # --- NEW: Chronological Gap Repair ---
    print("🗓️ Repairing missing timeline gaps...")
    # Create the complete theoretical grid of dates, stores, and families
    all_dates = pd.date_range(
        start=train["date"].min(), end=train["date"].max(), freq="D"
    )
    all_stores = train["store_nbr"].unique()
    all_families = train["family"].unique()

    # Generate a complete multi-index grid
    grid = (
        pd.MultiIndex.from_product(
            [all_dates, all_stores, all_families], names=["date", "store_nbr", "family"]
        )
        .to_frame()
        .reset_index(drop=True)
    )

    # Merge train back into the complete grid to catch missing days
    train = pd.merge(grid, train, on=["date", "store_nbr", "family"], how="left")

    # Missing days mean stores were closed: Sales and promotions are structurally 0
    train["sales"] = train["sales"].fillna(0.0)
    train["onpromotion"] = train["onpromotion"].fillna(0).astype(np.int16)
    # -------------------------------------

    print("⏳ Loading store metadata...")
    stores = pd.read_csv(config.STORES_PATH)

    print("🔗 Merging datasets...")
    # Use left merge onto our repaired train dataset
    df = pd.merge(train, stores, on="store_nbr", how="left")

    # Re-fill metadata text columns that became null during grid expansion
    df["city"] = df.groupby("store_nbr")["city"].ffill().bfill()
    df["state"] = df.groupby("store_nbr")["state"].ffill().bfill()
    df["type"] = df.groupby("store_nbr")["type"].ffill().bfill()
    df["cluster"] = df.groupby("store_nbr")["cluster"].ffill().bfill()

    # --- NEW: Safely fill the missing Kaggle IDs for added holiday records ---
    df["id"] = df["id"].fillna(-1).astype(np.int32)
    # -------------------------------------------------------------------------

    print("% Optimizing memory usage...")
    df = reduce_mem_usage(df)

    print(f"✓ Data loaded successfully! Final shape: {df.shape}")
    return df


def merge_oil_data(df: pd.DataFrame) -> pd.DataFrame:
    """Merges crude oil prices into the main dataframe and imputes missing weekend/holiday prices."""
    print("⏳ Ingesting and treating oil prices...")
    oil = pd.read_csv(config.OIL_PATH, parse_dates=["date"])

    # Rename column to be highly explicit
    oil = oil.rename(columns={"dcoilwtico": "oil_price"})

    # Merge oil into our unified main dataframe
    df = pd.merge(df, oil, on="date", how="left")

    # Crucial Time Series Imputation:
    # Forward-fill gaps (e.g., Saturday uses Friday's price).
    # Backward-fill handles any missing values at the very start of the timeline.
    df["oil_price"] = df["oil_price"].ffill().bfill()

    # Downcast the new column to save RAM
    df["oil_price"] = df["oil_price"].astype(np.float32)

    print("✓ Oil price feature integrated cleanly.")
    return df
