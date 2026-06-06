import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Raw File Paths
TRAIN_PATH = os.path.join(RAW_DATA_DIR, "train.csv")
TEST_PATH = os.path.join(RAW_DATA_DIR, "test.csv")
STORES_PATH = os.path.join(RAW_DATA_DIR, "stores.csv")
HOLIDAYS_PATH = os.path.join(RAW_DATA_DIR, "holidays_events.csv")
OIL_PATH = os.path.join(RAW_DATA_DIR, "oil.csv")
TRANSACTIONS_PATH = os.path.join(RAW_DATA_DIR, "transactions.csv")

# Validation Settings
TARGET_COL = "sales"
