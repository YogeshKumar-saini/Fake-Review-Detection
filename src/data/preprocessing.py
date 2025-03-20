import pandas as pd
import os
import yaml
import logging

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/preprocessing.log"),  # Log file
        logging.StreamHandler()  # Print logs to console
    ]
)

logging.info("Logging initialized successfully")

logging.info("Starting data preprocessing...")

# Load parameters from params.yaml
with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

RAW_PATH = params["data_paths"]["raw"]
PROCESSED_PATH = params["data_paths"]["processed"]
DROP_DUPLICATES = params["preprocessing"]["drop_duplicates"]
DROP_NA = params["preprocessing"]["drop_na"]
RATING_DTYPE = params["preprocessing"]["rating_dtype"]

def load_data(path):
    """Load raw data from CSV."""
    logging.info(f"Loading data from {path}...")
    return pd.read_csv(path)

def optimize_memory(df, column, dtype):
    """Optimize memory usage by converting specific column data types."""
    before = df.memory_usage(deep=True).sum()
    df[column] = df[column].astype(dtype)
    after = df.memory_usage(deep=True).sum()
    saved_memory = before - after
    logging.info(f"Converted '{column}' to {dtype}, saved {saved_memory} bytes.")
    return df

def clean_data(df, drop_duplicates, drop_na):
    """Remove missing values and duplicates if enabled in params."""
    if drop_na:
        missing_count = df.isnull().sum().sum()
        df.dropna(inplace=True)
        logging.info(f"Removed {missing_count} missing values.")

    if drop_duplicates:
        before = df.shape[0]
        df.drop_duplicates(inplace=True)
        after = df.shape[0]
        logging.info(f"Removed {before - after} duplicate rows.")

    return df

def save_data(df, path):
    """Save processed data to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logging.info(f"Processed data saved at {path}")

if __name__ == "__main__":
    logging.info("Starting preprocessing pipeline...")

    df = load_data(RAW_PATH)
    df = optimize_memory(df, "rating", RATING_DTYPE)
    df = clean_data(df, DROP_DUPLICATES, DROP_NA)
    save_data(df, PROCESSED_PATH)

    logging.info("Preprocessing pipeline completed successfully.")
