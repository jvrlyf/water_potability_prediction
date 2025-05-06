import pandas as pd
import numpy as np
import os
import logging

# Setup logging
log_file_path = "features.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)

def load_data(filepath: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath)
        logging.info(f"Loaded data from {filepath}, shape={df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {filepath}: {e}")
        raise

def fill_missing_with_mean(df: pd.DataFrame) -> pd.DataFrame:
    try:
        for column in df.columns:
            if df[column].isnull().any():
                mean_value = df[column].mean()
                df[column].fillna(mean_value, inplace=True)
                logging.info(f"Filled missing values in '{column}' with mean: {mean_value:.4f}")
        return df
    except Exception as e:
        logging.error(f"Error filling missing values with mean: {e}")
        raise

def save_data(df: pd.DataFrame, filepath: str) -> None:
    try:
        df.to_csv(filepath, index=False)
        logging.info(f"Saved processed data to {filepath}, shape={df.shape}")
    except Exception as e:
        logging.error(f"Error saving data to {filepath}: {e}")
        raise

def main():
    try:
        raw_data_path = "./data/interim/"
        processed_data_path = "./data/processed/"

        train_data = load_data(os.path.join(raw_data_path, "train.csv"))
        test_data = load_data(os.path.join(raw_data_path, "test.csv"))

        train_processed_data = fill_missing_with_mean(train_data)
        test_processed_data = fill_missing_with_mean(test_data)

        os.makedirs(processed_data_path, exist_ok=True)
        logging.info(f"Ensured processed data directory exists: {processed_data_path}")

        save_data(train_processed_data, os.path.join(processed_data_path, "train_processed_mean.csv"))
        save_data(test_processed_data, os.path.join(processed_data_path, "test_processed_mean.csv"))

    except Exception as e:
        logging.critical(f"An error occurred during data processing: {e}")
        raise

if __name__ == "__main__":
    main()
