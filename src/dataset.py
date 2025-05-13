import pandas as pd
import numpy as np
import os
import logging
from sklearn.model_selection import train_test_split
import yaml

import mlflow
mlflow.set_experiment("water_potability_prediction")


# Configure logging to both file and terminal
log_file_path = "dataset.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)

def load_params(filepath: str) -> float:
    try:
        with open(filepath, "r") as file:
            params = yaml.safe_load(file)
        logging.info(f"Loaded parameters: {params}")

        test_size = params["dataset"]["test_size"]
        logging.info(f"Using test_size={test_size}")

        with mlflow.start_run():
            mlflow.log_param("test_size", test_size)

        return test_size
    except KeyError as e:
        logging.error(f"Missing key in params.yaml: {e}")
        raise
    except Exception as e:
        logging.error(f"Failed to load parameters from {filepath}: {e}")
        raise


def load_data(filepath: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath)
        logging.info(f"Loaded data from {filepath}, shape={df.shape}")
        return df
    except Exception as e:
        logging.error(f"Failed to load data from {filepath}: {e}")
        raise

def split_data(data: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        train, test = train_test_split(data, test_size=test_size, random_state=42)
        logging.info(f"Split data into train ({train.shape}) and test ({test.shape})")
        return train, test
    except Exception as e:
        logging.error(f"Failed to split data: {e}")
        raise

def save_data(df: pd.DataFrame, filepath: str) -> None:
    try:
        df.to_csv(filepath, index=False)
        logging.info(f"Saved data to {filepath}, shape={df.shape}")
    except Exception as e:
        logging.error(f"Failed to save data to {filepath}: {e}")
        raise

def main():
    data_filepath = r"/Volumes/JV-R/Files/MLops Project/Water potability prediction/water_potability.csv"
    params_filepath = "params.yaml"
    raw_data_path = os.path.join("data", "interim")

    try:
        data = load_data(data_filepath)
        test_size = load_params(params_filepath)
        train_data, test_data = split_data(data, test_size)

        os.makedirs(raw_data_path, exist_ok=True)
        logging.info(f"Directory created or already exists: {raw_data_path}")

        save_data(train_data, os.path.join(raw_data_path, "train.csv"))
        save_data(test_data, os.path.join(raw_data_path, "test.csv"))

    except Exception as e:
        logging.critical(f"An error occurred during execution: {e}")
        raise

if __name__ == "__main__":
    main()
