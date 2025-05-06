import pandas as pd
import yaml
import pickle
import os
import logging
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple

# Setup logging to both console and file
log_file_path = "train.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)

def load_params(params_path: str) -> int:
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        n_estimators = params["model_train"]["n_estimators"]
        logging.info(f"Loaded n_estimators={n_estimators} from {params_path}")
        return n_estimators
    except KeyError as e:
        logging.error(f"Missing key in params.yaml: {e}")
        raise
    except Exception as e:
        logging.error(f"Error loading parameters from {params_path}: {e}")
        raise

def load_data(data_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        logging.info(f"Loaded training data from {data_path}, shape={df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {data_path}: {e}")
        raise

def prepare_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    try:
        X = data.drop(columns=['Potability'], axis=1)
        y = data['Potability']
        logging.info("Prepared features and target variable from training data")
        return X, y
    except Exception as e:
        logging.error(f"Error preparing data: {e}")
        raise

def train_model(X: pd.DataFrame, y: pd.Series, n_estimators: int) -> RandomForestClassifier:
    try:
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        clf.fit(X, y)
        logging.info(f"Model trained successfully with {n_estimators} estimators")
        return clf
    except Exception as e:
        logging.error(f"Error training model: {e}")
        raise

def save_model(model: RandomForestClassifier, model_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as file:
            pickle.dump(model, file)
        logging.info(f"Model saved to {model_path}")
    except Exception as e:
        logging.error(f"Error saving model to {model_path}: {e}")
        raise

def main():
    try:
        params_path = "params.yaml"
        data_path = "./data/processed/train_processed_mean.csv"
        model_path = "models/model.pkl"

        n_estimators = load_params(params_path)
        train_data = load_data(data_path)
        X_train, y_train = prepare_data(train_data)

        model = train_model(X_train, y_train, n_estimators)
        save_model(model, model_path)

    except Exception as e:
        logging.critical(f"An error occurred during model training: {e}")
        raise

if __name__ == "__main__":
    main()
