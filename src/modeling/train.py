import pandas as pd
import yaml
import pickle
import os
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from typing import Tuple

# Configure logging
log_file_path = "train.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)

def load_params(params_path: str) -> dict:
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logging.info(f"Loaded parameters from {params_path}")
        return params
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
        X = data.drop(columns=['Potability'])
        y = data['Potability']
        logging.info("Prepared features and target variable from training data")
        return X, y
    except Exception as e:
        logging.error(f"Error preparing data: {e}")
        raise

def train_model(X: pd.DataFrame, y: pd.Series, model_params: dict) -> Pipeline:
    try:
        clf = RandomForestClassifier(
            n_estimators=model_params["n_estimators"],
            max_depth=model_params["max_depth"],
            min_samples_split=model_params["min_samples_split"],
            min_samples_leaf=model_params["min_samples_leaf"],
            random_state=42 
        )

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', clf)
        ])

        pipeline.fit(X, y)
        logging.info("Model trained successfully")
        return pipeline
    except Exception as e:
        logging.error(f"Error training model: {e}")
        raise

def save_model(model: Pipeline, model_path: str) -> None:
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

        params = load_params(params_path)
        model_params = params.get("model_train", {})

        train_data = load_data(data_path)
        X_train, y_train = prepare_data(train_data)

        model = train_model(X_train, y_train, model_params)
        save_model(model, model_path)

    except Exception as e:
        logging.critical(f"An error occurred during model training: {e}")
        raise

if __name__ == "__main__":
    main()
