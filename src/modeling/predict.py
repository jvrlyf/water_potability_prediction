import numpy as np
import pandas as pd
import pickle
import json
import os
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Tuple, Dict

# Configure logging
log_file_path = "predict.log"
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
        logging.info(f"Loaded test data from {filepath}, shape={df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {filepath}: {e}")
        raise

def prepare_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    try:
        X = data.drop(columns=['Potability'], axis=1)
        y = data['Potability']
        logging.info("Prepared features and target from test data")
        return X, y
    except Exception as e:
        logging.error(f"Error preparing data: {e}")
        raise

def load_model(filepath: str):
    try:
        with open(filepath, "rb") as file:
            model = pickle.load(file)
        logging.info(f"Loaded model from {filepath}")
        return model
    except Exception as e:
        logging.error(f"Error loading model from {filepath}: {e}")
        raise

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    try:
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        pre = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1score = f1_score(y_test, y_pred)

        metrics_dict = {
            "accuracy": acc,
            "precision": pre,
            "recall": recall,
            "f1_score": f1score
        }

        logging.info(f"Evaluation metrics: {metrics_dict}")
        return metrics_dict
    except Exception as e:
        logging.error(f"Error evaluating model: {e}")
        raise

def save_metrics(metrics: Dict[str, float], metrics_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logging.info(f"Saved evaluation metrics to {metrics_path}")
    except Exception as e:
        logging.error(f"Error saving metrics to {metrics_path}: {e}")
        raise

def main():
    try:
        test_data_path = "./data/processed/test_processed_mean.csv"
        model_path = "models/model.pkl"
        metrics_path = "reports/metrics.json"

        test_data = load_data(test_data_path)
        X_test, y_test = prepare_data(test_data)
        model = load_model(model_path)
        metrics = evaluate_model(model, X_test, y_test)
        save_metrics(metrics, metrics_path)

    except Exception as e:
        logging.critical(f"An error occurred during model evaluation: {e}")
        raise

if __name__ == "__main__":
    main()
