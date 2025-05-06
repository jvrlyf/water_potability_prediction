import numpy as np
import pandas as pd
import pickle
import json
import os
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Tuple, Dict

from dvclive import Live
import yaml


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
        params=yaml.safe_load(open("params.yaml","r"))
        test_size= params['dataset']['test_size']
        n_estimators= params['model_train']['n_estimators']
        max_depth= params['model_train']['max_depth']
        min_samples_split= params['model_train']['min_samples_split']
        min_samples_leaf= params['model_train']['min_samples_leaf']

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        pre = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1score = f1_score(y_test, y_pred)

        with Live(save_dvc_exp=True) as live:
            live.log_metric("accuracy",acc)
            live.log_metric("precision",pre)
            live.log_metric("recall",recall)
            live.log_metric("f1_score",f1score)

            live.log_metric("test_size",test_size)
            live.log_metric("n_estimators",n_estimators)
            live.log_metric("max_depth",max_depth)
            live.log_metric("min_samples_split",min_samples_split)
            live.log_metric("min_samples_leaf",min_samples_leaf)


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
