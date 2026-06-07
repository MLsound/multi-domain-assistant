import os
from pathlib import Path

# Define a local, writable directory for MLflow tracking data
MLFLOW_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "mlflow_data"
MLFLOW_DATA_DIR.mkdir(parents=True, exist_ok=True)

tracking_uri = f"file://{MLFLOW_DATA_DIR}"
experiment_name = "knowledge-assistant"

# Set the required environment variables BEFORE mlflow is imported
os.environ["MLFLOW_TRACKING_DIR"] = str(MLFLOW_DATA_DIR)
os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name

# Explicitly initialize MLflow so decorators don't fail or fallback to experiment '0'
import mlflow
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(experiment_name)

# Suppress the known non-fatal 'MlflowV3SpanProcessor' attribute warning
import logging
logging.getLogger("mlflow.entities.span").setLevel(logging.ERROR)
