import os

import mlflow
import mlflow.pytorch
import torch


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5001"
)

MODEL_URI = "models:/CIFAR10Classifier@champion"

EXPORT_PATH = "release/champion_model.pth"


# ------------------------------------------------------------
# Connect to MLflow
# ------------------------------------------------------------

mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)


# ------------------------------------------------------------
# Load approved champion
# ------------------------------------------------------------

print(
    f"Loading champion model from: "
    f"{MODEL_URI}"
)

model = mlflow.pytorch.load_model(
    MODEL_URI
)

model.eval()


# ------------------------------------------------------------
# Export only the model weights
# ------------------------------------------------------------

os.makedirs(
    "release",
    exist_ok=True
)

torch.save(
    model.state_dict(),
    EXPORT_PATH
)


print(
    f"Champion exported to: "
    f"{EXPORT_PATH}"
)