import mlflow
from mlflow import MlflowClient


mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
)

client = MlflowClient()

client.set_registered_model_alias(
    name="CIFAR10Classifier",
    alias="champion",
    version="2"
)

print(
    "Alias 'champion' now points to "
    "CIFAR10Classifier version 2."
)

