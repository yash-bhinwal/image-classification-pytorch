import mlflow
import mlflow.pytorch
import torch

from src.model import CNN

mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
)

model = CNN()

model.load_state_dict(
    torch.load(
        "models/best_model.pth",
        weights_only=True
    )
)

model.eval()


with mlflow.start_run(
    run_name="register-best-cifar10-model"
):

    mlflow.pytorch.log_model(
        pytorch_model=model,
        name="model",
        registered_model_name="CIFAR10Classifier",
        serialization_format="pickle"
    )

    print(
        "Registered model: CIFAR10Classifier"
    )