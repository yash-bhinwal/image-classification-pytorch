import mlflow
import mlflow.pytorch
import torch

from PIL import Image
from torchvision import transforms


# ------------------------------------------------------------
# MLflow connection
# ------------------------------------------------------------

mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
)


# ------------------------------------------------------------
# Model Registry
# ------------------------------------------------------------

MODEL_URI = "models:/CIFAR10Classifier@champion"

model = mlflow.pytorch.load_model(
    MODEL_URI
)

model.eval()


# ------------------------------------------------------------
# CIFAR-10 classes
# ------------------------------------------------------------

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


# ------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])


# ------------------------------------------------------------
# Prediction function
# ------------------------------------------------------------

def predict_image(image: Image.Image):

    image = image.convert("RGB")

    image_tensor = transform(
        image
    )

    image_tensor = image_tensor.unsqueeze(0)

    with torch.no_grad():

        outputs = model(
            image_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            1
        )

    predicted_class = CLASS_NAMES[
        predicted.item()
    ]

    confidence_value = confidence.item()

    return {
        "class": predicted_class,
        "confidence": confidence_value
    }