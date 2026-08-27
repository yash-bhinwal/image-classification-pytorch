import os

import torch

from PIL import Image
from torchvision import transforms

from src.model import CNN


# ------------------------------------------------------------
# Model configuration
# ------------------------------------------------------------

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "release/champion_model.pth"
)


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
# Lazy model loading
# ------------------------------------------------------------

_model = None


def get_model():

    global _model

    if _model is None:

        model = CNN()

        model.load_state_dict(
            torch.load(
                MODEL_PATH,
                map_location="cpu",
                weights_only=True
            )
        )

        model.eval()

        _model = model

    return _model


# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

def predict_image(
    image: Image.Image
):

    model = get_model()

    image = image.convert("RGB")

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    with torch.no_grad():

        outputs = model(image_tensor)

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