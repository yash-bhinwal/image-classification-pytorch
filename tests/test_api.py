from io import BytesIO
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from src.api import app


client = TestClient(app)


def test_health_endpoint():

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json() == {
        "status": "healthy"
    }


@patch("src.api.predict_image")
def test_predict_endpoint(mock_predict_image):

    # --------------------------------------------------------
    # Fake model prediction
    # --------------------------------------------------------

    mock_predict_image.return_value = {
        "class": "cat",
        "confidence": 0.95
    }


    # --------------------------------------------------------
    # Create fake image
    # --------------------------------------------------------

    image = Image.new(
        "RGB",
        (32, 32),
        color="white"
    )

    image_bytes = BytesIO()

    image.save(
        image_bytes,
        format="JPEG"
    )

    image_bytes.seek(0)


    # --------------------------------------------------------
    # Call API
    # --------------------------------------------------------

    response = client.post(
        "/predict",
        files={
            "file": (
                "test.jpg",
                image_bytes,
                "image/jpeg"
            )
        }
    )


    # --------------------------------------------------------
    # Assertions
    # --------------------------------------------------------

    assert response.status_code == 200

    data = response.json()

    assert data["filename"] == "test.jpg"
    assert data["predicted_class"] == "cat"
    assert data["confidence"] == 0.95

    mock_predict_image.assert_called_once()