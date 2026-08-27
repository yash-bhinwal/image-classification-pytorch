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

    mock_predict_image.return_value = {
        "class": "cat",
        "confidence": 0.95
    }

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

    assert response.status_code == 200

    data = response.json()

    assert data["filename"] == "test.jpg"
    assert data["predicted_class"] == "cat"
    assert data["confidence"] == 0.95

    assert "latency_ms" in data

    mock_predict_image.assert_called_once()


def test_predict_rejects_invalid_image():

    response = client.post(
        "/predict",
        files={
            "file": (
                "not-an-image.txt",
                b"this is definitely not an image",
                "text/plain"
            )
        }
    )

    assert response.status_code == 400

    assert response.json() == {
        "detail": "Uploaded file is not a valid image"
    }