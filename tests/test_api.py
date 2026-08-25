from io import BytesIO

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


def test_predict_endpoint():

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

    assert "predicted_class" in data
    assert "confidence" in data

    assert isinstance(
        data["predicted_class"],
        str
    )

    assert 0.0 <= data["confidence"] <= 1.0