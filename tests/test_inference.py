from PIL import Image

from src.inference import predict_image


def test_predict_image_returns_valid_output():

    # Create a dummy RGB image
    image = Image.new(
        "RGB",
        (32, 32),
        color="white"
    )

    result = predict_image(
        image
    )

    assert "class" in result
    assert "confidence" in result

    assert isinstance(
        result["class"],
        str
    )

    assert isinstance(
        result["confidence"],
        float
    )

    assert 0.0 <= result["confidence"] <= 1.0