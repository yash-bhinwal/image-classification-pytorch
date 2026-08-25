import sys

from PIL import Image

from src.inference import predict_image

if len(sys.argv) != 2:

    print(
        "Usage: python src/predict.py "
        "path/to/image.jpg"
    )

    sys.exit(1)


image_path = sys.argv[1]

image = Image.open(
    image_path
)

result = predict_image(
    image
)


print(
    f"Predicted class: "
    f"{result['class']}"
)

print(
    f"Confidence: "
    f"{result['confidence'] * 100:.2f}%"
)