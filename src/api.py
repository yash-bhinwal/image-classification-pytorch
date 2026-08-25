from io import BytesIO

from fastapi import FastAPI, File, UploadFile
from PIL import Image


app = FastAPI(
    title="CIFAR-10 Image Classifier",
    description="PyTorch CIFAR-10 classifier served through FastAPI",
    version="1.0.0"
)


def predict_image(image):
    from src.inference import predict_image as run_prediction

    return run_prediction(image)


@app.get("/")
def root():
    return {
        "message": "CIFAR-10 classifier API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image_bytes = await file.read()

    image = Image.open(
        BytesIO(image_bytes)
    )

    result = predict_image(
        image
    )

    return {
        "filename": file.filename,
        "predicted_class": result["class"],
        "confidence": result["confidence"]
    }