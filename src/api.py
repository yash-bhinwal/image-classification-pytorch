import logging
import time

from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError


# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    )
)

logger = logging.getLogger("cifar10-api")


# ------------------------------------------------------------
# Simple in-memory metrics
# ------------------------------------------------------------

metrics = {
    "requests_total": 0,
    "predictions_successful": 0,
    "predictions_failed": 0,
    "invalid_uploads": 0,
    "total_latency_ms": 0.0,
    "total_confidence": 0.0,
    "prediction_counts": {
        "airplane": 0,
        "automobile": 0,
        "bird": 0,
        "cat": 0,
        "deer": 0,
        "dog": 0,
        "frog": 0,
        "horse": 0,
        "ship": 0,
        "truck": 0
    }
}

BASELINE_DISTRIBUTION = {
    "airplane": 10.0,
    "automobile": 10.0,
    "bird": 10.0,
    "cat": 10.0,
    "deer": 10.0,
    "dog": 10.0,
    "frog": 10.0,
    "horse": 10.0,
    "ship": 10.0,
    "truck": 10.0
}


DRIFT_THRESHOLD = 20.0

# ------------------------------------------------------------
# FastAPI
# ------------------------------------------------------------

app = FastAPI(
    title="CIFAR-10 Image Classifier",
    description="PyTorch CIFAR-10 classifier served through FastAPI",
    version="1.0.0"
)


# ------------------------------------------------------------
# Lazy inference import
# ------------------------------------------------------------

def predict_image(image):

    from src.inference import predict_image as run_prediction

    return run_prediction(image)


# ------------------------------------------------------------
# Root
# ------------------------------------------------------------

@app.get("/")
def root():

    return {
        "message": "CIFAR-10 classifier API is running"
    }


# ------------------------------------------------------------
# Health
# ------------------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

@app.get("/metrics")
def get_metrics():

    successful = metrics["predictions_successful"]

    average_latency_ms = (
        metrics["total_latency_ms"] / successful
        if successful > 0
        else 0.0
    )

    average_confidence = (
        metrics["total_confidence"] / successful
        if successful > 0
        else 0.0
    )

    prediction_distribution = {}

    for class_name, count in metrics["prediction_counts"].items():

        percentage = (
            count / successful * 100
            if successful > 0
            else 0.0
        )

        prediction_distribution[class_name] = {
            "count": count,
            "percentage": round(
                percentage,
                2
            )
        }

    return {
        "requests_total": metrics["requests_total"],
        "predictions_successful": successful,
        "predictions_failed": metrics["predictions_failed"],
        "invalid_uploads": metrics["invalid_uploads"],
        "average_latency_ms": round(
            average_latency_ms,
            2
        ),
        "average_confidence": round(
            average_confidence,
            4
        ),
        "prediction_distribution": prediction_distribution
    }

@app.get("/drift")
def get_drift():

    successful = metrics["predictions_successful"]

    if successful == 0:

        return {
            "status": "insufficient_data",
            "message": "No successful predictions available"
        }

    drift_details = {}

    drift_detected = False

    for class_name, baseline_percentage in BASELINE_DISTRIBUTION.items():

        count = metrics["prediction_counts"][
            class_name
        ]

        current_percentage = (
            count / successful
        ) * 100

        difference = abs(
            current_percentage -
            baseline_percentage
        )

        class_drifted = (
            difference >
            DRIFT_THRESHOLD
        )

        if class_drifted:
            drift_detected = True

        drift_details[class_name] = {
            "baseline_percentage": baseline_percentage,
            "current_percentage": round(
                current_percentage,
                2
            ),
            "difference_percentage_points": round(
                difference,
                2
            ),
            "drift_detected": class_drifted
        }

    return {
        "status": (
            "drift_detected"
            if drift_detected
            else "no_drift_detected"
        ),
        "threshold_percentage_points": DRIFT_THRESHOLD,
        "classes": drift_details
    }

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    start_time = time.perf_counter()

    metrics["requests_total"] += 1

    logger.info(
        "prediction_request_started | filename=%s",
        file.filename
    )


    # --------------------------------------------------------
    # Read and validate image
    # --------------------------------------------------------

    try:

        image_bytes = await file.read()

        image = Image.open(
            BytesIO(image_bytes)
        )

        # Verify that this is actually a valid image
        image.verify()

        # Re-open because verify() can invalidate
        # the original PIL image object
        image = Image.open(
            BytesIO(image_bytes)
        )

    except (
        UnidentifiedImageError,
        OSError
    ):

        metrics["invalid_uploads"] += 1

        logger.warning(
            "invalid_image_upload | filename=%s",
            file.filename
        )

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image"
        )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    try:

        result = predict_image(
            image
        )

    except Exception:

        metrics["predictions_failed"] += 1

        logger.exception(
            "prediction_failed | filename=%s",
            file.filename
        )

        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        )


    # --------------------------------------------------------
    # Latency
    # --------------------------------------------------------

    latency_ms = (
        time.perf_counter() - start_time
    ) * 1000


    # --------------------------------------------------------
    # Update metrics
    # --------------------------------------------------------

    metrics["predictions_successful"] += 1
    metrics["total_latency_ms"] += latency_ms
    metrics["total_confidence"] += result["confidence"]

    metrics["prediction_counts"][
        result["class"]
    ] += 1

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    logger.info(
        (
            "prediction_request_completed | "
            "filename=%s | "
            "predicted_class=%s | "
            "confidence=%.4f | "
            "latency_ms=%.2f"
        ),
        file.filename,
        result["class"],
        result["confidence"],
        latency_ms
    )


    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {
        "filename": file.filename,
        "predicted_class": result["class"],
        "confidence": result["confidence"],
        "latency_ms": round(
            latency_ms,
            2
        )
    }