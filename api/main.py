import io

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Depends,
)

from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError

from api.predict import predict_xray
from api.explain import create_gradcam
from api.counterfactual import generate_counterfactual

from api.audit import (
    log_analysis,
    log_review,
)

from api.monitoring import (
    log_api_event,
    start_timer,
    stop_timer,
)

from api.security import (
    require_api_key,
    require_reviewer,
)


app = FastAPI(
    title="Medical AI Capstone API",
    description=(
        "Research prototype API for chest X-ray classification, "
        "explainability, counterfactual analysis, audit logging, "
        "operational monitoring, authentication and authorization."
    ),
    version="0.8.0",
)


# =========================================================
# SETTINGS
# =========================================================

MAX_FILE_SIZE = 5 * 1024 * 1024


# =========================================================
# REVIEW REQUEST MODEL
# =========================================================

class ReviewRequest(BaseModel):
    filename: str
    prediction: str
    normal_probability: float
    pneumonia_probability: float
    reviewer_decision: str


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():

    return {
        "message": "Medical AI Capstone API is running"
    }


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# =========================================================
# FILE TYPE VALIDATION
# =========================================================

def validate_file_type(
    file: UploadFile
):

    allowed_types = [
        "image/jpeg",
        "image/png",
    ]

    if file.content_type not in allowed_types:

        raise HTTPException(
            status_code=400,
            detail=(
                "Only JPEG and PNG images are supported."
            ),
        )


# =========================================================
# IMAGE CONTENT VALIDATION
# =========================================================

def validate_image_bytes(
    image_bytes: bytes
):

    if len(image_bytes) == 0:

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    if len(image_bytes) > MAX_FILE_SIZE:

        raise HTTPException(
            status_code=413,
            detail=(
                "Uploaded image is too large. "
                "Maximum allowed size is 5 MB."
            ),
        )

    try:

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        image.verify()

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "The uploaded file is not a valid image."
            ),
        )


# =========================================================
# READ + VALIDATE IMAGE
# =========================================================

async def read_and_validate_image(
    file: UploadFile
):

    validate_file_type(
        file
    )

    image_bytes = await file.read()

    validate_image_bytes(
        image_bytes
    )

    return image_bytes


# =========================================================
# PREDICTION
# Researcher OR Reviewer can access
# =========================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    api_key: str = Depends(require_api_key),
):

    start_time = start_timer()

    try:

        image_bytes = await read_and_validate_image(
            file
        )

        result = predict_xray(
            image_bytes
        )

        log_analysis(
            filename=file.filename,
            prediction=result["prediction"],
            normal_probability=(
                result["normal_probability"]
            ),
            pneumonia_probability=(
                result["pneumonia_probability"]
            ),
            explanation_method="Prediction",
        )

        latency = stop_timer(
            start_time
        )

        log_api_event(
            endpoint="/predict",
            method="POST",
            status_code=200,
            success=True,
            latency_seconds=latency,
        )

        return {
            "filename": file.filename,
            "research_use_only": True,
            "authenticated": True,
            **result,
        }

    except HTTPException as error:

        latency = stop_timer(
            start_time
        )

        log_api_event(
            endpoint="/predict",
            method="POST",
            status_code=error.status_code,
            success=False,
            latency_seconds=latency,
            error_message=str(
                error.detail
            ),
        )

        raise

    except Exception as error:

        latency = stop_timer(
            start_time
        )

        log_api_event(
            endpoint="/predict",
            method="POST",
            status_code=500,
            success=False,
            latency_seconds=latency,
            error_message=str(error),
        )

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


# =========================================================
# GRAD-CAM EXPLANATION
# Researcher OR Reviewer can access
# =========================================================

@app.post("/explain")
async def explain(
    file: UploadFile = File(...),
    api_key: str = Depends(require_api_key),
):

    start_time = start_timer()

    try:

        image_bytes = await read_and_validate_image(
            file
        )

        result = create_gradcam(
            image_bytes
        )

        log_analysis(
            filename=file.filename,
            prediction=result["prediction"],
            normal_probability=(
                result["normal_probability"]
            ),
            pneumonia_probability=(
                result["pneumonia_probability"]
            ),
            explanation_method="Grad-CAM",
        )

        latency = stop_timer(
            start_time
        )

        log_api_event(
            endpoint="/explain",
            method="POST",
            status_code=200,
            success=True,
            latency_seconds=latency,
        )

        return {
            "filename": file.filename,
            "research_use_only": True,
            "authenticated": True,
            **result,
        }

    except HTTPException as error:

        latency = stop_timer(
            start_time
        )

        log_api_event(
            endpoint="/explain",
            method="POST",
            status_code=error.status_code,
            success=False,
            latency_seconds=latency,
            error_message=str(
                error.detail
            ),
        )

        raise

    except Exception as error:

        latency = stop_timer(
            start_time
        )

        log_api_event(
            endpoint="/explain",
            method="POST",
            status_code=500,
            success=False,
            latency_seconds=latency,
            error_message=str(error),
        )

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


# =========================================================
# COUNTERFACTUAL
# Researcher OR Reviewer can access
# =========================================================

@app.post("/counterfactual")
async def counterfactual(
    file: UploadFile = File(...),
    api_key: str = Depends(require_api_key),
):

    start_time = start_timer()

    try:

        image_bytes = await read_and_validate_image(
            file
        )

        result = generate_counterfactual(
            image_bytes
        )

        log_analysis(
            filename=file.filename,
            prediction=(
                result[
                    "counterfactual_prediction"
                ]
            ),
            normal_probability=(
                result[
                    "counterfactual_normal_probability"
                ]
            ),
            pneumonia_probability=(
                result[
                    "counterfactual_pneumonia_probability"
                ]
            ),
            explanation_method="Counterfactual",
        )

        latency = stop_timer(
            start_time
        )

        log_api_event(
            endpoint="/counterfactual",
            method="POST",
            status_code=200,
            success=True,
            latency_seconds=latency,
        )

        return {
            "filename": file.filename,
            "research_use_only": True,
            "authenticated": True,
            **result,
        }

    except HTTPException as error:

        latency = stop_timer(
            start_time
        )

        log_api_event(
            endpoint="/counterfactual",
            method="POST",
            status_code=error.status_code,
            success=False,
            latency_seconds=latency,
            error_message=str(
                error.detail
            ),
        )

        raise

    except Exception as error:

        latency = stop_timer(
            start_time
        )

        log_api_event(
            endpoint="/counterfactual",
            method="POST",
            status_code=500,
            success=False,
            latency_seconds=latency,
            error_message=str(error),
        )

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


# =========================================================
# REVIEWER FEEDBACK
# Reviewer only
# =========================================================

@app.post("/review")
def review(
    review_data: ReviewRequest,
    api_key: str = Depends(require_reviewer),
):

    start_time = start_timer()

    allowed_decisions = [
        "Agree with AI",
        "Disagree with AI",
        "Needs further review",
    ]

    allowed_predictions = [
        "NORMAL",
        "PNEUMONIA",
    ]

    try:

        # -------------------------------------------------
        # REVIEW DECISION VALIDATION
        # -------------------------------------------------

        if (
            review_data.reviewer_decision
            not in allowed_decisions
        ):

            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid reviewer decision."
                ),
            )


        # -------------------------------------------------
        # PREDICTION VALIDATION
        # -------------------------------------------------

        if (
            review_data.prediction
            not in allowed_predictions
        ):

            raise HTTPException(
                status_code=400,
                detail=(
                    "Prediction must be NORMAL "
                    "or PNEUMONIA."
                ),
            )


        # -------------------------------------------------
        # NORMAL PROBABILITY VALIDATION
        # -------------------------------------------------

        if not (
            0.0
            <= review_data.normal_probability
            <= 1.0
        ):

            raise HTTPException(
                status_code=400,
                detail=(
                    "NORMAL probability must be "
                    "between 0 and 1."
                ),
            )


        # -------------------------------------------------
        # PNEUMONIA PROBABILITY VALIDATION
        # -------------------------------------------------

        if not (
            0.0
            <= review_data.pneumonia_probability
            <= 1.0
        ):

            raise HTTPException(
                status_code=400,
                detail=(
                    "PNEUMONIA probability must be "
                    "between 0 and 1."
                ),
            )


        # -------------------------------------------------
        # PROBABILITY SUM VALIDATION
        # -------------------------------------------------

        probability_sum = (
            review_data.normal_probability
            + review_data.pneumonia_probability
        )

        if abs(
            probability_sum - 1.0
        ) > 0.01:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Class probabilities must "
                    "approximately sum to 1."
                ),
            )


        # -------------------------------------------------
        # SAVE REVIEW
        # -------------------------------------------------

        record = log_review(
            filename=review_data.filename,
            prediction=review_data.prediction,
            normal_probability=(
                review_data.normal_probability
            ),
            pneumonia_probability=(
                review_data.pneumonia_probability
            ),
            reviewer_decision=(
                review_data.reviewer_decision
            ),
        )

        latency = stop_timer(
            start_time
        )

        log_api_event(
            endpoint="/review",
            method="POST",
            status_code=200,
            success=True,
            latency_seconds=latency,
        )

        return {
            "message":
                "Reviewer decision saved successfully.",

            "authenticated":
                True,

            "authorized_role":
                "reviewer",

            "audit_record":
                record,
        }

    except HTTPException as error:

        latency = stop_timer(
            start_time
        )

        log_api_event(
            endpoint="/review",
            method="POST",
            status_code=error.status_code,
            success=False,
            latency_seconds=latency,
            error_message=str(
                error.detail
            ),
        )

        raise

    except Exception as error:

        latency = stop_timer(
            start_time
        )

        log_api_event(
            endpoint="/review",
            method="POST",
            status_code=500,
            success=False,
            latency_seconds=latency,
            error_message=str(error),
        )

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )