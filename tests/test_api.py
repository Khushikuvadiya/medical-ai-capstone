import sys
import types


# =========================================================
# MOCK MODEL MODULES FOR CI
# =========================================================

mock_predict = types.ModuleType("api.predict")


def fake_predict_xray(image_bytes):
    return {
        "prediction": "NORMAL",
        "normal_probability": 0.8,
        "pneumonia_probability": 0.2,
    }


mock_predict.predict_xray = fake_predict_xray

sys.modules["api.predict"] = mock_predict


mock_explain = types.ModuleType("api.explain")


def fake_create_gradcam(image_bytes):
    return {
        "prediction": "NORMAL",
        "normal_probability": 0.8,
        "pneumonia_probability": 0.2,
        "gradcam_image_base64": "fake",
    }


mock_explain.create_gradcam = fake_create_gradcam

sys.modules["api.explain"] = mock_explain


mock_counterfactual = types.ModuleType(
    "api.counterfactual"
)


def fake_generate_counterfactual(image_bytes):
    return {
        "original_prediction": "NORMAL",
        "original_normal_probability": 0.8,
        "original_pneumonia_probability": 0.2,

        "counterfactual_prediction": "PNEUMONIA",
        "counterfactual_normal_probability": 0.2,
        "counterfactual_pneumonia_probability": 0.8,

        "mean_absolute_change": 0.02,
        "maximum_change": 0.05,
        "fraction_changed_over_005": 0.1,

        "counterfactual_image_base64": "fake",
        "difference_image_base64": "fake",
    }


mock_counterfactual.generate_counterfactual = (
    fake_generate_counterfactual
)

sys.modules[
    "api.counterfactual"
] = mock_counterfactual


# =========================================================
# IMPORT APP AFTER MOCKING
# =========================================================

from fastapi.testclient import TestClient
from api.main import app


client = TestClient(app)


# =========================================================
# ROOT
# =========================================================

def test_root():

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert (
        data["message"]
        == "Medical AI Capstone API is running"
    )


# =========================================================
# HEALTH
# =========================================================

def test_health():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"


# =========================================================
# INVALID PREDICT FILE
# =========================================================

def test_predict_rejects_invalid_file():

    files = {
        "file": (
            "test.txt",
            b"this is not an image",
            "text/plain"
        )
    }

    response = client.post(
        "/predict",
        files=files
    )

    assert response.status_code == 400


# =========================================================
# INVALID EXPLAIN FILE
# =========================================================

def test_explain_rejects_invalid_file():

    files = {
        "file": (
            "test.txt",
            b"this is not an image",
            "text/plain"
        )
    }

    response = client.post(
        "/explain",
        files=files
    )

    assert response.status_code == 400


# =========================================================
# INVALID COUNTERFACTUAL FILE
# =========================================================

def test_counterfactual_rejects_invalid_file():

    files = {
        "file": (
            "test.txt",
            b"this is not an image",
            "text/plain"
        )
    }

    response = client.post(
        "/counterfactual",
        files=files
    )

    assert response.status_code == 400


# =========================================================
# VALID REVIEW
# =========================================================

def test_review_valid_decision():

    payload = {
        "filename": "test_image.jpeg",
        "prediction": "NORMAL",
        "normal_probability": 0.80,
        "pneumonia_probability": 0.20,
        "reviewer_decision": "Agree with AI"
    }

    response = client.post(
        "/review",
        json=payload
    )

    assert response.status_code == 200


# =========================================================
# INVALID REVIEW
# =========================================================

def test_review_invalid_decision():

    payload = {
        "filename": "test_image.jpeg",
        "prediction": "NORMAL",
        "normal_probability": 0.80,
        "pneumonia_probability": 0.20,
        "reviewer_decision": "Invalid Decision"
    }

    response = client.post(
        "/review",
        json=payload
    )

    assert response.status_code == 400