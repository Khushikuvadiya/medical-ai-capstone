from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_root():

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert (
        data["message"]
        == "Medical AI Capstone API is running"
    )


def test_health():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"


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