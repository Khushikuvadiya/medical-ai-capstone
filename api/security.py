import os

from fastapi import (
    Header,
    HTTPException,
)


# =========================================================
# API KEYS FROM ENVIRONMENT
# =========================================================

RESEARCHER_API_KEY = os.getenv(
    "MEDICAL_AI_RESEARCHER_KEY"
)

REVIEWER_API_KEY = os.getenv(
    "MEDICAL_AI_REVIEWER_KEY"
)


# =========================================================
# AUTHENTICATION
# =========================================================

def require_api_key(
    x_api_key: str = Header(...)
):

    if not RESEARCHER_API_KEY or not REVIEWER_API_KEY:

        raise HTTPException(
            status_code=503,
            detail=(
                "API authentication is not configured."
            ),
        )

    valid_keys = {
        RESEARCHER_API_KEY,
        REVIEWER_API_KEY,
    }

    if x_api_key not in valid_keys:

        raise HTTPException(
            status_code=401,
            detail="Invalid API key.",
        )

    return x_api_key


# =========================================================
# AUTHORIZATION
# =========================================================

def require_reviewer(
    x_api_key: str = Header(...)
):

    if not REVIEWER_API_KEY:

        raise HTTPException(
            status_code=503,
            detail=(
                "Reviewer authentication is not configured."
            ),
        )

    if x_api_key != REVIEWER_API_KEY:

        raise HTTPException(
            status_code=403,
            detail="Reviewer permission required.",
        )

    return x_api_key