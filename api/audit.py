import csv
from datetime import datetime
from pathlib import Path
from threading import Lock


# -----------------------------
# AUDIT LOG SETTINGS
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

AUDIT_DIR = PROJECT_ROOT / "outputs" / "audit"

AUDIT_FILE = AUDIT_DIR / "audit_log.csv"


# Thread lock so multiple requests
# do not write at the same time.
audit_lock = Lock()


# -----------------------------
# CSV COLUMNS
# -----------------------------
FIELDNAMES = [
    "timestamp",
    "filename",
    "prediction",
    "normal_probability",
    "pneumonia_probability",
    "confidence",
    "explanation_method",
    "reviewer_decision",
    "research_use_only",
]


# -----------------------------
# CREATE AUDIT FILE
# -----------------------------
def initialize_audit_log():

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if not AUDIT_FILE.exists():

        with open(
            AUDIT_FILE,
            mode="w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=FIELDNAMES
            )

            writer.writeheader()


# -----------------------------
# WRITE AUDIT EVENT
# -----------------------------
def log_analysis(
    filename,
    prediction,
    normal_probability,
    pneumonia_probability,
    explanation_method="None",
    reviewer_decision="Not reviewed",
):

    initialize_audit_log()

    confidence = max(
        float(normal_probability),
        float(pneumonia_probability)
    )

    record = {
        "timestamp":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "filename":
            filename,

        "prediction":
            prediction,

        "normal_probability":
            round(
                float(normal_probability),
                6
            ),

        "pneumonia_probability":
            round(
                float(pneumonia_probability),
                6
            ),

        "confidence":
            round(
                confidence,
                6
            ),

        "explanation_method":
            explanation_method,

        "reviewer_decision":
            reviewer_decision,

        "research_use_only":
            True,
    }

    with audit_lock:

        with open(
            AUDIT_FILE,
            mode="a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=FIELDNAMES
            )

            writer.writerow(
                record
            )

    return record


# -----------------------------
# LOG REVIEW DECISION
# -----------------------------
def log_review(
    filename,
    prediction,
    normal_probability,
    pneumonia_probability,
    reviewer_decision,
):

    return log_analysis(
        filename=filename,
        prediction=prediction,
        normal_probability=normal_probability,
        pneumonia_probability=pneumonia_probability,
        explanation_method="Reviewer feedback",
        reviewer_decision=reviewer_decision,
    )