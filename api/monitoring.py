import csv
import time
from datetime import datetime
from pathlib import Path
from threading import Lock


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MONITORING_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "monitoring"
)

MONITORING_FILE = (
    MONITORING_DIR
    / "api_monitoring.csv"
)


# =========================================================
# THREAD LOCK
# =========================================================

monitoring_lock = Lock()


# =========================================================
# CSV COLUMNS
# =========================================================

FIELDNAMES = [
    "timestamp",
    "endpoint",
    "method",
    "status_code",
    "success",
    "latency_seconds",
    "error_message",
]


# =========================================================
# INITIALIZE FILE
# =========================================================

def initialize_monitoring():

    MONITORING_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if not MONITORING_FILE.exists():

        with open(
            MONITORING_FILE,
            mode="w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=FIELDNAMES
            )

            writer.writeheader()


# =========================================================
# SAVE MONITORING EVENT
# =========================================================

def log_api_event(
    endpoint,
    method,
    status_code,
    success,
    latency_seconds,
    error_message=""
):

    initialize_monitoring()

    record = {
        "timestamp":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "endpoint":
            endpoint,

        "method":
            method,

        "status_code":
            status_code,

        "success":
            success,

        "latency_seconds":
            round(
                float(latency_seconds),
                6
            ),

        "error_message":
            error_message,
    }


    with monitoring_lock:

        with open(
            MONITORING_FILE,
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


# =========================================================
# TIMER HELPER
# =========================================================

def start_timer():

    return time.perf_counter()


def stop_timer(start_time):

    return (
        time.perf_counter()
        - start_time
    )