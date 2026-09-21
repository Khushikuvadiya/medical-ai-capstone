import time
from pathlib import Path

import pandas as pd

from api.predict import predict_xray
from api.explain import create_gradcam
from api.counterfactual import generate_counterfactual


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_IMAGE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "chest_xray"
    / "test"
    / "PNEUMONIA"
    / "person1_virus_6.jpeg"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "latency_results.csv"
)


# ---------------------------------------------------------
# HELPER
# ---------------------------------------------------------

def measure_latency(function, image_bytes, runs=3):

    times = []

    for _ in range(runs):

        start = time.perf_counter()

        function(
            image_bytes
        )

        end = time.perf_counter()

        times.append(
            end - start
        )

    average_time = sum(times) / len(times)

    return average_time, times


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    if not TEST_IMAGE.exists():

        raise FileNotFoundError(
            f"Test image not found: {TEST_IMAGE}"
        )

    with open(
        TEST_IMAGE,
        "rb"
    ) as file:

        image_bytes = file.read()


    print(
        "\nTesting prediction latency..."
    )

    prediction_average, prediction_runs = measure_latency(
        predict_xray,
        image_bytes,
        runs=5
    )


    print(
        "\nTesting Grad-CAM latency..."
    )

    gradcam_average, gradcam_runs = measure_latency(
        create_gradcam,
        image_bytes,
        runs=3
    )


    print(
        "\nTesting counterfactual latency..."
    )

    counterfactual_average, counterfactual_runs = measure_latency(
        generate_counterfactual,
        image_bytes,
        runs=1
    )


    results = pd.DataFrame(
        [
            {
                "operation":
                    "Prediction",

                "average_seconds":
                    prediction_average,

                "runs":
                    len(prediction_runs)
            },

            {
                "operation":
                    "Grad-CAM",

                "average_seconds":
                    gradcam_average,

                "runs":
                    len(gradcam_runs)
            },

            {
                "operation":
                    "Counterfactual",

                "average_seconds":
                    counterfactual_average,

                "runs":
                    len(counterfactual_runs)
            }
        ]
    )


    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False
    )


    print(
        "\nLatency Results"
    )

    print(
        results.to_string(
            index=False
        )
    )

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":

    main()