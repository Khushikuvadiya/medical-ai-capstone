import pandas as pd

metrics = {
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ],
    "Value": [
        0.8285,
        0.8004,
        0.9667,
        0.8757,
        0.9434
    ]
}

df = pd.DataFrame(metrics)

df.to_csv(
    "outputs/baseline_metrics.csv",
    index=False
)

print(df)

print("\nSaved: outputs/baseline_metrics.csv")