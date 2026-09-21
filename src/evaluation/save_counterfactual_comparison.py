from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# RESULTS FROM OUR 3 METHODS
# -----------------------------
results = {
    "Method": [
        "Basic Counterfactual",
        "Smooth Counterfactual",
        "Low-Resolution Counterfactual"
    ],

    "Target Confidence": [
        0.8380,
        0.8212,
        0.8954
    ],

    "Mean Absolute Change": [
        0.046196,
        0.045256,
        0.026228
    ],

    "Maximum Change": [
        0.143047,
        0.138903,
        0.078849
    ],

    "Fraction Changed > 0.05": [
        None,
        0.3694,
        0.1150
    ]
}


# -----------------------------
# CREATE DATAFRAME
# -----------------------------
df = pd.DataFrame(results)


# -----------------------------
# OUTPUT DIRECTORY
# -----------------------------
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)


# -----------------------------
# SAVE CSV
# -----------------------------
csv_path = output_dir / "counterfactual_comparison.csv"

df.to_csv(
    csv_path,
    index=False
)

print("\nCOUNTERFACTUAL COMPARISON")
print("-" * 70)

print(df)

print("-" * 70)
print(f"Saved: {csv_path}")


# -----------------------------
# CHART 1:
# TARGET CONFIDENCE
# -----------------------------
plt.figure(figsize=(9, 5))

plt.bar(
    df["Method"],
    df["Target Confidence"]
)

plt.ylabel("Target NORMAL Probability")

plt.title(
    "Counterfactual Target Confidence Comparison"
)

plt.ylim(0, 1)

plt.xticks(
    rotation=15,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    output_dir / "counterfactual_target_confidence.png",
    dpi=300
)

plt.close()


# -----------------------------
# CHART 2:
# MEAN ABSOLUTE CHANGE
# -----------------------------
plt.figure(figsize=(9, 5))

plt.bar(
    df["Method"],
    df["Mean Absolute Change"]
)

plt.ylabel("Mean Absolute Pixel Change")

plt.title(
    "Counterfactual Mean Pixel Change Comparison"
)

plt.xticks(
    rotation=15,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    output_dir / "counterfactual_mean_change.png",
    dpi=300
)

plt.close()


# -----------------------------
# CHART 3:
# MAXIMUM CHANGE
# -----------------------------
plt.figure(figsize=(9, 5))

plt.bar(
    df["Method"],
    df["Maximum Change"]
)

plt.ylabel("Maximum Pixel Change")

plt.title(
    "Counterfactual Maximum Pixel Change Comparison"
)

plt.xticks(
    rotation=15,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    output_dir / "counterfactual_max_change.png",
    dpi=300
)

plt.close()


# -----------------------------
# CHART 4:
# FRACTION CHANGED
# -----------------------------
fraction_df = df.dropna(
    subset=["Fraction Changed > 0.05"]
)

plt.figure(figsize=(8, 5))

plt.bar(
    fraction_df["Method"],
    fraction_df["Fraction Changed > 0.05"]
)

plt.ylabel(
    "Fraction of Pixels Changed > 0.05"
)

plt.title(
    "Counterfactual Changed-Pixel Fraction"
)

plt.ylim(0, 0.5)

plt.xticks(
    rotation=15,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    output_dir / "counterfactual_changed_fraction.png",
    dpi=300
)

plt.close()


# -----------------------------
# DONE
# -----------------------------
print(
    "Saved: outputs/counterfactual_target_confidence.png"
)

print(
    "Saved: outputs/counterfactual_mean_change.png"
)

print(
    "Saved: outputs/counterfactual_max_change.png"
)

print(
    "Saved: outputs/counterfactual_changed_fraction.png"
)

print("\nComparison complete.")