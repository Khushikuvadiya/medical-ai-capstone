import torch
import torch.nn as nn
import pandas as pd

from torch.utils.data import DataLoader
from torchvision import models

from src.data.dataset import ChestXrayDataset


# -----------------------------
# SETTINGS
# -----------------------------
TEST_CSV = "data/test_split.csv"
MODEL_PATH = "outputs/resnet18_baseline.pth"

BATCH_SIZE = 16
N_BINS = 10

# Temperature from previous experiment
TEMPERATURE = 0.8129


# -----------------------------
# DEVICE
# -----------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# -----------------------------
# DATA
# -----------------------------
test_dataset = ChestXrayDataset(
    TEST_CSV
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# -----------------------------
# MODEL
# -----------------------------
model = models.resnet18(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)
model.eval()


# -----------------------------
# COLLECT LOGITS + LABELS
# -----------------------------
all_logits = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        logits = model(images)

        all_logits.append(
            logits.cpu()
        )

        all_labels.append(
            labels
        )


all_logits = torch.cat(
    all_logits
)

all_labels = torch.cat(
    all_labels
)


# -----------------------------
# ECE FUNCTION
# -----------------------------
def expected_calibration_error(
    logits,
    labels,
    n_bins=10
):

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    confidences, predictions = torch.max(
        probabilities,
        dim=1
    )

    accuracies = predictions.eq(
        labels
    )

    bin_boundaries = torch.linspace(
        0,
        1,
        n_bins + 1
    )

    ece = torch.zeros(1)

    bin_results = []

    for i in range(n_bins):

        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]

        in_bin = (
            confidences > lower
        ) & (
            confidences <= upper
        )

        proportion_in_bin = (
            in_bin.float().mean()
        )

        if proportion_in_bin.item() > 0:

            accuracy_in_bin = (
                accuracies[in_bin]
                .float()
                .mean()
            )

            confidence_in_bin = (
                confidences[in_bin]
                .mean()
            )

            gap = torch.abs(
                confidence_in_bin
                - accuracy_in_bin
            )

            ece += (
                gap * proportion_in_bin
            )

            bin_results.append({
                "Lower Bound": lower.item(),
                "Upper Bound": upper.item(),
                "Count": int(
                    in_bin.sum().item()
                ),
                "Mean Confidence":
                    confidence_in_bin.item(),
                "Accuracy":
                    accuracy_in_bin.item(),
                "Gap":
                    gap.item()
            })

    return (
        ece.item(),
        bin_results
    )


# -----------------------------
# BEFORE CALIBRATION
# -----------------------------
ece_before, bins_before = (
    expected_calibration_error(
        all_logits,
        all_labels,
        N_BINS
    )
)


# -----------------------------
# AFTER TEMPERATURE SCALING
# -----------------------------
scaled_logits = (
    all_logits / TEMPERATURE
)

ece_after, bins_after = (
    expected_calibration_error(
        scaled_logits,
        all_labels,
        N_BINS
    )
)


# -----------------------------
# RESULTS
# -----------------------------
print("\nECE RESULTS")
print("-" * 40)

print(
    f"ECE Before: {ece_before:.4f}"
)

print(
    f"ECE After:  {ece_after:.4f}"
)

print("-" * 40)


# -----------------------------
# SAVE SUMMARY
# -----------------------------
summary_df = pd.DataFrame({
    "Metric": [
        "ECE Before",
        "ECE After",
        "Temperature"
    ],

    "Value": [
        ece_before,
        ece_after,
        TEMPERATURE
    ]
})

summary_df.to_csv(
    "outputs/calibration_ece_summary.csv",
    index=False
)


# -----------------------------
# SAVE BIN DETAILS
# -----------------------------
pd.DataFrame(
    bins_before
).to_csv(
    "outputs/ece_bins_before.csv",
    index=False
)

pd.DataFrame(
    bins_after
).to_csv(
    "outputs/ece_bins_after.csv",
    index=False
)


print(
    "Saved: outputs/calibration_ece_summary.csv"
)

print(
    "Saved: outputs/ece_bins_before.csv"
)

print(
    "Saved: outputs/ece_bins_after.csv"
)

print(
    "\nECE evaluation complete."
)