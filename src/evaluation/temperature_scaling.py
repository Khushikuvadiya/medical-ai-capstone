import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd

from torch.utils.data import DataLoader
from torchvision import models

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from src.data.dataset import ChestXrayDataset


# -----------------------------
# SETTINGS
# -----------------------------
VAL_CSV = "data/val_split.csv"
TEST_CSV = "data/test_split.csv"
MODEL_PATH = "outputs/resnet18_baseline.pth"

BATCH_SIZE = 16


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
val_dataset = ChestXrayDataset(
    VAL_CSV
)

test_dataset = ChestXrayDataset(
    TEST_CSV
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
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
# COLLECT LOGITS
# -----------------------------
def collect_logits(loader):

    logits_list = []
    labels_list = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            logits = model(images)

            logits_list.append(
                logits.cpu()
            )

            labels_list.append(
                labels
            )

    logits = torch.cat(
        logits_list
    )

    labels = torch.cat(
        labels_list
    )

    return logits, labels


print(
    "Collecting validation logits..."
)

val_logits, val_labels = collect_logits(
    val_loader
)

print(
    "Collecting test logits..."
)

test_logits, test_labels = collect_logits(
    test_loader
)


# -----------------------------
# TEMPERATURE PARAMETER
# -----------------------------
temperature = nn.Parameter(
    torch.ones(1)
)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.LBFGS(
    [temperature],
    lr=0.01,
    max_iter=100
)


# -----------------------------
# FIT TEMPERATURE
# USING VALIDATION SET ONLY
# -----------------------------
def closure():

    optimizer.zero_grad()

    scaled_logits = (
        val_logits / temperature
    )

    loss = criterion(
        scaled_logits,
        val_labels
    )

    loss.backward()

    return loss


print(
    "\nFitting temperature "
    "on validation set..."
)

optimizer.step(
    closure
)


# Prevent invalid temperature
with torch.no_grad():
    temperature.clamp_(min=0.05)


temperature_value = (
    temperature.item()
)


print(
    f"Optimal Temperature: "
    f"{temperature_value:.4f}"
)


# -----------------------------
# TEST PROBABILITIES
# -----------------------------
uncalibrated_probabilities = (
    torch.softmax(
        test_logits,
        dim=1
    )[:, 1]
    .numpy()
)

calibrated_probabilities = (
    torch.softmax(
        test_logits / temperature_value,
        dim=1
    )[:, 1]
    .numpy()
)

test_labels_np = (
    test_labels.numpy()
)


# -----------------------------
# BRIER SCORES
# -----------------------------
brier_before = brier_score_loss(
    test_labels_np,
    uncalibrated_probabilities
)

brier_after = brier_score_loss(
    test_labels_np,
    calibrated_probabilities
)


print("\nCALIBRATION RESULTS")
print("-" * 45)

print(
    f"Brier Score Before: "
    f"{brier_before:.4f}"
)

print(
    f"Brier Score After:  "
    f"{brier_after:.4f}"
)


# -----------------------------
# CALIBRATION CURVES
# -----------------------------
true_before, pred_before = (
    calibration_curve(
        test_labels_np,
        uncalibrated_probabilities,
        n_bins=10,
        strategy="uniform"
    )
)

true_after, pred_after = (
    calibration_curve(
        test_labels_np,
        calibrated_probabilities,
        n_bins=10,
        strategy="uniform"
    )
)


# -----------------------------
# SAVE RESULTS
# -----------------------------
summary = pd.DataFrame({
    "Metric": [
        "Temperature",
        "Brier Before",
        "Brier After"
    ],

    "Value": [
        temperature_value,
        brier_before,
        brier_after
    ]
})

summary.to_csv(
    "outputs/temperature_scaling_results.csv",
    index=False
)

print(
    "\nSaved: "
    "outputs/temperature_scaling_results.csv"
)


# -----------------------------
# PLOT
# -----------------------------
plt.figure(
    figsize=(7, 7)
)

plt.plot(
    pred_before,
    true_before,
    marker="o",
    label=(
        f"Before "
        f"(Brier={brier_before:.4f})"
    )
)

plt.plot(
    pred_after,
    true_after,
    marker="o",
    label=(
        f"After "
        f"(Brier={brier_after:.4f})"
    )
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect calibration"
)

plt.xlabel(
    "Mean Predicted Probability"
)

plt.ylabel(
    "Observed Fraction Positive"
)

plt.title(
    "Temperature Scaling Calibration"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "outputs/temperature_scaling_curve.png",
    dpi=300
)

plt.show()


print(
    "Saved: "
    "outputs/temperature_scaling_curve.png"
)

print(
    "\nTemperature scaling complete."
)