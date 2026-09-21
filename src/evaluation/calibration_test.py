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
# COLLECT PROBABILITIES
# -----------------------------
all_labels = []
all_probabilities = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        pneumonia_probabilities = (
            probabilities[:, 1]
            .cpu()
            .numpy()
        )

        all_probabilities.extend(
            pneumonia_probabilities
        )

        all_labels.extend(
            labels.numpy()
        )


# -----------------------------
# BRIER SCORE
# -----------------------------
brier_score = brier_score_loss(
    all_labels,
    all_probabilities
)

print(
    f"Brier Score: {brier_score:.4f}"
)


# -----------------------------
# CALIBRATION CURVE
# -----------------------------
prob_true, prob_pred = calibration_curve(
    all_labels,
    all_probabilities,
    n_bins=10,
    strategy="uniform"
)


# -----------------------------
# SAVE CALIBRATION VALUES
# -----------------------------
calibration_df = pd.DataFrame({
    "Mean Predicted Probability": prob_pred,
    "Observed Fraction Positive": prob_true
})

calibration_df.to_csv(
    "outputs/calibration_values.csv",
    index=False
)

print(
    "Saved: outputs/calibration_values.csv"
)


# -----------------------------
# PLOT
# -----------------------------
plt.figure(
    figsize=(7, 7)
)

plt.plot(
    prob_pred,
    prob_true,
    marker="o",
    label="ResNet18"
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
    f"Calibration Curve\n"
    f"Brier Score = {brier_score:.4f}"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "outputs/calibration_curve.png",
    dpi=300
)

plt.show()


print(
    "Saved: outputs/calibration_curve.png"
)

print(
    "\nCalibration evaluation complete."
)