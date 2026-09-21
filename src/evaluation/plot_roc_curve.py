import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import models

from sklearn.metrics import roc_curve, roc_auc_score

from src.data.dataset import ChestXrayDataset


# -----------------------------
# DATA
# -----------------------------
test_dataset = ChestXrayDataset("data/test_split.csv")

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)


# -----------------------------
# DEVICE
# -----------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# -----------------------------
# MODEL
# -----------------------------
model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)

model.load_state_dict(
    torch.load(
        "outputs/resnet18_baseline.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()


# -----------------------------
# COLLECT LABELS + PROBABILITIES
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

        all_labels.extend(
            labels.numpy()
        )

        all_probabilities.extend(
            probabilities[:, 1]
            .cpu()
            .numpy()
        )


# -----------------------------
# ROC
# -----------------------------
fpr, tpr, thresholds = roc_curve(
    all_labels,
    all_probabilities
)

auc_score = roc_auc_score(
    all_labels,
    all_probabilities
)


# -----------------------------
# PLOT
# -----------------------------
plt.figure()

plt.plot(
    fpr,
    tpr,
    label=f"ResNet18 (AUC = {auc_score:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Baseline ResNet18 ROC Curve")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "outputs/roc_curve_baseline.png",
    dpi=300
)

plt.show()

print(
    f"ROC-AUC: {auc_score:.4f}"
)

print(
    "Saved: outputs/roc_curve_baseline.png"
)