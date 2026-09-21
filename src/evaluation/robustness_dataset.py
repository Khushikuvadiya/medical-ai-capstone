from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# -----------------------------
# SETTINGS
# -----------------------------
TEST_CSV = "data/test_split.csv"
MODEL_PATH = "outputs/resnet18_baseline.pth"

BATCH_SIZE = 16

np.random.seed(42)
torch.manual_seed(42)


# -----------------------------
# DEVICE
# -----------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


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
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)
model.eval()


# -----------------------------
# BASIC TRANSFORM
# -----------------------------
final_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------
# DATASET CLASS
# -----------------------------
class RobustnessDataset(Dataset):

    def __init__(self, csv_file, condition):
        self.data = pd.read_csv(csv_file)
        self.condition = condition

        self.label_map = {
            "NORMAL": 0,
            "PNEUMONIA": 1
        }

    def __len__(self):
        return len(self.data)

    def apply_condition(self, image):

        if self.condition == "Original":
            return image

        if self.condition == "Darker":
            return ImageEnhance.Brightness(
                image
            ).enhance(0.7)

        if self.condition == "Brighter":
            return ImageEnhance.Brightness(
                image
            ).enhance(1.3)

        if self.condition == "Blurred":
            return image.filter(
                ImageFilter.GaussianBlur(radius=2)
            )

        if self.condition == "Low Contrast":
            return ImageEnhance.Contrast(
                image
            ).enhance(0.7)

        if self.condition == "Gaussian Noise":

            image_array = np.array(
                image
            ).astype(np.float32)

            noise = np.random.normal(
                0,
                12,
                image_array.shape
            )

            noisy_array = np.clip(
                image_array + noise,
                0,
                255
            ).astype(np.uint8)

            return Image.fromarray(
                noisy_array
            )

        return image

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_path = Path(
            row["image_path"]
        )

        label_name = row["label"]

        image = Image.open(
            image_path
        ).convert("L")

        image = self.apply_condition(
            image
        )

        image = final_transform(
            image
        )

        label = self.label_map[
            label_name
        ]

        return (
            image,
            torch.tensor(
                label,
                dtype=torch.long
            )
        )


# -----------------------------
# CONDITIONS
# -----------------------------
conditions = [
    "Original",
    "Darker",
    "Brighter",
    "Blurred",
    "Gaussian Noise",
    "Low Contrast"
]


# -----------------------------
# EVALUATION FUNCTION
# -----------------------------
def evaluate_condition(condition):

    # Make Gaussian noise reproducible
    np.random.seed(42)

    dataset = RobustnessDataset(
        TEST_CSV,
        condition
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    labels_all = []
    predictions_all = []
    probabilities_all = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            outputs = model(images)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            predictions = torch.argmax(
                probabilities,
                dim=1
            )

            labels_all.extend(
                labels.numpy()
            )

            predictions_all.extend(
                predictions
                .cpu()
                .numpy()
            )

            probabilities_all.extend(
                probabilities[:, 1]
                .cpu()
                .numpy()
            )


    accuracy = accuracy_score(
        labels_all,
        predictions_all
    )

    precision = precision_score(
        labels_all,
        predictions_all
    )

    recall = recall_score(
        labels_all,
        predictions_all
    )

    f1 = f1_score(
        labels_all,
        predictions_all
    )

    roc_auc = roc_auc_score(
        labels_all,
        probabilities_all
    )


    return {
        "Condition": condition,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": roc_auc
    }


# -----------------------------
# RUN ALL CONDITIONS
# -----------------------------
results = []

print("\nDATASET ROBUSTNESS TEST")
print("-" * 75)

for condition in conditions:

    result = evaluate_condition(
        condition
    )

    results.append(result)

    print(
        f"{condition:15s} | "
        f"Accuracy: {result['Accuracy']:.4f} | "
        f"F1: {result['F1']:.4f} | "
        f"ROC-AUC: {result['ROC_AUC']:.4f}"
    )


# -----------------------------
# SAVE CSV
# -----------------------------
results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    "outputs/robustness_dataset_metrics.csv",
    index=False
)

print(
    "\nSaved: "
    "outputs/robustness_dataset_metrics.csv"
)


# -----------------------------
# ACCURACY GRAPH
# -----------------------------
plt.figure(figsize=(10, 5))

plt.bar(
    results_df["Condition"],
    results_df["Accuracy"]
)

plt.ylabel("Accuracy")

plt.title(
    "Accuracy Under Image Perturbations"
)

plt.ylim(0, 1)

plt.xticks(
    rotation=20,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    "outputs/robustness_accuracy.png",
    dpi=300
)

plt.close()


# -----------------------------
# ROC-AUC GRAPH
# -----------------------------
plt.figure(figsize=(10, 5))

plt.bar(
    results_df["Condition"],
    results_df["ROC_AUC"]
)

plt.ylabel("ROC-AUC")

plt.title(
    "ROC-AUC Under Image Perturbations"
)

plt.ylim(0, 1)

plt.xticks(
    rotation=20,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    "outputs/robustness_roc_auc.png",
    dpi=300
)

plt.close()


print(
    "Saved: outputs/robustness_accuracy.png"
)

print(
    "Saved: outputs/robustness_roc_auc.png"
)

print("\nRobustness evaluation complete.")