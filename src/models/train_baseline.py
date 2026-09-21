import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from src.data.dataset import ChestXrayDataset


# -----------------------------
# SETTINGS
# -----------------------------
BATCH_SIZE = 16
EPOCHS = 1
LEARNING_RATE = 0.001


# -----------------------------
# DATASETS
# -----------------------------
train_dataset = ChestXrayDataset("data/train_split.csv")
val_dataset = ChestXrayDataset("data/val_split.csv")


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


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
model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

# Freeze pretrained layers
for param in model.parameters():
    param.requires_grad = False

# Replace final classifier
model.fc = nn.Linear(
    model.fc.in_features,
    2
)

model = model.to(device)


# -----------------------------
# CLASS WEIGHTS
# -----------------------------
# Train counts:
# NORMAL = 1073
# PNEUMONIA = 3099

class_weights = torch.tensor(
    [3099 / 1073, 1.0],
    dtype=torch.float32
).to(device)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# -----------------------------
# OPTIMIZER
# -----------------------------
optimizer = torch.optim.Adam(
    model.fc.parameters(),
    lr=LEARNING_RATE
)


# -----------------------------
# TRAINING LOOP
# -----------------------------
for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_index, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

        if (batch_index + 1) % 50 == 0:
            print(
                f"Batch {batch_index + 1}/{len(train_loader)} "
                f"Loss: {loss.item():.4f}"
            )

    train_loss = (
        running_loss / len(train_loader)
    )

    train_accuracy = (
        100 * correct / total
    )

    print()
    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )
    print(
        f"Training Loss: {train_loss:.4f}"
    )
    print(
        f"Training Accuracy: "
        f"{train_accuracy:.2f}%"
    )


    # -------------------------
    # VALIDATION
    # -------------------------
    model.eval()

    val_correct = 0
    val_total = 0
    val_loss = 0.0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_loss += loss.item()

            _, predicted = torch.max(
                outputs,
                1
            )

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()


    val_loss = (
        val_loss / len(val_loader)
    )

    val_accuracy = (
        100 * val_correct / val_total
    )

    print(
        f"Validation Loss: "
        f"{val_loss:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{val_accuracy:.2f}%"
    )

    print("-" * 40)


# -----------------------------
# SAVE MODEL
# -----------------------------
torch.save(
    model.state_dict(),
    "outputs/resnet18_baseline.pth"
)

print(
    "Model saved to "
    "outputs/resnet18_baseline.pth"
)