import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from src.data.dataset import ChestXrayDataset


dataset = ChestXrayDataset("data/train_split.csv")

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

images, labels = next(iter(loader))

model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)

model.eval()

with torch.no_grad():
    outputs = model(images)

print("Input batch shape:", images.shape)
print("True labels:", labels)
print("Model output shape:", outputs.shape)
print("Raw outputs:")
print(outputs)