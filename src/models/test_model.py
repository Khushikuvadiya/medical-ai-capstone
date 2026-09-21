import torch
import torch.nn as nn
from torchvision import models


model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)

print(model)

dummy_input = torch.randn(
    1,
    3,
    224,
    224
)

model.eval()

with torch.no_grad():
    output = model(dummy_input)

print("Output shape:", output.shape)
print("Output:", output)