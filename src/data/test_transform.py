from pathlib import Path

import pandas as pd
from PIL import Image
from torchvision import transforms


df = pd.read_csv("data/train_split.csv")

sample = df.iloc[0]

image_path = Path(sample["image_path"])
label = sample["label"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

image = Image.open(image_path)

tensor = transform(image)

print("Label:", label)
print("Original image:", image.size)
print("Tensor shape:", tensor.shape)
print("Tensor type:", tensor.dtype)
print("Minimum value:", tensor.min().item())
print("Maximum value:", tensor.max().item())