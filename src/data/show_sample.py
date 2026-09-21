from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


split_file = Path("data/train_split.csv")

df = pd.read_csv(split_file)

sample = df.iloc[0]

image_path = Path(sample["image_path"])
label = sample["label"]

image = Image.open(image_path).convert("L")

print("Image path:", image_path)
print("Label:", label)
print("Image size:", image.size)

plt.imshow(image, cmap="gray")
plt.title(f"Label: {label}")
plt.axis("off")
plt.show()