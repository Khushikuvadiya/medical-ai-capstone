from pathlib import Path

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ChestXrayDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),

            # ImageNet normalization for pretrained ResNet18
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        self.label_map = {
            "NORMAL": 0,
            "PNEUMONIA": 1,
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        image_path = Path(row["image_path"])
        label_name = row["label"]

        image = Image.open(image_path).convert("L")
        image = self.transform(image)

        label = self.label_map[label_name]
        label = torch.tensor(label, dtype=torch.long)

        return image, label