import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# -----------------------------
# SETTINGS
# -----------------------------
MODEL_PATH = "outputs/resnet18_baseline.pth"
DATASET_PATH = Path("data/raw/chest_xray/test")
OUTPUT_DIR = Path("outputs/gradcam_samples")

SAMPLES_PER_CLASS = 3

random.seed(42)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# -----------------------------
# DEVICE
# -----------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)


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
# TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Grayscale(
        num_output_channels=3
    ),
    transforms.Resize(
        (224, 224)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    ),
])


class_names = [
    "NORMAL",
    "PNEUMONIA"
]


# -----------------------------
# GRAD-CAM
# -----------------------------
target_layers = [
    model.layer4[-1]
]

cam = GradCAM(
    model=model,
    target_layers=target_layers
)


# -----------------------------
# PROCESS EACH CLASS
# -----------------------------
for true_class in class_names:

    folder = (
        DATASET_PATH
        / true_class
    )

    image_files = [
        file
        for file in folder.iterdir()
        if file.is_file()
    ]

    selected_images = random.sample(
        image_files,
        SAMPLES_PER_CLASS
    )

    for image_path in selected_images:

        image = Image.open(
            image_path
        ).convert("L")

        input_tensor = (
            transform(image)
            .unsqueeze(0)
            .to(device)
        )

        with torch.no_grad():

            output = model(
                input_tensor
            )

            probabilities = torch.softmax(
                output,
                dim=1
            )

            predicted_class = torch.argmax(
                probabilities,
                dim=1
            ).item()

            confidence = probabilities[
                0,
                predicted_class
            ].item()


        targets = [
            ClassifierOutputTarget(
                predicted_class
            )
        ]

        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets
        )[0]


        # -------------------------
        # DISPLAY IMAGE
        # -------------------------
        display_image = image.resize(
            (224, 224)
        )

        display_image = np.array(
            display_image
        ).astype(
            np.float32
        )

        display_image /= 255.0

        display_image = np.stack(
            [display_image] * 3,
            axis=-1
        )


        visualization = show_cam_on_image(
            display_image,
            grayscale_cam,
            use_rgb=True
        )


        # -------------------------
        # PLOT
        # -------------------------
        plt.figure(
            figsize=(10, 5)
        )

        plt.subplot(
            1,
            2,
            1
        )

        plt.imshow(
            display_image
        )

        plt.title(
            f"True: {true_class}"
        )

        plt.axis("off")


        plt.subplot(
            1,
            2,
            2
        )

        plt.imshow(
            visualization
        )

        plt.title(
            f"Predicted: "
            f"{class_names[predicted_class]} "
            f"({confidence:.2%})"
        )

        plt.axis("off")

        plt.tight_layout()


        output_name = (
            f"{true_class}_"
            f"{image_path.stem}_gradcam.png"
        )

        save_path = (
            OUTPUT_DIR
            / output_name
        )

        plt.savefig(
            save_path,
            dpi=200
        )

        plt.close()

        print(
            f"Saved: {save_path}"
        )


print(
    "\nGrad-CAM sample generation complete."
)
