import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image, ImageFilter
from torchvision import models, transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# -----------------------------
# SETTINGS
# -----------------------------
IMAGE_PATH = "data/raw/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg"
MODEL_PATH = "outputs/resnet18_baseline.pth"

DELETE_LEVELS = [0, 10, 20, 30, 40, 50]


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
# TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------
# LOAD IMAGE
# -----------------------------
image = Image.open(
    IMAGE_PATH
).convert("L")

resized_image = image.resize(
    (224, 224)
)

display_image = np.array(
    resized_image
).astype(np.float32)

display_image /= 255.0


# -----------------------------
# CREATE BLURRED VERSION
# -----------------------------
blurred_pil = resized_image.filter(
    ImageFilter.GaussianBlur(radius=12)
)

blurred_image = np.array(
    blurred_pil
).astype(np.float32)

blurred_image /= 255.0


input_tensor = transform(
    image
).unsqueeze(0).to(device)


# -----------------------------
# ORIGINAL PREDICTION
# -----------------------------
with torch.no_grad():

    output = model(input_tensor)

    probabilities = torch.softmax(
        output,
        dim=1
    )

    predicted_class = torch.argmax(
        probabilities,
        dim=1
    ).item()

    original_confidence = probabilities[
        0,
        predicted_class
    ].item()


class_names = [
    "NORMAL",
    "PNEUMONIA"
]

print(
    "Predicted class:",
    class_names[predicted_class]
)

print(
    f"Original confidence: "
    f"{original_confidence:.4f}"
)


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

targets = [
    ClassifierOutputTarget(
        predicted_class
    )
]

grayscale_cam = cam(
    input_tensor=input_tensor,
    targets=targets
)[0]


# -----------------------------
# DELETION CURVE
# -----------------------------
confidences = []

for delete_percent in DELETE_LEVELS:

    if delete_percent == 0:

        confidence = original_confidence

    else:

        fraction = delete_percent / 100.0

        threshold = np.quantile(
            grayscale_cam,
            1 - fraction
        )

        important_mask = (
            grayscale_cam >= threshold
        )

        masked_image = display_image.copy()

        # Replace important pixels with blurred pixels
        masked_image[
            important_mask
        ] = blurred_image[
            important_mask
        ]

        masked_uint8 = (
            masked_image * 255
        ).astype(np.uint8)

        masked_pil = Image.fromarray(
            masked_uint8
        )

        masked_tensor = transform(
            masked_pil
        ).unsqueeze(0).to(device)

        with torch.no_grad():

            masked_output = model(
                masked_tensor
            )

            masked_probabilities = torch.softmax(
                masked_output,
                dim=1
            )

            confidence = masked_probabilities[
                0,
                predicted_class
            ].item()

    confidences.append(confidence)

    print(
        f"Deleted {delete_percent}% "
        f"-> Confidence: {confidence:.4f}"
    )


# -----------------------------
# PLOT CURVE
# -----------------------------
plt.figure(figsize=(8, 5))

plt.plot(
    DELETE_LEVELS,
    confidences,
    marker="o"
)

plt.xlabel(
    "Percentage of Important Region Deleted"
)

plt.ylabel(
    "Model Confidence"
)

plt.title(
    f"Blur-Based Deletion Curve - "
    f"{class_names[predicted_class]}"
)

plt.ylim(0, 1)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "outputs/deletion_curve_blur.png",
    dpi=300
)

plt.show()


# -----------------------------
# SAVE NUMERIC RESULTS
# -----------------------------
np.savetxt(
    "outputs/deletion_curve_blur_values.csv",
    np.column_stack(
        [DELETE_LEVELS, confidences]
    ),
    delimiter=",",
    header="deleted_percent,confidence",
    comments=""
)

print(
    "Saved: outputs/deletion_curve_blur.png"
)

print(
    "Saved: outputs/deletion_curve_blur_values.csv"
)