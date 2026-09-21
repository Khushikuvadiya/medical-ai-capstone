import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from torchvision import models, transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# -----------------------------
# SETTINGS
# -----------------------------
IMAGE_PATH = "data/raw/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg"
MODEL_PATH = "outputs/resnet18_baseline.pth"

DELETE_PERCENT = 0.20


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
# PREPROCESSING
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

display_image = display_image / 255.0

input_tensor = transform(
    image
).unsqueeze(0).to(device)


# -----------------------------
# ORIGINAL PREDICTION
# -----------------------------
with torch.no_grad():

    original_output = model(
        input_tensor
    )

    original_probabilities = torch.softmax(
        original_output,
        dim=1
    )

    predicted_class = torch.argmax(
        original_probabilities,
        dim=1
    ).item()

    original_confidence = original_probabilities[
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
# CREATE DELETION MASK
# -----------------------------
threshold = np.quantile(
    grayscale_cam,
    1 - DELETE_PERCENT
)

important_mask = (
    grayscale_cam >= threshold
)

masked_image = display_image.copy()

# Replace important region with mid-gray
masked_image[
    important_mask
] = 0.5


# -----------------------------
# PREPARE MASKED IMAGE
# -----------------------------
masked_uint8 = (
    masked_image * 255
).astype(np.uint8)

masked_pil = Image.fromarray(
    masked_uint8
)

masked_tensor = transform(
    masked_pil
).unsqueeze(0).to(device)


# -----------------------------
# MASKED PREDICTION
# -----------------------------
with torch.no_grad():

    masked_output = model(
        masked_tensor
    )

    masked_probabilities = torch.softmax(
        masked_output,
        dim=1
    )

    masked_confidence = masked_probabilities[
        0,
        predicted_class
    ].item()


confidence_drop = (
    original_confidence
    - masked_confidence
)


print(
    f"Masked confidence: "
    f"{masked_confidence:.4f}"
)

print(
    f"Confidence drop: "
    f"{confidence_drop:.4f}"
)


# -----------------------------
# PLOT
# -----------------------------
plt.figure(
    figsize=(15, 5)
)

plt.subplot(
    1,
    3,
    1
)

plt.imshow(
    display_image,
    cmap="gray"
)

plt.title(
    f"Original\n"
    f"{class_names[predicted_class]} "
    f"{original_confidence:.2%}"
)

plt.axis("off")


plt.subplot(
    1,
    3,
    2
)

plt.imshow(
    grayscale_cam,
    cmap="jet"
)

plt.title(
    "Grad-CAM Importance"
)

plt.axis("off")


plt.subplot(
    1,
    3,
    3
)

plt.imshow(
    masked_image,
    cmap="gray"
)

plt.title(
    f"Top {int(DELETE_PERCENT * 100)}% Deleted\n"
    f"Confidence: "
    f"{masked_confidence:.2%}"
)

plt.axis("off")


plt.tight_layout()

plt.savefig(
    "outputs/deletion_test.png",
    dpi=300
)

plt.show()

print(
    "Saved: outputs/deletion_test.png"
)