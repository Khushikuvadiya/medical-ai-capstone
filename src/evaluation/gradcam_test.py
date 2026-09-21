import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from torchvision import models, transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# -----------------------------
# SETTINGS
# -----------------------------
IMAGE_PATH = "data/raw/chest_xray/test/NORMAL/IM-0001-0001.jpeg"
MODEL_PATH = "outputs/resnet18_baseline.pth"


# -----------------------------
# DEVICE
# -----------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# -----------------------------
# LOAD MODEL
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
# IMAGE PREPROCESSING
# -----------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

image = Image.open(IMAGE_PATH).convert("L")

input_tensor = transform(image).unsqueeze(0).to(device)


# -----------------------------
# MODEL PREDICTION
# -----------------------------
with torch.no_grad():
    output = model(input_tensor)

probabilities = torch.softmax(output, dim=1)

predicted_class = torch.argmax(
    probabilities,
    dim=1
).item()

class_names = [
    "NORMAL",
    "PNEUMONIA"
]

print("Prediction:", class_names[predicted_class])
print(
    "Probability:",
    probabilities[0, predicted_class].item()
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
    ClassifierOutputTarget(predicted_class)
]

grayscale_cam = cam(
    input_tensor=input_tensor,
    targets=targets
)

grayscale_cam = grayscale_cam[0]


# -----------------------------
# ORIGINAL IMAGE FOR DISPLAY
# -----------------------------
display_image = image.resize(
    (224, 224)
)

display_image = np.array(
    display_image
).astype(np.float32)

display_image = display_image / 255.0

display_image = np.stack(
    [display_image] * 3,
    axis=-1
)


# -----------------------------
# OVERLAY
# -----------------------------
visualization = show_cam_on_image(
    display_image,
    grayscale_cam,
    use_rgb=True
)


# -----------------------------
# SHOW RESULTS
# -----------------------------
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(display_image)
plt.title("Original X-ray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(visualization)
plt.title(
    f"Grad-CAM: {class_names[predicted_class]}"
)
plt.axis("off")

plt.tight_layout()

plt.savefig(
    "outputs/gradcam_test.png",
    dpi=300
)

plt.show()

print(
    "Saved: outputs/gradcam_test.png"
)