import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from torchvision import models, transforms
from captum.attr import IntegratedGradients


# -----------------------------
# SETTINGS
# -----------------------------
IMAGE_PATH = "data/raw/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg"
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

image = Image.open(
    IMAGE_PATH
).convert("L")

input_tensor = transform(
    image
).unsqueeze(0).to(device)

input_tensor.requires_grad = True


# -----------------------------
# PREDICTION
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

class_names = [
    "NORMAL",
    "PNEUMONIA"
]

confidence = probabilities[
    0,
    predicted_class
].item()

print(
    "Prediction:",
    class_names[predicted_class]
)

print(
    f"Confidence: {confidence:.4f}"
)


# -----------------------------
# INTEGRATED GRADIENTS
# -----------------------------
ig = IntegratedGradients(model)

baseline = torch.zeros_like(
    input_tensor
)

attributions, delta = ig.attribute(
    input_tensor,
    baselines=baseline,
    target=predicted_class,
    n_steps=50,
    return_convergence_delta=True
)

print(
    "Convergence delta:",
    delta.item()
)


# -----------------------------
# PROCESS ATTRIBUTION
# -----------------------------
attribution = attributions.squeeze(
    0
).detach().cpu().numpy()

# Combine RGB attribution channels
attribution = np.mean(
    np.abs(attribution),
    axis=0
)

# Normalize to 0-1
attribution = (
    attribution - attribution.min()
)

if attribution.max() > 0:
    attribution = (
        attribution / attribution.max()
    )


# -----------------------------
# ORIGINAL IMAGE
# -----------------------------
display_image = image.resize(
    (224, 224)
)

display_image = np.array(
    display_image
).astype(np.float32)

display_image = (
    display_image / 255.0
)


# -----------------------------
# DISPLAY
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
    "Original X-ray"
)

plt.axis("off")


plt.subplot(
    1,
    3,
    2
)

plt.imshow(
    attribution,
    cmap="hot"
)

plt.title(
    "Integrated Gradients"
)

plt.axis("off")


plt.subplot(
    1,
    3,
    3
)

plt.imshow(
    display_image,
    cmap="gray"
)

plt.imshow(
    attribution,
    cmap="hot",
    alpha=0.5
)

plt.title(
    f"{class_names[predicted_class]} "
    f"({confidence:.2%})"
)

plt.axis("off")


plt.tight_layout()

plt.savefig(
    "outputs/integrated_gradients_test.png",
    dpi=300
)

plt.show()

print(
    "Saved: outputs/integrated_gradients_test.png"
)