import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from torchvision import models, transforms

from captum.attr import IntegratedGradients
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


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
# IMAGE
# -----------------------------
image = Image.open(
    IMAGE_PATH
).convert("L")

input_tensor = transform(
    image
).unsqueeze(0).to(device)


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


# -----------------------------
# DISPLAY IMAGE
# -----------------------------
display_image = image.resize(
    (224, 224)
)

display_image = np.array(
    display_image
).astype(np.float32)

display_image /= 255.0

rgb_image = np.stack(
    [display_image] * 3,
    axis=-1
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

gradcam_image = show_cam_on_image(
    rgb_image,
    grayscale_cam,
    use_rgb=True
)


# -----------------------------
# INTEGRATED GRADIENTS
# -----------------------------
input_tensor_ig = input_tensor.clone()

input_tensor_ig.requires_grad = True

ig = IntegratedGradients(model)

baseline = torch.zeros_like(
    input_tensor_ig
)

attributions = ig.attribute(
    input_tensor_ig,
    baselines=baseline,
    target=predicted_class,
    n_steps=50
)

attribution = (
    attributions
    .squeeze(0)
    .detach()
    .cpu()
    .numpy()
)

attribution = np.mean(
    np.abs(attribution),
    axis=0
)

attribution -= attribution.min()

if attribution.max() > 0:
    attribution /= attribution.max()


# -----------------------------
# PLOT
# -----------------------------
plt.figure(
    figsize=(15, 5)
)


plt.subplot(1, 3, 1)

plt.imshow(
    display_image,
    cmap="gray"
)

plt.title("Original X-ray")

plt.axis("off")


plt.subplot(1, 3, 2)

plt.imshow(
    gradcam_image
)

plt.title("Grad-CAM")

plt.axis("off")


plt.subplot(1, 3, 3)

plt.imshow(
    display_image,
    cmap="gray"
)

plt.imshow(
    attribution,
    cmap="hot",
    alpha=0.5
)

plt.title("Integrated Gradients")

plt.axis("off")


plt.suptitle(
    f"Prediction: {class_names[predicted_class]} "
    f"({confidence:.2%})"
)

plt.tight_layout()

plt.savefig(
    "outputs/explanation_comparison.png",
    dpi=300
)

plt.show()

print(
    "Saved: outputs/explanation_comparison.png"
)