import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from torchvision import models, transforms


# -----------------------------
# SETTINGS
# -----------------------------
IMAGE_PATH = "data/raw/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg"
MODEL_PATH = "outputs/resnet18_baseline.pth"

TARGET_CLASS = 0

MAX_STEPS = 400
LEARNING_RATE = 0.02

LOW_RES_SIZE = 28

LAMBDA_DISTANCE = 0.30
LAMBDA_PERTURBATION = 0.10

class_names = [
    "NORMAL",
    "PNEUMONIA"
]


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
# IMAGE PREPARATION
# -----------------------------
resize = transforms.Resize((224, 224))
to_tensor = transforms.ToTensor()

mean = torch.tensor(
    [0.485, 0.456, 0.406],
    device=device
).view(1, 3, 1, 1)

std = torch.tensor(
    [0.229, 0.224, 0.225],
    device=device
).view(1, 3, 1, 1)


def normalize(x):
    return (x - mean) / std


image = Image.open(
    IMAGE_PATH
).convert("L")

image = resize(image)

image_tensor = to_tensor(image)

image_tensor = image_tensor.repeat(
    3,
    1,
    1
)

original_pixels = (
    image_tensor
    .unsqueeze(0)
    .to(device)
)


# -----------------------------
# ORIGINAL PREDICTION
# -----------------------------
with torch.no_grad():

    original_output = model(
        normalize(original_pixels)
    )

    original_probs = torch.softmax(
        original_output,
        dim=1
    )

    original_class = torch.argmax(
        original_probs,
        dim=1
    ).item()


print(
    "Original prediction:",
    class_names[original_class]
)

print(
    f"Original NORMAL probability: "
    f"{original_probs[0, 0].item():.4f}"
)

print(
    f"Original PNEUMONIA probability: "
    f"{original_probs[0, 1].item():.4f}"
)


# -----------------------------
# LOW-RES PERTURBATION
# -----------------------------
perturbation = torch.zeros(
    1,
    1,
    LOW_RES_SIZE,
    LOW_RES_SIZE,
    device=device,
    requires_grad=True
)


optimizer = torch.optim.Adam(
    [perturbation],
    lr=LEARNING_RATE
)


criterion = nn.CrossEntropyLoss()

target = torch.tensor(
    [TARGET_CLASS],
    device=device
)


# -----------------------------
# OPTIMIZATION
# -----------------------------
for step in range(MAX_STEPS):

    optimizer.zero_grad()


    # Upsample perturbation smoothly
    smooth_change = F.interpolate(
        perturbation,
        size=(224, 224),
        mode="bilinear",
        align_corners=False
    )

    # Same perturbation across 3 channels
    smooth_change = smooth_change.repeat(
        1,
        3,
        1,
        1
    )


    counterfactual = (
        original_pixels
        + smooth_change
    )

    counterfactual = torch.clamp(
        counterfactual,
        0.0,
        1.0
    )


    output = model(
        normalize(counterfactual)
    )


    class_loss = criterion(
        output,
        target
    )


    distance_loss = torch.mean(
        torch.abs(
            counterfactual
            - original_pixels
        )
    )


    perturbation_loss = torch.mean(
        perturbation ** 2
    )


    total_loss = (
        class_loss
        + LAMBDA_DISTANCE * distance_loss
        + LAMBDA_PERTURBATION * perturbation_loss
    )


    total_loss.backward()

    optimizer.step()


    with torch.no_grad():

        probabilities = torch.softmax(
            model(
                normalize(counterfactual)
            ),
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

        target_probability = probabilities[
            0,
            TARGET_CLASS
        ].item()


    if (
        step % 25 == 0
        or step == MAX_STEPS - 1
    ):

        print(
            f"Step {step:03d} | "
            f"Loss: {total_loss.item():.4f} | "
            f"Distance: {distance_loss.item():.4f} | "
            f"NORMAL: {probabilities[0, 0].item():.4f} | "
            f"PNEUMONIA: {probabilities[0, 1].item():.4f}"
        )


    if (
        predicted_class == TARGET_CLASS
        and target_probability >= 0.80
    ):

        print(
            f"\nLow-resolution counterfactual "
            f"reached at step {step}"
        )

        break


# -----------------------------
# FINAL COUNTERFACTUAL
# -----------------------------
with torch.no_grad():

    smooth_change = F.interpolate(
        perturbation,
        size=(224, 224),
        mode="bilinear",
        align_corners=False
    )

    smooth_change = smooth_change.repeat(
        1,
        3,
        1,
        1
    )


    final_counterfactual = torch.clamp(
        original_pixels + smooth_change,
        0.0,
        1.0
    )


    final_output = model(
        normalize(
            final_counterfactual
        )
    )

    final_probs = torch.softmax(
        final_output,
        dim=1
    )

    final_class = torch.argmax(
        final_probs,
        dim=1
    ).item()


# -----------------------------
# ARRAYS
# -----------------------------
original_np = (
    original_pixels[
        0,
        0
    ]
    .detach()
    .cpu()
    .numpy()
)


counterfactual_np = (
    final_counterfactual[
        0,
        0
    ]
    .detach()
    .cpu()
    .numpy()
)


difference = np.abs(
    counterfactual_np
    - original_np
)


# -----------------------------
# METRICS
# -----------------------------
mean_change = np.mean(
    difference
)

max_change = np.max(
    difference
)

changed_fraction = np.mean(
    difference > 0.05
)


print("\nFINAL RESULT")
print("-" * 40)

print(
    "Final prediction:",
    class_names[final_class]
)

print(
    f"Final NORMAL probability: "
    f"{final_probs[0, 0].item():.4f}"
)

print(
    f"Final PNEUMONIA probability: "
    f"{final_probs[0, 1].item():.4f}"
)

print(
    f"Mean absolute pixel change: "
    f"{mean_change:.6f}"
)

print(
    f"Maximum pixel change: "
    f"{max_change:.6f}"
)

print(
    f"Fraction pixels changed > 0.05: "
    f"{changed_fraction:.4f}"
)


# -----------------------------
# SAVE IMAGE
# -----------------------------
output_image = Image.fromarray(
    (
        counterfactual_np * 255
    ).astype(np.uint8)
)

output_image.save(
    "outputs/counterfactual_lowres_image.png"
)


# -----------------------------
# PLOT
# -----------------------------
plt.figure(
    figsize=(15, 5)
)


plt.subplot(1, 3, 1)

plt.imshow(
    original_np,
    cmap="gray"
)

plt.title(
    f"Original\n"
    f"{class_names[original_class]} "
    f"({original_probs[0, original_class].item():.2%})"
)

plt.axis("off")


plt.subplot(1, 3, 2)

plt.imshow(
    counterfactual_np,
    cmap="gray"
)

plt.title(
    f"Low-Resolution Counterfactual\n"
    f"{class_names[final_class]} "
    f"({final_probs[0, final_class].item():.2%})"
)

plt.axis("off")


plt.subplot(1, 3, 3)

plt.imshow(
    difference,
    cmap="hot"
)

plt.title(
    "Absolute Difference"
)

plt.axis("off")


plt.tight_layout()

plt.savefig(
    "outputs/counterfactual_lowres.png",
    dpi=300
)

plt.show()


print(
    "\nSaved: outputs/counterfactual_lowres_image.png"
)

print(
    "Saved: outputs/counterfactual_lowres.png"
)