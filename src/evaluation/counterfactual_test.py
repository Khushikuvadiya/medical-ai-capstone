import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from torchvision import models, transforms


# -----------------------------
# SETTINGS
# -----------------------------
IMAGE_PATH = "data/raw/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg"
MODEL_PATH = "outputs/resnet18_baseline.pth"

TARGET_CLASS = 0          # 0 = NORMAL
MAX_STEPS = 300
LEARNING_RATE = 0.01
LAMBDA_DISTANCE = 0.05

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
# IMAGE TRANSFORMS
# -----------------------------
resize = transforms.Resize(
    (224, 224)
)

to_tensor = transforms.ToTensor()

mean = torch.tensor(
    [0.485, 0.456, 0.406],
    device=device
).view(1, 3, 1, 1)

std = torch.tensor(
    [0.229, 0.224, 0.225],
    device=device
).view(1, 3, 1, 1)


# -----------------------------
# LOAD ORIGINAL IMAGE
# -----------------------------
image = Image.open(
    IMAGE_PATH
).convert("L")

image = resize(image)

image_tensor = to_tensor(image)

# Convert grayscale to 3 channels
image_tensor = image_tensor.repeat(
    3,
    1,
    1
)

original_pixels = image_tensor.unsqueeze(
    0
).to(device)


# -----------------------------
# HELPER FUNCTION
# -----------------------------
def normalize(x):
    return (x - mean) / std


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
# CREATE OPTIMIZABLE IMAGE
# -----------------------------
counterfactual_pixels = (
    original_pixels.clone()
    .detach()
    .requires_grad_(True)
)


optimizer = torch.optim.Adam(
    [counterfactual_pixels],
    lr=LEARNING_RATE
)

classification_loss = nn.CrossEntropyLoss()

target = torch.tensor(
    [TARGET_CLASS],
    device=device
)


# -----------------------------
# OPTIMIZATION LOOP
# -----------------------------
for step in range(MAX_STEPS):

    optimizer.zero_grad()

    output = model(
        normalize(counterfactual_pixels)
    )

    class_loss = classification_loss(
        output,
        target
    )

    distance_loss = torch.mean(
        (
            counterfactual_pixels
            - original_pixels
        ) ** 2
    )

    total_loss = (
        class_loss
        + LAMBDA_DISTANCE * distance_loss
    )

    total_loss.backward()

    optimizer.step()

    # Keep image pixel values valid
    with torch.no_grad():

        counterfactual_pixels.clamp_(
            0.0,
            1.0
        )


    if (
        step % 25 == 0
        or step == MAX_STEPS - 1
    ):

        with torch.no_grad():

            probs = torch.softmax(
                model(
                    normalize(
                        counterfactual_pixels
                    )
                ),
                dim=1
            )

            predicted_class = torch.argmax(
                probs,
                dim=1
            ).item()

        print(
            f"Step {step:03d} | "
            f"Loss: {total_loss.item():.4f} | "
            f"NORMAL: {probs[0, 0].item():.4f} | "
            f"PNEUMONIA: {probs[0, 1].item():.4f}"
        )


    # Stop once target class is reached confidently
    with torch.no_grad():

        probs = torch.softmax(
            model(
                normalize(
                    counterfactual_pixels
                )
            ),
            dim=1
        )

        predicted_class = torch.argmax(
            probs,
            dim=1
        ).item()

        target_probability = probs[
            0,
            TARGET_CLASS
        ].item()

    if (
        predicted_class == TARGET_CLASS
        and target_probability >= 0.80
    ):

        print(
            f"\nCounterfactual reached at "
            f"step {step}"
        )

        break


# -----------------------------
# FINAL PREDICTION
# -----------------------------
with torch.no_grad():

    final_output = model(
        normalize(
            counterfactual_pixels
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


# -----------------------------
# CONVERT FOR DISPLAY
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
    counterfactual_pixels[
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
# CHANGE METRICS
# -----------------------------
mean_absolute_change = np.mean(
    difference
)

max_change = np.max(
    difference
)

print(
    f"Mean absolute pixel change: "
    f"{mean_absolute_change:.6f}"
)

print(
    f"Maximum pixel change: "
    f"{max_change:.6f}"
)


# -----------------------------
# SAVE COUNTERFACTUAL IMAGE
# -----------------------------
counterfactual_uint8 = (
    counterfactual_np * 255
).astype(np.uint8)

counterfactual_image = Image.fromarray(
    counterfactual_uint8
)

counterfactual_image.save(
    "outputs/counterfactual_image.png"
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
    original_np,
    cmap="gray"
)

plt.title(
    f"Original\n"
    f"{class_names[original_class]} "
    f"({original_probs[0, original_class].item():.2%})"
)

plt.axis("off")


plt.subplot(
    1,
    3,
    2
)

plt.imshow(
    counterfactual_np,
    cmap="gray"
)

plt.title(
    f"Counterfactual\n"
    f"{class_names[final_class]} "
    f"({final_probs[0, final_class].item():.2%})"
)

plt.axis("off")


plt.subplot(
    1,
    3,
    3
)

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
    "outputs/counterfactual_test.png",
    dpi=300
)

plt.show()


print(
    "\nSaved: outputs/counterfactual_image.png"
)

print(
    "Saved: outputs/counterfactual_test.png"
)