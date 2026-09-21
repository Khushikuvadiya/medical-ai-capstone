import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image, ImageEnhance, ImageFilter
from torchvision import models, transforms


# -----------------------------
# SETTINGS
# -----------------------------
IMAGE_PATH = "data/raw/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg"
MODEL_PATH = "outputs/resnet18_baseline.pth"

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
# PREDICTION FUNCTION
# -----------------------------
def predict_image(pil_image):

    tensor = transform(
        pil_image
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

    return (
        predicted_class,
        probabilities[0].cpu().numpy()
    )


# -----------------------------
# LOAD ORIGINAL IMAGE
# -----------------------------
original = Image.open(
    IMAGE_PATH
).convert("L")


# -----------------------------
# CREATE PERTURBATIONS
# -----------------------------
darker = ImageEnhance.Brightness(
    original
).enhance(0.7)

brighter = ImageEnhance.Brightness(
    original
).enhance(1.3)

blurred = original.filter(
    ImageFilter.GaussianBlur(radius=2)
)

low_contrast = ImageEnhance.Contrast(
    original
).enhance(0.7)


# Gaussian noise
noise_array = np.array(
    original
).astype(np.float32)

noise = np.random.normal(
    0,
    12,
    noise_array.shape
)

noisy_array = np.clip(
    noise_array + noise,
    0,
    255
).astype(np.uint8)

noisy = Image.fromarray(
    noisy_array
)


# -----------------------------
# TEST CASES
# -----------------------------
test_images = {
    "Original": original,
    "Darker": darker,
    "Brighter": brighter,
    "Blurred": blurred,
    "Gaussian Noise": noisy,
    "Low Contrast": low_contrast
}


# -----------------------------
# RUN ROBUSTNESS TEST
# -----------------------------
results = []

for name, image in test_images.items():

    predicted_class, probabilities = predict_image(
        image
    )

    normal_prob = probabilities[0]
    pneumonia_prob = probabilities[1]

    results.append({
        "name": name,
        "prediction": class_names[predicted_class],
        "normal_prob": normal_prob,
        "pneumonia_prob": pneumonia_prob
    })

    print(
        f"{name:15s} | "
        f"Prediction: {class_names[predicted_class]:10s} | "
        f"NORMAL: {normal_prob:.4f} | "
        f"PNEUMONIA: {pneumonia_prob:.4f}"
    )


# -----------------------------
# VISUALIZE
# -----------------------------
plt.figure(
    figsize=(15, 10)
)

for index, (
    name,
    image
) in enumerate(
    test_images.items()
):

    result = results[index]

    plt.subplot(
        2,
        3,
        index + 1
    )

    plt.imshow(
        image,
        cmap="gray"
    )

    plt.title(
        f"{name}\n"
        f"{result['prediction']} "
        f"(PNEU {result['pneumonia_prob']:.2%})"
    )

    plt.axis("off")


plt.tight_layout()

plt.savefig(
    "outputs/robustness_examples.png",
    dpi=300
)

plt.show()


# -----------------------------
# CONFIDENCE BAR CHART
# -----------------------------
names = [
    result["name"]
    for result in results
]

pneumonia_probs = [
    result["pneumonia_prob"]
    for result in results
]

plt.figure(
    figsize=(10, 5)
)

plt.bar(
    names,
    pneumonia_probs
)

plt.ylabel(
    "PNEUMONIA Probability"
)

plt.title(
    "Model Robustness Under Image Perturbations"
)

plt.ylim(
    0,
    1
)

plt.xticks(
    rotation=20,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    "outputs/robustness_confidence.png",
    dpi=300
)

plt.show()


print(
    "\nSaved: outputs/robustness_examples.png"
)

print(
    "Saved: outputs/robustness_confidence.png"
)