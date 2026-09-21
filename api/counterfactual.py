import base64
import io

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision import models, transforms


MODEL_PATH = "outputs/resnet18_baseline.pth"

CLASS_NAMES = [
    "NORMAL",
    "PNEUMONIA"
]

TARGET_CLASS = 0
MAX_STEPS = 100
LEARNING_RATE = 0.02
LOW_RES_SIZE = 28

LAMBDA_DISTANCE = 0.30
LAMBDA_PERTURBATION = 0.10


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


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


def normalize(x):
    return (x - mean) / std


def image_to_base64(image_array):

    image_uint8 = (
        image_array * 255
    ).clip(
        0,
        255
    ).astype(np.uint8)

    image = Image.fromarray(
        image_uint8
    )

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


def generate_counterfactual(image_bytes):

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("L")

    image = resize(image)

    image_tensor = to_tensor(
        image
    )

    image_tensor = image_tensor.repeat(
        3,
        1,
        1
    )

    original_pixels = image_tensor.unsqueeze(
        0
    ).to(device)


    # Original prediction
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


    # Target opposite class
    target_class = (
        1 - original_class
    )


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
        [target_class],
        device=device
    )


    final_counterfactual = original_pixels.clone()

    for step in range(MAX_STEPS):

        optimizer.zero_grad()

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

        counterfactual = torch.clamp(
            original_pixels + smooth_change,
            0.0,
            1.0
        )

        output = model(
            normalize(
                counterfactual
            )
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
                    normalize(
                        counterfactual
                    )
                ),
                dim=1
            )

            predicted_class = torch.argmax(
                probabilities,
                dim=1
            ).item()

            target_probability = probabilities[
                0,
                target_class
            ].item()


        final_counterfactual = (
            counterfactual.detach()
        )


        if (
            predicted_class == target_class
            and target_probability >= 0.80
        ):

            break


    with torch.no_grad():

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

    mean_change = float(
        np.mean(
            difference
        )
    )

    max_change = float(
        np.max(
            difference
        )
    )

    changed_fraction = float(
        np.mean(
            difference > 0.05
        )
    )


    return {
        "original_prediction":
            CLASS_NAMES[original_class],

        "original_normal_probability":
            float(
                original_probs[0, 0].item()
            ),

        "original_pneumonia_probability":
            float(
                original_probs[0, 1].item()
            ),

        "counterfactual_prediction":
            CLASS_NAMES[final_class],

        "counterfactual_normal_probability":
            float(
                final_probs[0, 0].item()
            ),

        "counterfactual_pneumonia_probability":
            float(
                final_probs[0, 1].item()
            ),

        "mean_absolute_change":
            mean_change,

        "maximum_change":
            max_change,

        "fraction_changed_over_005":
            changed_fraction,

        "counterfactual_image_base64":
            image_to_base64(
                counterfactual_np
            ),

        "difference_image_base64":
            image_to_base64(
                difference
                / max(
                    difference.max(),
                    1e-8
                )
            )
    }