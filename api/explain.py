import io
import base64

import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


MODEL_PATH = "outputs/resnet18_baseline.pth"

CLASS_NAMES = [
    "NORMAL",
    "PNEUMONIA"
]


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


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


def create_gradcam(image_bytes):

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("L")

    input_tensor = transform(
        image
    ).unsqueeze(0).to(device)

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


    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )


    result_image = Image.fromarray(
        visualization
    )

    buffer = io.BytesIO()

    result_image.save(
        buffer,
        format="PNG"
    )

    encoded_image = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


    return {
        "prediction":
            CLASS_NAMES[predicted_class],

        "normal_probability":
            float(
                probabilities[0, 0].item()
            ),

        "pneumonia_probability":
            float(
                probabilities[0, 1].item()
            ),

        "gradcam_image_base64":
            encoded_image
    }