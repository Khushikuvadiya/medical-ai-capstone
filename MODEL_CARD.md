# Model Card — Chest X-ray Pneumonia Classification

## 1. Model Details

**Model name:** ResNet18 Chest X-ray Classifier  
**Project:** Counterfactual and Explainable Medical Imaging Decision Support  
**Framework:** PyTorch  
**Task:** Binary image classification  
**Input:** Chest X-ray image  
**Output classes:**

- NORMAL
- PNEUMONIA

**Class mapping:**

- NORMAL = 0
- PNEUMONIA = 1

**Input image size:** 224 × 224 pixels

The model uses a ResNet18 architecture with ImageNet pretrained weights.

The final fully connected classification layer was replaced with a two-class output layer.

---

## 2. Intended Use

This model is intended for:

- academic research
- explainable AI experimentation
- counterfactual explanation research
- robustness evaluation
- medical imaging AI demonstration
- decision-support workflow prototyping

This model is not intended for:

- clinical diagnosis
- emergency decision making
- treatment recommendation
- replacing radiologists or healthcare professionals
- direct use in real patient-care environments

---

## 3. Dataset

The project uses the Chest X-Ray Images (Pneumonia) dataset.

The dataset contains two classes:

- NORMAL
- PNEUMONIA

### Dataset Split

Training split:

4,172 images

Validation split:

1,044 images

Held-out test set:

624 images

### Test Set Composition

NORMAL:

234 images

PNEUMONIA:

390 images

The original test set was kept unchanged for final evaluation.

---

## 4. Preprocessing

Input images are processed using:

1. Grayscale conversion
2. Conversion to three channels
3. Resize to 224 × 224 pixels
4. Tensor conversion
5. ImageNet normalization

ImageNet normalization values:

```text
Mean:
[0.485, 0.456, 0.406]

Standard deviation:
[0.229, 0.224, 0.225]