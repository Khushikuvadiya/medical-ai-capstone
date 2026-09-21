# Counterfactual and Explainable Medical Imaging Decision Support

## Project Overview

This project is a research prototype for chest X-ray classification with explainability, counterfactual analysis, robustness evaluation, calibration analysis, API deployment, audit logging, and a clinician-style review interface.

The prototype classifies chest X-ray images into:

- NORMAL
- PNEUMONIA

The system is intended for academic research and demonstration only.

It is not validated for clinical diagnosis, treatment decisions, or patient-care use.

---

## Main Features

- ResNet18 chest X-ray classifier
- Transfer learning using ImageNet pretrained weights
- Grad-CAM explainability
- Integrated Gradients
- Deletion-based faithfulness testing
- Counterfactual explanations
- Robustness testing
- Calibration analysis
- FastAPI backend
- Streamlit user interface
- Reviewer feedback workflow
- Audit logging
- Automated API tests

---

## Dataset

Dataset:

Chest X-Ray Images (Pneumonia)

Original source:

Kaggle dataset by Paul Mooney

Classes:

- NORMAL
- PNEUMONIA

### Data Split

Training split:

4,172 images

Validation split:

1,044 images

Held-out test set:

624 images

Test set composition:

- NORMAL: 234
- PNEUMONIA: 390

The original test set was kept unchanged.

---

## Model

Architecture:

ResNet18

Input size:

224 × 224 pixels

Class mapping:

- NORMAL = 0
- PNEUMONIA = 1

The pretrained convolutional layers were frozen during baseline training and the final classification layer was replaced with a two-class output layer.

Weighted cross-entropy loss was used to help address class imbalance.

---

## Baseline Performance

Held-out test results:

| Metric | Result |
|---|---:|
| Accuracy | 82.85% |
| Precision | 80.04% |
| Recall | 96.67% |
| F1 Score | 87.57% |
| ROC-AUC | 94.34% |

### Confusion Matrix

- True NORMAL: 140
- NORMAL predicted as PNEUMONIA: 94
- PNEUMONIA predicted as NORMAL: 13
- True PNEUMONIA: 377

The model achieved high pneumonia recall, while producing a noticeable number of false-positive pneumonia predictions.

---

## Explainability

### Grad-CAM

Grad-CAM is used to highlight image regions that influenced the model prediction.

These regions should not be interpreted as confirmed disease locations or clinical causation.

### Integrated Gradients

Integrated Gradients was also evaluated as a second attribution method.

### Faithfulness Testing

Deletion-based experiments were performed by masking or blurring highly attributed regions.

The deletion curve was not strictly monotonic, highlighting limitations of perturbation-based explainability evaluation.

---

## Counterfactual Analysis

Three counterfactual methods were evaluated:

1. Direct pixel optimization
2. Smoothness-regularized optimization
3. Low-resolution perturbation optimization

The low-resolution perturbation approach produced the strongest experimental result among the implemented methods.

Example result:

- Target NORMAL probability: 89.54%
- Mean absolute pixel change: 0.026228
- Maximum pixel change: 0.078849
- Fraction of pixels changed above 0.05: 0.1150

Counterfactual images are research artifacts and are not clinically validated medical images.

---

## Robustness Evaluation

The model was evaluated under image perturbations.

| Condition | Accuracy |
|---|---:|
| Original | 82.85% |
| Darker | 69.07% |
| Brighter | 87.50% |
| Blurred | 83.17% |
| Gaussian Noise | 79.33% |
| Low Contrast | 70.35% |

The largest performance degradation occurred for darker and low-contrast images.

---

## Calibration

Baseline calibration:

- Brier Score: 0.1172
- ECE: 0.0679

Temperature scaling was evaluated using the validation set.

Optimal temperature:

0.8129

After temperature scaling:

- Brier Score: 0.1217
- ECE: 0.0865

Temperature scaling did not improve held-out calibration and was therefore not retained as an improvement.

---

## API

The backend uses FastAPI.

Available endpoints:

```text
GET  /
GET  /health
POST /predict
POST /explain
POST /counterfactual
POST /review