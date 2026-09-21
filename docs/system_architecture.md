# System Architecture and Data Flow

## 1. System Overview

The project implements an end-to-end research prototype for explainable and counterfactual chest X-ray classification.

The system contains:

- a trained ResNet18 classification model
- preprocessing pipeline
- prediction service
- Grad-CAM explanation service
- counterfactual generation service
- FastAPI backend
- Streamlit frontend
- audit logging
- operational monitoring
- reviewer feedback workflow

The system is intended for academic research and demonstration only.

---

## 2. High-Level Architecture

```text
User
  |
  v
Streamlit Frontend
  |
  v
FastAPI Backend
  |
  +-----------------------------+
  |                             |
  v                             v
Prediction Service         Explanation Service
  |                             |
  v                             v
ResNet18 Model             Grad-CAM
  |
  +-----------------------------+
  |
  v
Counterfactual Service
  |
  v
Low-resolution Perturbation Optimization

FastAPI Backend
  |
  +----------------------+
  |                      |
  v                      v
Audit Logging       Monitoring Logging
  |                      |
  v                      v
CSV Files           CSV Files