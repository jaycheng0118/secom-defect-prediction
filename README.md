# 🏭 Semiconductor Yield Anomaly Detection & Root Cause Analysis System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-green.svg)](https://lightgbm.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/Serving-FastAPI-teal.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Container-Docker-2496ED.svg)](https://www.docker.com/)

An end-to-end Machine Learning pipeline and inference service designed for early wafer yield anomaly detection and sensor root cause analysis (RCA) in semiconductor manufacturing, built on the **UCI SECOM Dataset**.

---

## 📌 Business & Technical Overview

In advanced semiconductor fabrication, detecting yield anomalies (Defect / Failures) at early inspection stages is critical to preventing defective wafers from proceeding downstream. 

### Key Engineering Challenges:
1. **Extreme Class Imbalance:** Yield failure rate is typically $< 7\%$ (Positive:Negative ratio $\approx$ 1:14).
2. **High-Dimensional Sensor Noise:** 590+ continuous sensor signals containing missing values, multicollinearity, and zero-variance features.
3. **Black-Box Limitation:** Process Engineers (PE) require actionable root causes rather than opaque probability scores.

---

## 🏗️ Architecture & Pipeline Workflow

```mermaid
graph LR
    A[Raw Sensor Data<br/>590 Features] --> B[Data Preprocessing &<br/>Variance Filtering]
    B --> C[Imbalanced Strategy<br/>SMOTE / Focal Weighting]
    C --> D[LightGBM Classifier<br/>Bayesian Optimization]
    D --> E[Explainable AI<br/>SHAP Feature Attribution]
    D --> F[FastAPI Microservice<br/>Real-time Scoring]
