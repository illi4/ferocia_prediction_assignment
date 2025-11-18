# Marketing prediction system

A production-ready Machine Learning pipeline and REST API to predict term deposit subscriptions.

## System architecture

The system is divided into two modular components: 
1. **Model Pipeline (Part A):** Handles data cleaning, training, and artifact generation. 
2. **Serving API (Part B):** A FastAPI service that hosts the trained model and handles real-time preprocessing.

### Workflow

1.  **Raw data** -\> `pipeline.py`
2.  **Preprocessing** (Cleaning, Feature Engineering, Encoding) -\>
    `preprocessor.pkl`
3.  **Training** (XGBoost) -\> `xgboost_model.pkl`
4.  **Evaluation** (F1 Score Check)
5.  **Deployment** (FastAPI) -\> Loads artifacts -\> Serves Predictions

## Part A: Model Pipeline

The training pipeline uses **XGBoost** to handle the classification task, chosen for its robustness against tabular data and built-in handling of class imbalance.

### Key Features

-   **Config-driven:** All thresholds (outliers, hyperparameters) are defined in `config.yaml`.
-   **Preprocessing:**
    -   *Feature engineering:* Splits `pdays` into binary `was_contacted` and numeric `days_since`.
    -   *Outlier removal:* Applies 3xIQR rule for most numeric columns. Remove rows with `previous` > 50 and rows with `days_since_contact` > 800. 
    -   *Encoding:* Label encoding for categorical variables.
-   **Class imbalance:** Handled via `scale_pos_weight` parameter in XGBoost.
-   **Evaluation:** Uses F1-Score as the primary metric with an automatic acceptance/rejection threshold.

## Part B: Serving API

The API is built with **FastAPI** to provide a lightweight and high-performance serving layer.

-   **Endpoint:** `POST /predict`
-   **Input:** Accepts raw data dictionary format; 
-   **Intelligence:** The API automatically applies the *exact* preprocessing steps used in training (via `predictor.py`). The client does not need to pre-encode data.
-   **Validation:** Uses Pydantic models to enforce data types and ranges (e.g., age 18-100).
-   **Output:** JSON containing `prediction` (yes/no), `probability` (0-1), and `confidence` level.

## Project Structure

-   `config.yaml`: Master configuration for Training & API
-   `setup.py`: Setup validation
-   `pipeline.py`: Orchestrator for Model Training
-   `preprocessing.py`: Reusable data cleaning logic
-   `train.py`: XGBoost training logic
-   `package_model.py`: Artifact bundler (Pickle/PyTorch)
-   `run_api.py`: API Startup Script
-   `test_api.py`: API Integration Tests
-   `api/`: Source Code for the API
    -   `main.py`: FastAPI Routes
    -   `models.py`: Request/Response Schemas
    -   `predictor.py`: Inference & Preprocessing Logic
-   `logs/`: Training metrics  
