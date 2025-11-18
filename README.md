# Bank Marketing Prediction - ML Pipeline

A production-ready machine learning pipeline for predicting bank marketing campaign success (term deposit subscription).

## 📋 Project Overview

This project implements a complete ML pipeline that:
- ✅ Prepares and preprocesses bank marketing data with reproducible transformations
- ✅ Trains an XGBoost classifier with class imbalance handling
- ✅ Evaluates model performance with comprehensive metrics (F1, ROC-AUC, etc.)
- ✅ Packages the model in PyTorch-compatible format for production serving
- ✅ Logs all experiments and maintains full reproducibility

## 🏗️ Project Structure

```
.
├── config.yaml              # Complete configuration for the pipeline
├── preprocessing.py         # Reusable data preprocessing module
├── train.py                # Model training and evaluation module
├── package_model.py        # Model packaging for deployment
├── pipeline.py             # Main orchestration script
├── requirements.txt        # Python dependencies
├── README.md              # This file
│
├── preprocessor.pkl        # Saved preprocessor (after running)
├── model.pkl              # Saved model (after running)
│
├── logs/                  # Training logs and evaluation results
│   ├── training_YYYYMMDD_HHMMSS.log
│   ├── metrics_YYYYMMDD_HHMMSS.json
│   ├── predictions_YYYYMMDD_HHMMSS.csv
│   ├── confusion_matrix_YYYYMMDD_HHMMSS.png
│   └── feature_importance_YYYYMMDD_HHMMSS.png
│
└── models/                # Packaged models for deployment
    └── bank_marketing_model_vYYYYMMDD_HHMMSS/
        ├── xgboost_model.pkl
        ├── pytorch_model.pt
        ├── preprocessor.pkl
        ├── config.yaml
        ├── feature_names.json
        ├── metrics.json
        ├── metadata.json
        └── README.md
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Complete Pipeline

```bash
# Run the full pipeline
python pipeline.py --data path/to/bank-marketing.csv

# Run without model packaging (faster for experimentation)
python pipeline.py --data path/to/bank-marketing.csv --skip-packaging

# Use custom configuration
python pipeline.py --data path/to/bank-marketing.csv --config my_config.yaml
```

### 3. Check Results

After running, you'll find:
- **Model artifacts**: `model.pkl`, `preprocessor.pkl`
- **Training logs**: `logs/training_*.log`
- **Evaluation metrics**: `logs/metrics_*.json`
- **Visualizations**: `logs/confusion_matrix_*.png`, `logs/feature_importance_*.png`
- **Production package**: `models/bank_marketing_model_v*/`

## 📊 Configuration

All parameters are controlled via `config.yaml`:

### Key Configuration Sections:

#### 1. **Outlier Removal**
```yaml
outlier_removal:
  iqr_multiplier: 3.0
  iqr_features:
    - age
    - balance
    - duration
    - campaign
  threshold_removals:
    previous:
      max_value: 50
    days_since_contact:
      max_value: 800
```

#### 2. **Model Parameters**
```yaml
model:
  xgboost:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
    scale_pos_weight: 7.5  # Handles class imbalance
```

#### 3. **Evaluation Thresholds**
```yaml
evaluation:
  primary_metric: "f1"
  f1_threshold: 0.40  # Minimum acceptable F1 score
  classification_threshold: 0.5
```

## 🔧 Module Usage

### Individual Module Usage

#### Preprocessing
```python
from preprocessing import BankMarketingPreprocessor, load_data

# Load data
df = load_data("path/to/data.csv")

# Preprocess
preprocessor = BankMarketingPreprocessor("config.yaml")
X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)

# Save for reuse
preprocessor.save("preprocessor.pkl")

# Use on new data
X_new = preprocessor.transform(new_df)
```

#### Training
```python
from train import BankMarketingTrainer

# Train model
trainer = BankMarketingTrainer("config.yaml")
model = trainer.train(X_train, y_train)

# Evaluate
metrics = trainer.evaluate(X_test, y_test)

# Save model
trainer.save_model("model.pkl")
```

#### Packaging
```python
from package_model import ModelPackager

# Package for deployment
packager = ModelPackager("config.yaml")
package_dir = packager.package_model(
    model=model,
    preprocessor=preprocessor,
    feature_names=preprocessor.feature_names,
    metrics=metrics
)
```

## 📈 Evaluation Metrics

The pipeline evaluates models using:
- **Accuracy**: Overall correctness
- **Precision**: Positive predictive value
- **Recall**: True positive rate
- **F1 Score**: Harmonic mean of precision and recall (PRIMARY METRIC)
- **ROC-AUC**: Area under ROC curve
- **PR-AUC**: Area under Precision-Recall curve

### Model Acceptance Criteria

Models must achieve:
- **F1 Score ≥ 0.40** (configurable in `config.yaml`)

The pipeline logs whether the model is accepted or needs improvement.

## 🔄 Data Preprocessing Steps

The pipeline implements the following preprocessing:

### 1. Feature Engineering
- Remove `day` feature (not meaningful)
- Split `pdays` into:
  - `was_contacted_before`: Binary indicator
  - `days_since_contact`: Days since last contact

### 2. Outlier Removal
- Apply 3×IQR rule for: `age`, `balance`, `duration`, `campaign`
- Remove rows with `previous` > 50
- Remove rows with `days_since_contact` > 800

### 3. Categorical Encoding
- Label encoding for all categorical features
- Keep 'unknown' as separate category

### 4. Target Encoding
- Convert 'yes'/'no' to 1/0

### 5. Train-Test Split
- 80-20 split with stratification

## 🎯 Model Details

### Algorithm: XGBoost Classifier

**Why XGBoost?**
- Handles mixed numerical and categorical features well
- Robust to outliers (tree-based splits)
- Built-in class imbalance handling (`scale_pos_weight`)
- Native support for missing values
- Provides feature importance

### Class Imbalance Handling
- Dataset: 88.3% negative, 11.7% positive
- Solution: `scale_pos_weight = 7.5`
- Stratified train-test split

## 📦 Model Packaging

Models are packaged with:

1. **XGBoost Native Format** (`xgboost_model.pkl`)
   - For direct XGBoost inference
   
2. **PyTorch Wrapper** (`pytorch_model.pt`)
   - TorchScript format for PyTorch serving
   - Compatible with PyTorch ecosystem
   
3. **Preprocessor** (`preprocessor.pkl`)
   - Fitted label encoders
   - Feature transformation logic
   
4. **Metadata** (`metadata.json`)
   - Model version
   - Performance metrics
   - Feature names
   - Training configuration

### Loading Packaged Model

```python
import pickle
import torch

# Load XGBoost model
with open('models/bank_marketing_model_v*/xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load preprocessor
with open('models/bank_marketing_model_v*/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)
```

## 🔍 Logging and Tracking

All training runs are logged:

### Log Files
- **Training logs**: Complete execution trace
- **Metrics**: JSON format for programmatic access
- **Predictions**: CSV with true labels, predictions, and probabilities

### Visualizations
- **Confusion Matrix**: Classification performance breakdown
- **Feature Importance**: Top 15 most important features

## 🛠️ Future Improvements

The configuration file includes notes on future improvements:
- Hyperparameter tuning (GridSearch/Optuna)
- Feature selection (SHAP, RFE)
- Ensemble methods
- SMOTE or other resampling
- Advanced feature engineering
- Alternative algorithms (LightGBM, CatBoost)
- Online learning capabilities
- Data drift detection
- A/B testing framework

## 📝 Reproducibility

The pipeline ensures reproducibility through:
- ✅ Fixed random seeds (set to 42)
- ✅ Complete configuration tracking
- ✅ Version-controlled preprocessing
- ✅ Deterministic train-test splits
- ✅ Saved preprocessor states
