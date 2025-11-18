# Quick Start Guide: Bank Marketing Pipeline

Follow these steps to train the model and deploy the API in under 5
minutes.

## 1. Environment Setup

**Prerequisites:** Python 3.8+

1.  Create a virtual environment:

``` bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

2.  Install dependencies (merged requirements for Training + API):

``` bash
pip install -r requirements.txt
pip install -r requirements-api.txt
```

## 2. Part A: train the model (XGBoost)

Run the pipeline to process data, train the model, and generate artifacts in the model folder (`model.pkl`, `preprocessor.pkl` etc.).

``` bash
# Run the orchestration pipeline
python pipeline.py --data data\dataset.csv
```

**Note:** Replace `data\dataset.csv` with the actual path to your dataset.

### Outputs

- Packaged models saved to: `models/`
- Artifacts saved to: `xgboost_model.pkl`, `preprocessor.pkl`, with auxiliary json files.
- Evaluation metrics and plots saved to: `logs/`

## 3. Part B: Run the API

Once the model artifacts exist, launch the FastAPI server.

``` bash
python run_api.py
```

**Server Address:** http://localhost:8000\
**Interactive Docs:** http://localhost:8000/docs

## 4. Test using API for predictions

### Run automated tests

``` bash
python test_api.py
```

### Manual Test (cURL)

``` bash
curl -X POST "http://localhost:8000/predict"   -H "Content-Type: application/json"   -d '{
    "age": 30, "job": "technician", "marital": "single",
    "education": "secondary", "default": "no", "balance": 1500,
    "housing": "yes", "loan": "no", "contact": "cellular",
    "day": 15, "month": "may", "duration": 180, "campaign": 2,
    "pdays": -1, "previous": 0, "poutcome": "unknown"
  }'
```

## Configuration

To change various thresholds or model parameters, edit `config.yaml`.

``` yaml
outlier_removal:
  iqr_multiplier: 3.0
model:
  scale_pos_weight: 7.5
evaluation:
  f1_threshold: 0.40
```

## Extra: EDA 

### Jupyter venv setup and usage (macOS)

This guide covers the minimal steps to create a virtual environment (`venv`), install libraries, and use it as a kernel in Jupyter Notebook.

### 1. Activate environment if not yet activated:

```bash
source venv/bin/activate
```

### 2. Install libraries

```bash
pip install -r requirements-eda.txt
```

### 3. Add venv to Jupyter
This makes your venv selectable as a kernel:
```bash
python -m ipykernel install --user --name="dataset_eda" --display-name="Dataset EDA"
```

### 4. Run Jupyter
Execute: 
```bash
jupyter notebook
```

Inside Jupyter, open the EDA notebook, then go to Kernel > Change kernel and select the one you created. 
