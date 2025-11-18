# Bank Marketing Prediction API

## Overview

A lightweight FastAPI application for serving the Bank Marketing prediction model. This API provides a `/predict` endpoint that accepts raw customer data and returns predictions about term deposit subscription likelihood.

**Key Features:**
- ✅ RESTful API with automatic OpenAPI documentation
- ✅ Automatic input validation with Pydantic
- ✅ Comprehensive error handling
- ✅ Preprocessing integrated (handles raw input → model features)
- ✅ Health check endpoint
- ✅ MacOS compatible
- ✅ Easy local deployment

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Model artifacts from Part A:
  - `model.pkl` - Trained XGBoost model
  - `preprocessor.pkl` - Fitted preprocessor
  - `config.yaml` - Configuration file (optional)

### Installation

1. **Install dependencies:**
```bash
pip install -r requirements-api.txt
```

2. **Ensure model files are in the project root:**
```
project/
├── model.pkl
├── preprocessor.pkl
├── config.yaml (optional)
└── api/
    ├── __init__.py
    ├── main.py
    ├── models.py
    ├── predictor.py
    └── config.py
```

3. **Run the API:**
```bash
# From the project root directory
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Or use the startup script:
```bash
python run_api.py
```

4. **Verify it's running:**
```bash
curl http://localhost:8000/health
```

You should see:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

## 📡 API Endpoints

### 1. Root Endpoint
```
GET /
```
Returns API information and links to documentation.

### 2. Health Check
```
GET /health
```
Check if the API and model are ready.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### 3. Prediction Endpoint
```
POST /predict
```
Make a prediction for a single customer.

**Request Body:**
```json
{
  "age": 30,
  "job": "technician",
  "marital": "single",
  "education": "secondary",
  "default": "no",
  "balance": 1500,
  "housing": "yes",
  "loan": "no",
  "contact": "cellular",
  "day": 15,
  "month": "may",
  "duration": 180,
  "campaign": 2,
  "pdays": -1,
  "previous": 0,
  "poutcome": "unknown"
}
```

**Response:**
```json
{
  "prediction": "no",
  "probability": 0.23,
  "confidence": "medium"
}
```

**Field Descriptions:**

| Field | Type | Description | Valid Values |
|-------|------|-------------|--------------|
| age | integer | Customer age | 18-100 |
| job | string | Job type | "admin.", "unknown", "unemployed", "management", "housemaid", "entrepreneur", "student", "blue-collar", "self-employed", "retired", "technician", "services" |
| marital | string | Marital status | "married", "divorced", "single" |
| education | string | Education level | "unknown", "secondary", "primary", "tertiary" |
| default | string | Has credit in default? | "yes", "no" |
| balance | integer | Average yearly balance | Any integer |
| housing | string | Has housing loan? | "yes", "no" |
| loan | string | Has personal loan? | "yes", "no" |
| contact | string | Contact type | "unknown", "telephone", "cellular" |
| day | integer | Last contact day | 1-31 |
| month | string | Last contact month | "jan", "feb", ..., "dec" |
| duration | integer | Last contact duration (seconds) | >= 0 |
| campaign | integer | Number of contacts this campaign | >= 1 |
| pdays | integer | Days since last contact | -1 or >= 0 |
| previous | integer | Contacts before this campaign | >= 0 |
| poutcome | string | Previous campaign outcome | "unknown", "other", "failure", "success" |

## 🔧 Configuration

The API can be configured via environment variables or a `.env` file:

```bash
# API Settings
HOST=0.0.0.0
PORT=8000
RELOAD=True

# Model Paths
MODEL_PATH=model.pkl
PREPROCESSOR_PATH=preprocessor.pkl
CONFIG_PATH=config.yaml

# Logging
LOG_LEVEL=INFO
```

## 📖 Interactive Documentation

Once the API is running, you can access:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These provide interactive documentation where you can test the API directly in your browser.

## 🧪 Testing the API

### Using curl

**Health check:**
```bash
curl http://localhost:8000/health
```

**Make a prediction:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 30,
    "job": "technician",
    "marital": "single",
    "education": "secondary",
    "default": "no",
    "balance": 1500,
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "day": 15,
    "month": "may",
    "duration": 180,
    "campaign": 2,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown"
  }'
```

### Using Python

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Make prediction
data = {
    "age": 30,
    "job": "technician",
    "marital": "single",
    "education": "secondary",
    "default": "no",
    "balance": 1500,
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "day": 15,
    "month": "may",
    "duration": 180,
    "campaign": 2,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown"
}

response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
```

### Using the Test Script

Run the provided test script:
```bash
python test_api.py
```

This will test all endpoints and validate the API is working correctly.

## 🏗️ Architecture

### Request Flow

```
Raw Input (API Request)
    ↓
[Pydantic Validation]
    ↓
[ModelPredictor.predict()]
    ↓
[BankMarketingPreprocessor.transform()]
    ↓ (Feature Engineering)
    ↓ - Remove 'day' feature
    ↓ - Split 'pdays' → 'was_contacted_before' + 'days_since_contact'
    ↓ - Apply categorical encoding
    ↓
Preprocessed Features
    ↓
[XGBoost Model.predict()]
    ↓
Prediction + Probability
    ↓
[Format Response]
    ↓
API Response (JSON)
```

### Key Components

1. **`api/main.py`**: FastAPI application with endpoints
2. **`api/models.py`**: Pydantic models for validation
3. **`api/predictor.py`**: Model loading and prediction logic
4. **`api/config.py`**: Configuration management

## 🔒 Error Handling

The API provides clear error messages for common issues:

### 400 Bad Request
Invalid input data format

### 422 Unprocessable Entity
Input validation failed (e.g., invalid job category)

### 500 Internal Server Error
Model prediction error

### 503 Service Unavailable
Model not loaded yet

## 🎯 Important Notes

### Preprocessing

**CRITICAL**: The API payload (raw input) is different from the features the model expects!

The preprocessor automatically handles:
- ✅ Feature engineering (pdays transformation)
- ✅ Removing the 'day' feature
- ✅ Categorical encoding
- ✅ Feature ordering

**You don't need to preprocess the input manually** - just send the raw data as defined in the data dictionary.

### Model Compatibility

The API requires:
- Model trained with the preprocessing pipeline from Part A
- Preprocessor saved with fitted encoders
- Compatible XGBoost version

### MacOS Deployment

This API is fully compatible with MacOS:
- Uses standard Python packages
- No platform-specific dependencies
- FastAPI and uvicorn work natively on MacOS

## 📊 Monitoring and Logging

All requests are logged with:
- Timestamp
- Request details
- Prediction results
- Errors (if any)

Logs are written to stdout and can be redirected to a file:
```bash
python -m uvicorn api.main:app --log-config logging_config.yaml
```

## 🚀 Production Deployment

For production deployment:

1. **Disable debug mode:**
```bash
export RELOAD=False
```

2. **Use production server:**
```bash
gunicorn api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

3. **Add security headers** and **HTTPS**

4. **Set up monitoring** (Prometheus, Grafana, etc.)

5. **Configure CORS** for specific origins

## 📝 Additional Questions (from Assignment)

### 1. Model Promotion Strategy

**Question**: If there is a model already in production, what are your thought process behind how you would promote the model safely and ensure that it is still performant?

**Answer**:
- **Shadow mode**: Deploy new model alongside old, compare predictions
- **A/B testing**: Gradually route traffic (10% → 50% → 100%)
- **Performance monitoring**: Track F1 score, latency, error rate
- **Rollback plan**: Instant rollback if metrics degrade
- **Canary deployment**: Test on small subset first
- **Validation metrics**: Ensure F1 >= threshold before promotion

### 2. Schema Changes

**Question**: You're ready to deploy a new model, but the schema and preprocessing code is different to the current model in production - how would you handle this change? What would you do about the existing consumers of the API?

**Answer**:
- **Versioned endpoints**: `/v1/predict` and `/v2/predict`
- **Backward compatibility**: Keep v1 running with old model
- **Migration period**: Give consumers time to update
- **Documentation**: Clear migration guide
- **Deprecation warnings**: Add headers to v1 responses
- **Adapter pattern**: Convert v1 requests to v2 format if possible

### 3. Observability Metrics

**Question**: In an ideal world, what sort of observability metrics would you have on the API? When would you trigger an alert?

**Answer**:

**Metrics to track:**
- **Latency**: p50, p95, p99 response times
- **Throughput**: Requests per second
- **Error rate**: 4xx and 5xx errors
- **Model metrics**: Prediction distribution, average probability
- **Resource usage**: CPU, memory, disk
- **Data drift**: Input feature distributions

**Alert triggers:**
- Latency > 1 second (p95)
- Error rate > 1%
- Model predictions heavily skewed (> 90% one class)
- Feature values outside training distribution
- Resource usage > 80%
- Health check failures

## 🤝 Contributing

When extending this API:

1. Add new endpoints in `api/main.py`
2. Define request/response models in `api/models.py`
3. Add business logic in `api/predictor.py`
4. Update tests in `test_api.py`
5. Update this README

## 📄 License

This API is part of the Bank Marketing ML project.

## 🆘 Troubleshooting

### Model not loading
- Check that `model.pkl` and `preprocessor.pkl` exist
- Verify file paths in configuration
- Check file permissions

### Import errors
- Ensure `preprocessing.py` from Part A is available
- Install all dependencies: `pip install -r requirements-api.txt`

### Port already in use
- Change port: `uvicorn api.main:app --port 8001`
- Or kill existing process

### Prediction errors
- Check input data format
- Verify all required fields are present
- Check logs for detailed error messages

---

**Happy predicting! 🎯**
