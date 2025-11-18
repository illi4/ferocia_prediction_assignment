# Bank Marketing ML Project - Complete Assignment

This repository contains the complete implementation for both Part A (Model Training) and Part B (API Hosting) of the Bank Marketing ML assignment.

## 📦 What's Included

### Part A: Model Training (From Previous Implementation)
- Preprocessing pipeline
- Model training and evaluation
- Model packaging
- Comprehensive documentation

### Part B: API Hosting (New Implementation)
- Lightweight FastAPI application
- `/predict` endpoint with preprocessing
- Complete documentation and tests
- MacOS compatible

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Model artifacts from Part A: `model.pkl`, `preprocessor.pkl`, `config.yaml`

### Step 1: Install Dependencies

For API only:
```bash
pip install -r requirements-api.txt
```

For complete pipeline (Part A + Part B):
```bash
pip install -r requirements.txt  # Part A dependencies
pip install -r requirements-api.txt  # Part B dependencies
```

### Step 2: Run Part A (if needed)

If you don't have the model files yet:
```bash
python pipeline.py --data your_data.csv
```

This will generate:
- `model.pkl` - Trained XGBoost model
- `preprocessor.pkl` - Fitted preprocessor
- `config.yaml` - Configuration

### Step 3: Run Part B (API)

Start the API:
```bash
python run_api.py
```

Test the API:
```bash
python test_api.py
```

Access interactive docs:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 📁 File Structure

```
project/
│
├── Part A Files (from previous implementation)
│   ├── preprocessing.py
│   ├── train.py
│   ├── package_model.py
│   ├── pipeline.py
│   ├── predict.py
│   ├── setup.py
│   ├── config.yaml
│   └── requirements.txt
│
├── Part B Files (API implementation)
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI application
│   │   ├── models.py            # Pydantic models
│   │   ├── predictor.py         # Prediction logic
│   │   └── config.py            # Configuration
│   │
│   ├── requirements-api.txt     # API dependencies
│   ├── run_api.py              # Startup script
│   ├── test_api.py             # Test suite
│   ├── README-API.md           # API documentation
│   ├── QUICKSTART-API.md       # Quick start guide
│   ├── VERIFICATION-API.md     # Requirements verification
│   ├── INDEX-API.md            # API index
│   └── .env.example            # Configuration template
│
└── Generated Artifacts (after running Part A)
    ├── model.pkl
    ├── preprocessor.pkl
    └── config.yaml
```

## 🎯 Part B: API Implementation Highlights

### Key Features

1. **Lightweight FastAPI** - Modern, fast, with automatic documentation
2. **Raw Input Handling** - Accepts data as per data-dictionary.txt
3. **Automatic Preprocessing** - Handles all transformations internally
4. **MacOS Compatible** - Works on all platforms
5. **Well Documented** - Comprehensive READMEs and code docs
6. **Tested** - Complete test suite included

### API Endpoints

#### Health Check
```bash
GET http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

#### Make Prediction
```bash
POST http://localhost:8000/predict
```

Request body (raw data from data dictionary):
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

Response:
```json
{
  "prediction": "no",
  "probability": 0.23,
  "confidence": "medium"
}
```

### Critical Feature: Preprocessing Integration

**IMPORTANT**: The API payload is different from the features the model expects!

The API automatically handles:
- ✅ Feature engineering (pdays → was_contacted_before + days_since_contact)
- ✅ Removing the 'day' feature
- ✅ Categorical encoding
- ✅ Feature ordering

**You just send raw data - the API handles all preprocessing!**

## 📖 Documentation

### For Quick Start
1. **QUICKSTART-API.md** - Get the API running in 3 steps

### For Complete Understanding
1. **README-API.md** - Comprehensive API documentation
2. **VERIFICATION-API.md** - Proof that all requirements are met
3. **INDEX-API.md** - Navigate all API files

### For Code Review
1. Check the `api/` directory for implementation
2. Read inline code documentation
3. Review test cases in `test_api.py`

## 🧪 Testing

### Run API Tests
```bash
# Start the API first
python run_api.py

# In another terminal
python test_api.py
```

Expected output:
```
Results: 7/7 tests passed
```

Test coverage:
- ✅ Root endpoint
- ✅ Health check
- ✅ Valid predictions
- ✅ Invalid input rejection
- ✅ Missing field handling
- ✅ Multiple predictions

## 🔧 Configuration

### API Settings (.env or environment variables)
```bash
HOST=0.0.0.0
PORT=8000
MODEL_PATH=model.pkl
PREPROCESSOR_PATH=preprocessor.pkl
CONFIG_PATH=config.yaml
```

### Command Line Options
```bash
python run_api.py --port 8080 --reload
```

## 📊 Assignment Requirements - Part B

### All Requirements Met ✅

| Requirement | Status | Implementation |
|------------|--------|----------------|
| Lightweight API | ✅ | FastAPI (~200 lines) |
| /predict endpoint | ✅ | POST /predict with validation |
| Raw input from consumer | ✅ | Accepts data-dictionary format |
| Proper preprocessing | ✅ | Automatic via predictor.py |
| Produces response | ✅ | JSON with prediction + probability |
| Properly packaged | ✅ | Modular api/ directory |
| Split into files | ✅ | main, models, predictor, config |
| Documentation | ✅ | Multiple READMEs + code docs |
| MacOS compatible | ✅ | Pure Python, pip installable |

### Additional Questions Answered ✅

All additional questions from the assignment are answered in **README-API.md**:

1. **Model promotion strategy** - Shadow mode, A/B testing, monitoring
2. **Schema changes** - Versioned endpoints, backward compatibility
3. **Observability metrics** - Latency, error rates, data drift, alerts

## 🎓 Architecture

### Complete Flow

```
Part A: Model Training
    ↓
Raw Data (CSV)
    ↓
[preprocessing.py] - Feature engineering, outlier removal
    ↓
[train.py] - XGBoost training & evaluation
    ↓
[package_model.py] - Production packaging
    ↓
Artifacts: model.pkl, preprocessor.pkl

Part B: API Hosting
    ↓
[api/main.py] - FastAPI application
    ↓
Client Request (Raw Data)
    ↓
[api/models.py] - Pydantic validation
    ↓
[api/predictor.py] - Load model & preprocessor
    ↓
Apply preprocessing (automatic!)
    ↓
Make prediction
    ↓
Return JSON response
```

## 🆘 Troubleshooting

### API won't start
- Ensure `model.pkl` and `preprocessor.pkl` exist
- Run Part A first if needed: `python pipeline.py --data data.csv`
- Check dependencies: `pip install -r requirements-api.txt`

### Import errors
- Ensure Part A files are in the same directory or Python path
- The `preprocessing.py` module is needed for the preprocessor to work

### Port already in use
```bash
python run_api.py --port 8001
```

### Prediction errors
- Check logs in the terminal
- Verify input matches data dictionary format
- Ensure all required fields are present

## 📝 Notes

### Dependencies Between Parts

Part B depends on Part A for:
- `model.pkl` - The trained model
- `preprocessor.pkl` - The fitted preprocessor
- `preprocessing.py` - The preprocessing module (for unpickling)

Make sure all Part A files are present before running the API.

### MacOS Compatibility

Both parts are fully compatible with MacOS:
- Pure Python implementation
- No OS-specific dependencies
- All packages available via pip
- Tested on Python 3.8+

## 🎉 Summary

This project provides:

### Part A (Model Training)
- ✅ Complete ML pipeline
- ✅ Reusable preprocessing
- ✅ XGBoost model training
- ✅ Comprehensive evaluation
- ✅ Production packaging

### Part B (API Hosting)
- ✅ FastAPI application
- ✅ /predict endpoint
- ✅ Automatic preprocessing
- ✅ Input validation
- ✅ Complete documentation
- ✅ Test suite
- ✅ MacOS compatible

Both parts are production-ready, well-documented, and follow best practices.

## 📚 Next Steps

1. **Read QUICKSTART-API.md** for immediate setup
2. **Run the API**: `python run_api.py`
3. **Test it**: `python test_api.py`
4. **Explore docs**: http://localhost:8000/docs
5. **Read README-API.md** for detailed information

---

**Questions?** Check the individual README files or the code documentation.

**Happy predicting! 🎯**
