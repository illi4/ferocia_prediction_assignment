# Quick Start Guide - Part B: API Deployment

## 🚀 Get the API Running in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements-api.txt
```

### Step 2: Verify Model Files
Make sure these files exist in your project root:
```
✓ model.pkl
✓ preprocessor.pkl
✓ config.yaml (optional)
```

These should have been created in Part A. If they don't exist, run:
```bash
python pipeline.py --data your_data.csv
```

### Step 3: Start the API
```bash
python run_api.py
```

That's it! The API is now running at http://localhost:8000

## ✅ Verify It's Working

### Check Health
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### Make a Test Prediction
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

Expected response:
```json
{
  "prediction": "no",
  "probability": 0.23,
  "confidence": "medium"
}
```

## 📖 Interactive Documentation

Open your browser and go to:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

You can test the API directly from these interfaces!

## 🧪 Run Full Test Suite

Test all endpoints:
```bash
python test_api.py
```

This will verify:
- ✓ API is responding
- ✓ Model is loaded
- ✓ Predictions work
- ✓ Input validation works
- ✓ Error handling works

## 🛠️ Customization

### Change Port
```bash
python run_api.py --port 8080
```

### Enable Auto-Reload (Development)
```bash
python run_api.py --reload
```

### Use Different Model Files
Create a `.env` file:
```bash
MODEL_PATH=path/to/model.pkl
PREPROCESSOR_PATH=path/to/preprocessor.pkl
CONFIG_PATH=path/to/config.yaml
```

## 📁 Project Structure

```
project/
├── model.pkl                  # Trained model (from Part A)
├── preprocessor.pkl           # Fitted preprocessor (from Part A)
├── config.yaml               # Configuration (from Part A)
│
├── api/                      # API package
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── models.py            # Pydantic models
│   ├── predictor.py         # Prediction logic
│   └── config.py            # API configuration
│
├── requirements-api.txt      # API dependencies
├── run_api.py               # Startup script
├── test_api.py              # Test suite
└── README-API.md            # Full documentation
```

## 🔧 Troubleshooting

### "Module 'api' not found"
Make sure you're running from the project root directory.

### "Model file not found"
Run Part A first to generate model.pkl and preprocessor.pkl:
```bash
python pipeline.py --data your_data.csv
```

### "Port already in use"
Change the port:
```bash
python run_api.py --port 8001
```

### "Import error: preprocessing"
The API needs access to the preprocessing module from Part A. Make sure preprocessing.py is in your Python path or copy it to the api directory.

## 📊 What The API Does

### Input (Raw Data)
You send raw customer data as specified in the data dictionary:
```json
{
  "age": 30,
  "job": "technician",
  "marital": "single",
  ...
}
```

### Processing
The API automatically:
1. ✅ Validates input format
2. ✅ Applies feature engineering (pdays → was_contacted_before + days_since_contact)
3. ✅ Removes the 'day' feature
4. ✅ Encodes categorical features
5. ✅ Makes prediction

### Output (Prediction)
You get back:
```json
{
  "prediction": "yes" or "no",
  "probability": 0.0 to 1.0,
  "confidence": "low", "medium", or "high"
}
```

## 🎯 Key Features

- ✅ **Automatic Validation**: Pydantic validates all inputs
- ✅ **Preprocessing Included**: No manual preprocessing needed
- ✅ **Clear Error Messages**: Easy debugging
- ✅ **Interactive Docs**: Swagger UI and ReDoc
- ✅ **MacOS Compatible**: Works on all platforms
- ✅ **Production Ready**: Proper error handling and logging

## 📚 Next Steps

1. Read the full documentation: `README-API.md`
2. Explore the interactive docs: http://localhost:8000/docs
3. Run the test suite: `python test_api.py`
4. Try different input combinations
5. Check the logs for debugging

---

**Happy coding! 🎉**

Need help? Check the full README-API.md for detailed documentation.
