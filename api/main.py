"""
Main application

This FastAPI application provides a /predict endpoint for the marketing model.
It handles raw input from consumers and applies the necessary preprocessing before making predictions.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from api.models import PredictionRequest, PredictionResponse, HealthResponse
from api.predictor import ModelPredictor
from api.config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global predictor instance
predictor: ModelPredictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI app.
    Handles model loading on startup and cleanup on shutdown.
    """
    global predictor
    
    # Startup: Load model and preprocessor
    logger.info("Starting up API...")
    settings = get_settings()
    
    try:
        predictor = ModelPredictor(
            model_path=settings.MODEL_PATH,
            preprocessor_path=settings.PREPROCESSOR_PATH,
            config_path=settings.CONFIG_PATH
        )
        logger.info("✓ Model and preprocessor loaded successfully")
        logger.info(f"  Model: {settings.MODEL_PATH}")
        logger.info(f"  Preprocessor: {settings.PREPROCESSOR_PATH}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    yield
    
    # Shutdown: Cleanup
    logger.info("Shutting down API...")
    predictor = None


# Initialize FastAPI app
app = FastAPI(
    title="Bank Marketing Prediction API",
    description="""
    API for predicting term deposit subscription likelihood based on bank marketing data.
    
    Features:
    POST /predict: Make predictions for single or batch requests
    GET /health: Check API health status

    Input Format:
    The API accepts raw input matching the bank marketing data format.
    All preprocessing is handled automatically.
    """,
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Marketing Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns the API status and model readiness.
    """
    is_ready = predictor is not None and predictor.is_ready()
    
    return HealthResponse(
        status="healthy" if is_ready else "unhealthy",
        model_loaded=is_ready,
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Make prediction for bank marketing campaign.
    
    This endpoint accepts raw customer data and returns a prediction of whether
    the customer is likely to subscribe to a term deposit.
    
    ## Request Body
    Provide customer information with the following fields:
    - age: Customer age (numeric)
    - job: Job type (categorical)
    - marital: Marital status (categorical)
    - education: Education level (categorical)
    - default: Has credit in default? (yes/no)
    - balance: Average yearly balance (numeric)
    - housing: Has housing loan? (yes/no)
    - loan: Has personal loan? (yes/no)
    - contact: Contact communication type (categorical)
    - day: Last contact day of month (numeric)
    - month: Last contact month (categorical)
    - duration: Last contact duration in seconds (numeric)
    - campaign: Number of contacts in this campaign (numeric)
    - pdays: Days since last contact from previous campaign (numeric, -1 if not contacted)
    - previous: Number of contacts before this campaign (numeric)
    - poutcome: Outcome of previous campaign (categorical)
    
    ## Response
    Returns:
    - prediction: "yes" or "no" for term deposit subscription likelihood
    - probability: Probability of positive class (0-1)
    - confidence: Prediction confidence level
    
    ## Example
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
    """
    global predictor
    
    if predictor is None or not predictor.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready. Please try again later."
        )
    
    try:
        # Convert request to dictionary
        input_data = request.dict()
        
        # Make prediction
        result = predictor.predict(input_data)
        
        return PredictionResponse(**result)
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid input data: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error making prediction: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error occurred"}
    )


if __name__ == "__main__":
    import uvicorn
    
    # Get settings
    settings = get_settings()
    
    # Run the API
    logger.info(f"Starting API on {settings.HOST}:{settings.PORT}")
    uvicorn.run(
        "api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD
    )
