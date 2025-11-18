"""
Pydantic models

This module defines the request and response models for the API using Pydantic.
All input validation is handled automatically by these models based on the
data dictionary specifications.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field, validator


class PredictionRequest(BaseModel):
    """
    Request model for prediction endpoint.
    
    This matches the raw input format from the data dictionary.
    All preprocessing is handled by the API before prediction.
    """
    
    # Numeric features
    age: int = Field(..., ge=18, le=100, description="Customer age")
    balance: int = Field(..., description="Average yearly balance in euros")
    day: int = Field(..., ge=1, le=31, description="Last contact day of the month")
    duration: int = Field(..., ge=0, description="Last contact duration in seconds")
    campaign: int = Field(..., ge=1, description="Number of contacts performed during this campaign")
    pdays: int = Field(..., description="Days since last contact from previous campaign (-1 means not contacted)")
    previous: int = Field(..., ge=0, description="Number of contacts performed before this campaign")
    
    # Categorical features
    job: Literal[
        "admin.", "unknown", "unemployed", "management", "housemaid",
        "entrepreneur", "student", "blue-collar", "self-employed",
        "retired", "technician", "services"
    ] = Field(..., description="Type of job")
    
    marital: Literal["married", "divorced", "single"] = Field(
        ..., description="Marital status"
    )
    
    education: Literal["unknown", "secondary", "primary", "tertiary"] = Field(
        ..., description="Education level"
    )
    
    default: Literal["yes", "no"] = Field(
        ..., description="Has credit in default?"
    )
    
    housing: Literal["yes", "no"] = Field(
        ..., description="Has housing loan?"
    )
    
    loan: Literal["yes", "no"] = Field(
        ..., description="Has personal loan?"
    )
    
    contact: Literal["unknown", "telephone", "cellular"] = Field(
        ..., description="Contact communication type"
    )
    
    month: Literal[
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec"
    ] = Field(..., description="Last contact month of year")
    
    poutcome: Literal["unknown", "other", "failure", "success"] = Field(
        ..., description="Outcome of the previous marketing campaign"
    )
    
    class Config:
        schema_extra = {
            "example": {
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
        }
    
    @validator('pdays')
    def validate_pdays(cls, v):
        """Validate pdays is either -1 or positive."""
        if v != -1 and v < 0:
            raise ValueError('pdays must be -1 (not previously contacted) or >= 0')
        return v


class PredictionResponse(BaseModel):
    """
    Response model for prediction endpoint.
    
    Returns the prediction result with probability and confidence.
    """
    
    prediction: Literal["yes", "no"] = Field(
        ..., description="Predicted outcome: 'yes' if likely to subscribe, 'no' otherwise"
    )
    
    probability: float = Field(
        ..., ge=0.0, le=1.0,
        description="Probability of positive class (subscribing to term deposit)"
    )
    
    confidence: Literal["low", "medium", "high"] = Field(
        ..., description="Confidence level of the prediction"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": "no",
                "probability": 0.23,
                "confidence": "medium"
            }
        }


class HealthResponse(BaseModel):
    """
    Response model for health check endpoint.
    """
    
    status: Literal["healthy", "unhealthy"] = Field(
        ..., description="Overall API health status"
    )
    
    model_loaded: bool = Field(
        ..., description="Whether the model is loaded and ready"
    )
    
    version: str = Field(
        ..., description="API version"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "version": "1.0.0"
            }
        }


class BatchPredictionRequest(BaseModel):
    """
    Request model for batch predictions.
    
    Allows multiple predictions in a single request.
    """
    
    instances: list[PredictionRequest] = Field(
        ..., min_items=1, max_items=100,
        description="List of instances to predict (max 100)"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "instances": [
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
                ]
            }
        }


class BatchPredictionResponse(BaseModel):
    """
    Response model for batch predictions.
    """
    
    predictions: list[PredictionResponse] = Field(
        ..., description="List of prediction results"
    )
    
    count: int = Field(
        ..., description="Number of predictions made"
    )
