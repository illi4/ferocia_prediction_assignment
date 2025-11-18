"""
Predictor module

This module handles model loading, preprocessing, and prediction logic.
It ensures that raw input data is properly preprocessed before making predictions.

IMPORTANT: The input payload from the API is different from the features the model expects.
The preprocessor handles all necessary transformations including:
- Feature engineering (pdays -> was_contacted_before + days_since_contact)
- Removing the 'day' feature
- Outlier handling
- Categorical encoding
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np
import yaml

logger = logging.getLogger(__name__)


class ModelPredictor:
    """
    Model predictor that handles preprocessing and prediction.
    
    This class loads the trained model and preprocessor, and provides
    a predict() method that accepts raw input data and returns predictions.
    """
    
    def __init__(
        self,
        model_path: str,
        preprocessor_path: str,
        config_path: Optional[str] = None
    ):
        """
        Initialize the predictor with model and preprocessor.
        
        Args:
            model_path: Path to the trained model pickle file
            preprocessor_path: Path to the fitted preprocessor pickle file
            config_path: Optional path to config file for additional settings
        """
        self.model_path = Path(model_path)
        self.preprocessor_path = Path(preprocessor_path)
        self.config_path = Path(config_path) if config_path else None
        
        self.model = None
        self.preprocessor = None
        self.config = None
        
        # Load model and preprocessor
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load model, preprocessor, and config."""
        try:
            # Load model
            logger.info(f"Loading model from {self.model_path}")
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info("✓ Model loaded successfully")
            
            # Load preprocessor
            logger.info(f"Loading preprocessor from {self.preprocessor_path}")
            with open(self.preprocessor_path, 'rb') as f:
                preprocessor_data = pickle.load(f)
            
            # Handle different preprocessor formats
            if isinstance(preprocessor_data, dict):
                # Preprocessor was saved as a state dictionary
                # Need to reconstruct the BankMarketingPreprocessor
                logger.info("Preprocessor is a dictionary, reconstructing...")
                
                # Import the preprocessing module
                try:
                    from preprocessing import BankMarketingPreprocessor
                    
                    # Create a new instance
                    self.preprocessor = BankMarketingPreprocessor.__new__(BankMarketingPreprocessor)
                    
                    # Restore the state
                    self.preprocessor.config = preprocessor_data.get('config', {})
                    self.preprocessor.label_encoders = preprocessor_data.get('label_encoders', {})
                    self.preprocessor.feature_names = preprocessor_data.get('feature_names', [])
                    self.preprocessor.fitted = preprocessor_data.get('fitted', False)
                    
                    logger.info("✓ Preprocessor reconstructed from state dictionary")
                except ImportError:
                    logger.error("Could not import preprocessing module")
                    raise ImportError(
                        "preprocessing.py module not found. "
                        "Please ensure preprocessing.py from Part A is in the same directory."
                    )
            else:
                # Preprocessor is already an object
                self.preprocessor = preprocessor_data
                logger.info("✓ Preprocessor loaded successfully")
            
            # Load config if provided
            if self.config_path and self.config_path.exists():
                logger.info(f"Loading config from {self.config_path}")
                with open(self.config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
                logger.info("✓ Config loaded successfully")
            
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading artifacts: {e}")
            raise
    
    def is_ready(self) -> bool:
        """
        Check if the predictor is ready to make predictions.
        
        Returns:
            True if model and preprocessor are loaded, False otherwise
        """
        return self.model is not None and self.preprocessor is not None
    
    def _validate_input(self, data: Dict[str, Any]):
        """
        Validate input data has all required fields.
        
        Args:
            data: Input data dictionary
            
        Raises:
            ValueError: If required fields are missing
        """
        required_fields = [
            'age', 'job', 'marital', 'education', 'default', 'balance',
            'housing', 'loan', 'contact', 'day', 'month', 'duration',
            'campaign', 'pdays', 'previous', 'poutcome'
        ]
        
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
    
    def _get_confidence_level(self, probability: float) -> str:
        """
        Determine confidence level based on probability.
        
        Args:
            probability: Predicted probability (0-1)
            
        Returns:
            Confidence level: 'low', 'medium', or 'high'
        """
        # Confidence is based on how far the probability is from 0.5 (decision boundary)
        distance_from_boundary = abs(probability - 0.5)
        
        if distance_from_boundary >= 0.3:  # >= 0.8 or <= 0.2
            return "high"
        elif distance_from_boundary >= 0.15:  # >= 0.65 or <= 0.35
            return "medium"
        else:
            return "low"
    
    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make prediction for a single instance.
        
        This method:
        1. Validates the input data
        2. Converts to DataFrame
        3. Applies preprocessing (feature engineering, outlier removal, encoding)
        4. Makes prediction using the model
        5. Returns formatted result with probability and confidence
        
        Args:
            data: Dictionary containing raw input data matching the data dictionary
            
        Returns:
            Dictionary containing:
                - prediction: 'yes' or 'no'
                - probability: float between 0 and 1
                - confidence: 'low', 'medium', or 'high'
                
        Raises:
            ValueError: If input data is invalid
        """
        # Validate input
        self._validate_input(data)
        
        # Convert to DataFrame (preprocessor expects DataFrame)
        df = pd.DataFrame([data])
        
        logger.info(f"Raw input shape: {df.shape}")
        logger.info(f"Raw input columns: {df.columns.tolist()}")
        
        # Apply preprocessing
        # The preprocessor will:
        # 1. Remove 'day' feature
        # 2. Transform 'pdays' into 'was_contacted_before' and 'days_since_contact'
        # 3. Apply outlier removal rules (but won't remove during prediction)
        # 4. Apply categorical encoding
        try:
            # Check if preprocessor has transform method
            if not hasattr(self.preprocessor, 'transform'):
                raise AttributeError(
                    "Preprocessor does not have 'transform' method. "
                    "This usually means the preprocessor wasn't loaded correctly. "
                    "Please ensure you have preprocessing.py from Part A in your directory."
                )
            
            X = self.preprocessor.transform(df)
            logger.info(f"Preprocessed shape: {X.shape}")
            logger.info(f"Preprocessed columns: {X.columns.tolist()}")
        except AttributeError as e:
            logger.error(f"Preprocessor attribute error: {e}")
            raise ValueError(
                f"Error preprocessing data: {str(e)}. "
                "Make sure preprocessing.py from Part A is available."
            )
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            raise ValueError(f"Error preprocessing data: {str(e)}")
        
        # Make prediction
        try:
            # Get probability for positive class (subscribes = yes)
            probability = self.model.predict_proba(X)[0, 1]
            
            # Get binary prediction
            prediction_binary = self.model.predict(X)[0]
            
            # Convert to yes/no
            prediction = "yes" if prediction_binary == 1 else "no"
            
            # Determine confidence level
            confidence = self._get_confidence_level(probability)
            
            logger.info(f"Prediction: {prediction}, Probability: {probability:.4f}, Confidence: {confidence}")
            
            return {
                "prediction": prediction,
                "probability": float(probability),
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    def predict_batch(self, data_list: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        """
        Make predictions for multiple instances.
        
        Args:
            data_list: List of input data dictionaries
            
        Returns:
            List of prediction results
        """
        results = []
        
        for i, data in enumerate(data_list):
            try:
                result = self.predict(data)
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting instance {i}: {e}")
                # Add error result
                results.append({
                    "prediction": "error",
                    "probability": 0.0,
                    "confidence": "low",
                    "error": str(e)
                })
        
        return results
    
    def get_feature_names(self) -> list[str]:
        """
        Get the names of features after preprocessing.
        
        Returns:
            List of feature names
        """
        if hasattr(self.preprocessor, 'feature_names'):
            return self.preprocessor.feature_names
        return []
