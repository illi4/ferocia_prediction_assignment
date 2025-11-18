"""
Predictor module
"""

import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np
import yaml

# Ensure root directory is in sys.path to import preprocessing.py if needed
root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

logger = logging.getLogger(__name__)


class ModelPredictor:
    """
    Model predictor that handles preprocessing and prediction.
    """
    
    def __init__(
        self,
        model_path: str,
        preprocessor_path: str,
        config_path: Optional[str] = None
    ):
        self.model_path = Path(model_path)
        self.preprocessor_path = Path(preprocessor_path)
        self.config_path = Path(config_path) if config_path else None
        
        self.model = None
        self.preprocessor = None
        self.config = None
        
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load model, preprocessor, and config."""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            if not self.preprocessor_path.exists():
                raise FileNotFoundError(f"Preprocessor file not found: {self.preprocessor_path}")

            # Load model
            logger.info(f"Loading model from {self.model_path}")
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info("✓ Model loaded successfully")
            
            # Load preprocessor
            logger.info(f"Loading preprocessor from {self.preprocessor_path}")
            with open(self.preprocessor_path, 'rb') as f:
                self.preprocessor = pickle.load(f)
            logger.info("✓ Preprocessor loaded successfully")
            
            # Load config if provided
            if self.config_path and self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
            
        except ImportError:
            logger.error("Could not import required modules (e.g. preprocessing).")
            raise
        except Exception as e:
            logger.error(f"Error loading artifacts: {e}")
            raise
    
    def is_ready(self) -> bool:
        return self.model is not None and self.preprocessor is not None
    
    def _validate_input(self, data: Dict[str, Any]):
        required_fields = [
            'age', 'job', 'marital', 'education', 'default', 'balance',
            'housing', 'loan', 'contact', 'day', 'month', 'duration',
            'campaign', 'pdays', 'previous', 'poutcome'
        ]
        missing = [f for f in required_fields if f not in data]
        if missing:
            raise ValueError(f"Missing fields: {', '.join(missing)}")
    
    def _get_confidence(self, probability: float) -> str:
        dist = abs(probability - 0.5)
        if dist >= 0.3: return "high"
        if dist >= 0.15: return "medium"
        return "low"
    
    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self._validate_input(data)
        
        df = pd.DataFrame([data])
        
        try:
            # Use the fitted preprocessor
            X = self.preprocessor.transform(df)
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise ValueError(f"Preprocessing failed: {e}")
        
        try:
            # XGBoost prediction
            prob = self.model.predict_proba(X)[0, 1]
            pred = int(self.model.predict(X)[0])
            
            return {
                "prediction": "yes" if pred == 1 else "no",
                "probability": float(prob),
                "confidence": self._get_confidence(prob)
            }
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise