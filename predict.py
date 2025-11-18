"""
Prediction Utility for the Marketing Model

This script provides a simple interface for making predictions with a trained and packaged model.
"""

import argparse
import pandas as pd
import pickle
import torch
import json
import logging
from pathlib import Path
from typing import Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelPredictor:
    """
    Predictor class for making predictions with packaged models.
    """
    
    def __init__(self, model_dir: str):
        """
        Initialize predictor.
        
        Args:
            model_dir: Path to packaged model directory
        """
        self.model_dir = Path(model_dir)
        
        # Load model
        self.model = self._load_xgboost_model()
        
        # Load preprocessor
        self.preprocessor = self._load_preprocessor()
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        logger.info("Predictor initialized successfully")
        logger.info("Model version: %s", self.metadata.get('version', 'unknown'))
    
    def _load_xgboost_model(self):
        """Load XGBoost model."""
        model_path = self.model_dir / "xgboost_model.pkl"
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        logger.info("XGBoost model loaded from %s", model_path)
        return model
    
    def _load_preprocessor(self):
        """Load preprocessor."""
        preprocessor_path = self.model_dir / "preprocessor.pkl"
        
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
        
        logger.info("Preprocessor loaded from %s", preprocessor_path)
        return preprocessor
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load model metadata."""
        metadata_path = self.model_dir / "metadata.json"
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return metadata
    
    def predict(self, data: pd.DataFrame, return_proba: bool = True) -> pd.DataFrame:
        """
        Make predictions on new data.
        
        Args:
            data: Raw input data (before preprocessing)
            return_proba: Whether to return probabilities
            
        Returns:
            DataFrame with predictions (and optionally probabilities)
        """
        logger.info("Making predictions on %d samples", len(data))
        
        # Preprocess data
        X = self.preprocessor.transform(data)

        # XGBoost prediction
        predictions = self.model.predict(X)

        if return_proba:
            proba = self.model.predict_proba(X)
        
        # Create results dataframe
        results = pd.DataFrame({
            'prediction': predictions,
            'prediction_label': ['yes' if p == 1 else 'no' for p in predictions]
        })
        
        if return_proba:
            results['probability_no'] = proba[:, 0]
            results['probability_yes'] = proba[:, 1]
        
        logger.info("Predictions completed")
        return results
    
    def predict_single(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make prediction on a single sample.
        
        Args:
            sample: Dictionary with feature values
            
        Returns:
            Dictionary with prediction and probability
        """
        # Convert to DataFrame
        df = pd.DataFrame([sample])
        
        # Get prediction
        result = self.predict(df, return_proba=True)
        
        return {
            'prediction': int(result['prediction'].iloc[0]),
            'prediction_label': result['prediction_label'].iloc[0],
            'probability_no': float(result['probability_no'].iloc[0]),
            'probability_yes': float(result['probability_yes'].iloc[0])
        }


def main():
    """Main entry point for prediction script."""
    parser = argparse.ArgumentParser(
        description="Make predictions with trained Bank Marketing model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict on CSV file
  python predict.py --model models/bank_marketing_model_v20241118_143022 --data new_data.csv --output predictions.csv
  
  # Use PyTorch model
  python predict.py --model models/bank_marketing_model_v20241118_143022 --data new_data.csv --pytorch
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to packaged model directory'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to input data CSV file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='predictions.csv',
        help='Path to save predictions (default: predictions.csv)'
    )
    
    parser.add_argument(
        '--pytorch',
        action='store_true',
        help='Use PyTorch model instead of XGBoost'
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.model).exists():
        logger.error("Model directory not found: %s", args.model)
        return
    
    if not Path(args.data).exists():
        logger.error("Data file not found: %s", args.data)
        return
    
    # Load predictor
    logger.info("Loading model from %s", args.model)
    predictor = ModelPredictor(args.model, use_pytorch=args.pytorch)
    
    # Load data
    logger.info("Loading data from %s", args.data)
    data = pd.read_csv(args.data)
    logger.info("Data shape: %s", data.shape)
    
    # Make predictions
    predictions = predictor.predict(data, return_proba=True)
    
    # Combine with original data
    output = pd.concat([data, predictions], axis=1)
    
    # Save results
    output.to_csv(args.output, index=False)
    logger.info("Predictions saved to %s", args.output)
    
    # Print summary
    print("\n" + "=" * 80)
    print("PREDICTION SUMMARY")
    print("=" * 80)
    print(f"Total samples: {len(predictions)}")
    print(f"Predicted 'yes': {(predictions['prediction'] == 1).sum()} ({(predictions['prediction'] == 1).sum() / len(predictions) * 100:.2f}%)")
    print(f"Predicted 'no': {(predictions['prediction'] == 0).sum()} ({(predictions['prediction'] == 0).sum() / len(predictions) * 100:.2f}%)")
    print("=" * 80)


if __name__ == "__main__":
    main()
