"""
Model Packaging Module for the prediction module
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import json
import yaml
import shutil
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
import logging

import xgboost as xgb

# Suppress TorchScript tracing warnings
warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelPackager:
    """
    Package trained model and artifacts for production deployment.
    
    This class handles:
    - Creating model artifacts
    - Saving preprocessor
    - Saving configuration
    - Creating model metadata
    - Version management
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize packager with configuration.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config = self._load_config(config_path)
        self.package_dir = None
        
        logger.info("Model packager initialized")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def package_model(
        self,
        model: xgb.XGBClassifier,
        preprocessor: Any,
        feature_names: List[str],
        metrics: Dict[str, float],
        version: str = None
    ) -> Path:
        """
        Package all model artifacts for deployment.
        
        Args:
            model: Trained XGBoost model
            preprocessor: Fitted preprocessor
            feature_names: List of feature names
            metrics: Evaluation metrics
            version: Model version (auto-generated if None)
            
        Returns:
            Path to the packaged model directory
        """
        logger.info("=" * 80)
        logger.info("PACKAGING MODEL FOR DEPLOYMENT")
        logger.info("=" * 80)
        
        # Create version string
        if version is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            version = f"v{timestamp}"
        
        # Create package directory
        model_dir = Path(self.config['packaging']['model_dir'])
        model_name = self.config['packaging']['model_name']
        self.package_dir = model_dir / f"{model_name}_{version}"
        self.package_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Package directory: %s", self.package_dir)
        logger.info("Version: %s", version)
        
        # Save items based on configuration
        save_items = self.config['packaging']['save_items']
        
        if 'model' in save_items:
            self._save_xgboost_model(model)
        
        if 'preprocessor' in save_items:
            self._save_preprocessor(preprocessor)
        
        if 'config' in save_items:
            self._save_config()
        
        if 'feature_names' in save_items:
            self._save_feature_names(feature_names)
        
        if 'metrics' in save_items:
            self._save_metrics(metrics)
        
        # Create metadata
        self._create_metadata(version, feature_names, metrics)
        
        logger.info("=" * 80)
        logger.info("MODEL PACKAGING COMPLETED")
        logger.info("=" * 80)
        logger.info("All artifacts saved to: %s", self.package_dir)
        
        return self.package_dir
    
    def _save_xgboost_model(self, model: xgb.XGBClassifier) -> None:
        """Save XGBoost model in native format."""
        filepath = self.package_dir / "xgboost_model.pkl"
        
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info("✓ XGBoost model saved: %s", filepath.name)
    
    def _save_preprocessor(self, preprocessor: Any) -> None:
        """Save preprocessor."""
        filepath = self.package_dir / "preprocessor.pkl"
        
        with open(filepath, 'wb') as f:
            pickle.dump(preprocessor, f)
        
        logger.info("✓ Preprocessor saved: %s", filepath.name)
    
    def _save_config(self) -> None:
        """Save configuration."""
        filepath = self.package_dir / "config.yaml"
        
        with open(filepath, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        logger.info("✓ Configuration saved: %s", filepath.name)
    
    def _save_feature_names(self, feature_names: List[str]) -> None:
        """Save feature names."""
        filepath = self.package_dir / "feature_names.json"
        
        with open(filepath, 'w') as f:
            json.dump(feature_names, f, indent=2)
        
        logger.info("✓ Feature names saved: %s", filepath.name)
    
    def _save_metrics(self, metrics: Dict[str, float]) -> None:
        """Save evaluation metrics."""
        filepath = self.package_dir / "metrics.json"
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logger.info("✓ Metrics saved: %s", filepath.name)
    
    def _create_metadata(
        self,
        version: str,
        feature_names: List[str],
        metrics: Dict[str, float]
    ) -> None:
        """Create model metadata file."""
        metadata = {
            'version': version,
            'created_at': datetime.now().isoformat(),
            'model_type': 'XGBoost Classifier',
            'framework': 'XGBoost + PyTorch',
            'n_features': len(feature_names),
            'feature_names': feature_names,
            'performance_metrics': metrics,
            'model_config': self.config['model']['xgboost'],
            'preprocessing_config': {
                'outlier_removal': self.config['outlier_removal'],
                'feature_engineering': self.config['feature_engineering']
            }
        }
        
        filepath = self.package_dir / "metadata.json"
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        logger.info("✓ Metadata saved: %s", filepath.name)

def package_for_serving(
    model: xgb.XGBClassifier,
    preprocessor: Any,
    feature_names: List[str],
    metrics: Dict[str, float],
    config_path: str = "config.yaml",
    version: str = None
) -> Path:
    """
    Convenience function to package model for serving.
    
    Args:
        model: Trained XGBoost model
        preprocessor: Fitted preprocessor
        feature_names: List of feature names
        metrics: Evaluation metrics
        config_path: Path to configuration file
        version: Model version (auto-generated if None)
        
    Returns:
        Path to packaged model directory
    """
    packager = ModelPackager(config_path)
    package_dir = packager.package_model(model, preprocessor, feature_names, metrics, version)
    
    return package_dir


if __name__ == "__main__":
    print("=" * 80)
    print("Marketing prediction model packager")
    print("=" * 80)
    print("\nThis module provides model packaging functionality for deployment.")
    print("\nTo use:")
    print("1. Create packager: packager = ModelPackager('config.yaml')")
    print("2. Package model: package_dir = packager.package_model(model, preprocessor, feature_names, metrics)")
    print("\nThe packaged model includes:")
    print("  - XGBoost model (native format)")
    print("  - Preprocessor with fitted encoders")
    print("  - Configuration and metadata")
    print("  - Feature names and evaluation metrics")
