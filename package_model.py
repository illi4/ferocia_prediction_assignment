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


class XGBoostWrapper(nn.Module):
    """
    PyTorch wrapper for XGBoost model.
    
    This wrapper allows XGBoost models to be used within PyTorch's
    ecosystem, enabling deployment with PyTorch serving frameworks.
    """
    
    def __init__(self, xgboost_model: xgb.XGBClassifier, feature_names: List[str]):
        """
        Initialize the wrapper.
        
        Args:
            xgboost_model: Trained XGBoost classifier
            feature_names: List of feature names in order
        """
        super(XGBoostWrapper, self).__init__()
        self.xgboost_model = xgboost_model
        self.feature_names = feature_names
        self.n_features = len(feature_names)
        
        logger.info("XGBoost wrapper initialized with %d features", self.n_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, n_features)
            
        Returns:
            Output tensor of shape (batch_size, 2) with class probabilities
        """
        # Convert tensor to numpy
        x_np = x.cpu().numpy()
        
        # Get predictions from XGBoost
        proba = self.xgboost_model.predict_proba(x_np)
        
        # Convert back to tensor
        output = torch.from_numpy(proba).float()
        
        return output
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with optional threshold.
        
        Args:
            x: Input tensor
            threshold: Classification threshold
            
        Returns:
            Tuple of (class_predictions, probabilities)
        """
        proba = self.forward(x)
        pred_class = (proba[:, 1] >= threshold).long()
        
        return pred_class, proba


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
        
        # Create README
        self._create_readme(version, metrics)
        
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
    
    def _create_readme(self, version: str, metrics: Dict[str, float]) -> None:
        """Create README file for the model package."""
        readme_content = f"""# Bank Marketing Prediction Model - {version}

## Model Information

**Version:** {version}  
**Created:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Algorithm:** XGBoost Classifier  
**Framework:** XGBoost + PyTorch wrapper

## Performance Metrics

- **Accuracy:** {metrics.get('accuracy', 0):.4f}
- **Precision:** {metrics.get('precision', 0):.4f}
- **Recall:** {metrics.get('recall', 0):.4f}
- **F1 Score:** {metrics.get('f1', 0):.4f}
- **ROC AUC:** {metrics.get('roc_auc', 0):.4f}
- **PR AUC:** {metrics.get('pr_auc', 0):.4f}

## Model Acceptance

**Status:** {'✓ ACCEPTED' if metrics.get('model_accepted', False) else '✗ REJECTED'}

## Package Contents

- `xgboost_model.pkl` - Native XGBoost model
- `pytorch_model.pt` - PyTorch TorchScript wrapped model
- `preprocessor.pkl` - Data preprocessor with fitted encoders
- `config.yaml` - Complete configuration used for training
- `feature_names.json` - List of features in order
- `metrics.json` - Detailed evaluation metrics
- `metadata.json` - Complete model metadata

## Usage

### Loading the Model

```python
import pickle
import torch

# Load XGBoost model
with open('xgboost_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

# Load PyTorch model
pytorch_model = torch.jit.load('pytorch_model.pt')

# Load preprocessor
with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)
```

### Making Predictions

```python
import pandas as pd
import numpy as np

# Prepare new data
new_data = pd.DataFrame({{...}})  # Your raw data

# Preprocess
X_processed = preprocessor.transform(new_data)

# Predict with XGBoost
predictions_xgb = xgb_model.predict(X_processed)
probabilities_xgb = xgb_model.predict_proba(X_processed)

# Predict with PyTorch
X_tensor = torch.tensor(X_processed.values, dtype=torch.float32)
predictions_pytorch = pytorch_model(X_tensor)
```

## Model Details

### Feature Engineering
- Removed 'day' feature
- Split 'pdays' into 'was_contacted_before' and 'days_since_contact'

### Outlier Removal
- Applied 3×IQR rule for: age, balance, duration, campaign
- Removed rows with previous > 50
- Removed rows with days_since_contact > 800

### Class Imbalance Handling
- Used scale_pos_weight parameter in XGBoost
- Applied stratified train-test split

## Deployment Notes

1. **Input Format:** Model expects preprocessed features in the same order as training
2. **Preprocessing:** Always use the provided preprocessor before making predictions
3. **Output:** Returns probabilities for both classes [prob_no, prob_yes]
4. **Threshold:** Default classification threshold is 0.5 (configurable)

## Version History

- **{version}:** Initial production model

## Contact

For questions or issues, contact the ML Engineering team.
"""
        
        filepath = self.package_dir / "README.md"
        
        with open(filepath, 'w') as f:
            f.write(readme_content)
        
        logger.info("✓ README created: %s", filepath.name)


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
    print("Bank Marketing Model Packager")
    print("=" * 80)
    print("\nThis module provides model packaging functionality for deployment.")
    print("\nTo use:")
    print("1. Create packager: packager = ModelPackager('config.yaml')")
    print("2. Package model: package_dir = packager.package_model(model, preprocessor, feature_names, metrics)")
    print("\nThe packaged model includes:")
    print("  - XGBoost model (native format)")
    print("  - PyTorch wrapped model (TorchScript)")
    print("  - Preprocessor with fitted encoders")
    print("  - Configuration and metadata")
    print("  - Feature names and evaluation metrics")
