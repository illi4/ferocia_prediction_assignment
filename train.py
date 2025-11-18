"""
Model Training Module

This module handles model training, evaluation, and saving using XGBoost.
"""

import pandas as pd
import numpy as np
import yaml
import pickle
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Any, List

import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, auc
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BankMarketingTrainer:
    """
    Trainer for Bank Marketing prediction model.
    
    This class handles:
    - Model training with XGBoost
    - Model evaluation with multiple metrics
    - Logging and saving results
    - Feature importance analysis
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize trainer with configuration.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config = self._load_config(config_path)
        self.model = None
        self.feature_names = None
        self.metrics = {}
        
        # Set random seed
        np.random.seed(self.config['general']['random_seed'])
        
        # Set up logging directory
        self.log_dir = Path(self.config['logging']['log_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up file logger
        self._setup_file_logger()
        
        logger.info("Trainer initialized with config from %s", config_path)
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _setup_file_logger(self) -> None:
        """Set up file logger for training logs."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"training_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(self.config['logging']['log_format'])
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.info("=" * 80)
        logger.info("Training session started at %s", timestamp)
        logger.info("=" * 80)
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None
    ) -> xgb.XGBClassifier:
        """
        Train XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            
        Returns:
            Trained XGBoost model
        """
        logger.info("Starting model training...")
        logger.info("Training set size: %d", len(X_train))
        
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        # Get XGBoost parameters from config
        xgb_params = self.config['model']['xgboost'].copy()
        
        # Get early stopping from training config (not a constructor parameter)
        early_stopping = self.config['model'].get('training', {}).get('early_stopping_rounds', None)
        
        # Initialize model with all parameters including eval_metric
        self.model = xgb.XGBClassifier(**xgb_params)
        
        # Prepare evaluation set
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
            logger.info("Validation set size: %d", len(X_val))
        
        # Train model
        logger.info("Training with parameters:")
        for key, value in xgb_params.items():
            logger.info("  %s: %s", key, value)
        
        # Fit the model - only pass eval_set
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        logger.info("Model training completed")
        
        # Only log best_iteration and best_score if early stopping was used
        if early_stopping is not None:
            try:
                logger.info("Best iteration: %d", self.model.best_iteration)
                logger.info("Best score: %.4f", self.model.best_score)
            except AttributeError:
                logger.info("Early stopping was configured but not activated during training")
        else:
            logger.info("Trained for %d iterations (no early stopping)", self.model.n_estimators)
        
        return self.model
    
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        save_results: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate model performance on test set.
        
        Args:
            X_test: Test features
            y_test: Test target
            save_results: Whether to save evaluation results
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        logger.info("=" * 80)
        logger.info("MODEL EVALUATION")
        logger.info("=" * 80)
        logger.info("Test set size: %d", len(X_test))
        
        # Get predictions
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        threshold = self.config['evaluation']['classification_threshold']
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
        }
        
        # Calculate Precision-Recall AUC
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
        metrics['pr_auc'] = auc(recall_curve, precision_curve)
        
        # Store metrics
        self.metrics = metrics
        
        # Log metrics
        logger.info("\nPerformance Metrics:")
        logger.info("-" * 40)
        for metric_name, value in metrics.items():
            logger.info("%s: %.4f", metric_name.upper(), value)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info("\nConfusion Matrix:")
        logger.info("                Predicted No  Predicted Yes")
        logger.info("Actual No       %-13d %-13d", cm[0, 0], cm[0, 1])
        logger.info("Actual Yes      %-13d %-13d", cm[1, 0], cm[1, 1])
        
        # Classification Report
        logger.info("\nDetailed Classification Report:")
        logger.info("\n" + classification_report(y_test, y_pred, 
                                                 target_names=['No', 'Yes']))
        
        # Check F1 threshold
        f1_threshold = self.config['evaluation']['f1_threshold']
        logger.info("\n" + "=" * 80)
        logger.info("MODEL ACCEPTANCE EVALUATION")
        logger.info("=" * 80)
        logger.info("F1 Score: %.4f", metrics['f1'])
        logger.info("F1 Threshold: %.4f", f1_threshold)
        
        if metrics['f1'] >= f1_threshold:
            logger.info("✓ MODEL ACCEPTED - F1 score meets threshold")
            self.metrics['model_accepted'] = True
        else:
            logger.warning("✗ MODEL REJECTED - F1 score below threshold")
            logger.warning("  Current: %.4f | Required: %.4f | Gap: %.4f",
                         metrics['f1'], f1_threshold, f1_threshold - metrics['f1'])
            self.metrics['model_accepted'] = False
        
        # Save results
        if save_results:
            self._save_evaluation_results(metrics, y_test, y_pred, y_pred_proba)
        
        logger.info("=" * 80)
        
        return metrics

    
    def _save_evaluation_results(
        self,
        metrics: Dict[str, float],
        y_test: pd.Series,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> None:
        """Save evaluation results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics
        if self.config['logging']['save_metrics']:
            metrics_file = self.log_dir / f"metrics_{timestamp}.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=4)
            logger.info("Metrics saved to %s", metrics_file)
        
        # Save predictions
        if self.config['logging']['save_predictions']:
            predictions_df = pd.DataFrame({
                'y_true': y_test.values,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            })
            pred_file = self.log_dir / f"predictions_{timestamp}.csv"
            predictions_df.to_csv(pred_file, index=False)
            logger.info("Predictions saved to %s", pred_file)
    
    def get_model(self) -> xgb.XGBClassifier:
        """Get the trained model."""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        return self.model
    
    def get_metrics(self) -> Dict[str, float]:
        """Get evaluation metrics."""
        return self.metrics
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path where to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        
        logger.info("Model saved to %s", filepath)
    
    @classmethod
    def load_model(cls, filepath: str) -> xgb.XGBClassifier:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded XGBoost model
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        logger.info("Model loaded from %s", filepath)
        return model


def train_and_evaluate_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config_path: str = "config.yaml"
) -> Tuple[xgb.XGBClassifier, Dict[str, float]]:
    """
    Convenience function to train and evaluate model.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target
        config_path: Path to configuration file
        
    Returns:
        Tuple of (trained_model, evaluation_metrics)
    """
    trainer = BankMarketingTrainer(config_path)
    
    # Train model
    model = trainer.train(X_train, y_train)
    
    # Evaluate model
    metrics = trainer.evaluate(X_test, y_test)
    
    return model, metrics


if __name__ == "__main__":
    print("=" * 80)
    print("Bank Marketing Model Trainer")
    print("=" * 80)
    print("\nThis module provides model training and evaluation functionality.")
    print("\nTo use:")
    print("1. Create trainer: trainer = BankMarketingTrainer('config.yaml')")
    print("2. Train model: model = trainer.train(X_train, y_train)")
    print("3. Evaluate model: metrics = trainer.evaluate(X_test, y_test)")
    print("4. Save model: trainer.save_model('model.pkl')")
