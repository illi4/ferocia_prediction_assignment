"""
Main Pipeline for Bank Marketing Prediction Model

This script orchestrates the complete ML pipeline:
1. Data loading
2. Preprocessing
3. Model training
4. Model evaluation
5. Model packaging

Author: ML Engineering Team
Date: November 2025
"""

import argparse
import sys
from pathlib import Path
import logging

from preprocessing import BankMarketingPreprocessor, load_data
from train import BankMarketingTrainer
from package_model import ModelPackager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_pipeline(
    data_path: str,
    config_path: str = "config.yaml",
    skip_packaging: bool = False
) -> None:
    """
    Run the complete ML pipeline.
    
    Args:
        data_path: Path to the raw data CSV file
        config_path: Path to configuration file
        skip_packaging: If True, skip model packaging step
    """
    logger.info("=" * 80)
    logger.info("BANK MARKETING PREDICTION - ML PIPELINE")
    logger.info("=" * 80)
    
    try:
        # ============================================================================
        # STEP 1: Data Loading
        # ============================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: DATA LOADING")
        logger.info("=" * 80)
        
        df = load_data(data_path)
        logger.info("✓ Data loaded successfully")
        logger.info("  Shape: %s", df.shape)
        logger.info("  Columns: %s", list(df.columns))
        
        # ============================================================================
        # STEP 2: Data Preprocessing
        # ============================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: DATA PREPROCESSING")
        logger.info("=" * 80)
        
        preprocessor = BankMarketingPreprocessor(config_path)
        X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)
        
        logger.info("✓ Preprocessing completed")
        logger.info("  Training set: %s", X_train.shape)
        logger.info("  Test set: %s", X_test.shape)
        logger.info("  Features: %d", len(preprocessor.feature_names))
        
        # Save preprocessor
        preprocessor.save("preprocessor.pkl")
        logger.info("✓ Preprocessor saved to preprocessor.pkl")
        
        # ============================================================================
        # STEP 3: Model Training
        # ============================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: MODEL TRAINING")
        logger.info("=" * 80)
        
        trainer = BankMarketingTrainer(config_path)
        model = trainer.train(X_train, y_train)
        
        logger.info("✓ Model training completed")
        
        # Save model
        trainer.save_model("model.pkl")
        logger.info("✓ Model saved to model.pkl")
        
        # ============================================================================
        # STEP 4: Model Evaluation
        # ============================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: MODEL EVALUATION")
        logger.info("=" * 80)
        
        metrics = trainer.evaluate(X_test, y_test, save_results=True)
        
        logger.info("✓ Model evaluation completed")
        logger.info("\nFinal Metrics Summary:")
        logger.info("-" * 40)
        for metric_name, value in metrics.items():
            if metric_name != 'model_accepted':
                logger.info("  %s: %.4f", metric_name.upper(), value)
        
        # ============================================================================
        # STEP 5: Model Packaging
        # ============================================================================
        if not skip_packaging:
            logger.info("\n" + "=" * 80)
            logger.info("STEP 5: MODEL PACKAGING")
            logger.info("=" * 80)
            
            packager = ModelPackager(config_path)
            package_dir = packager.package_model(
                model=model,
                preprocessor=preprocessor,
                feature_names=preprocessor.feature_names,
                metrics=metrics
            )
            
            logger.info("✓ Model packaging completed")
            logger.info("  Package location: %s", package_dir)
        
        # ============================================================================
        # Pipeline Summary
        # ============================================================================
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info("=" * 80)
        logger.info("✓ All steps completed successfully!")
        logger.info("\nResults:")
        logger.info("  1. Preprocessor: preprocessor.pkl")
        logger.info("  2. Model: model.pkl")
        logger.info("  3. Logs: logs/")
        if not skip_packaging:
            logger.info("  4. Package: %s", package_dir)
        
        logger.info("\nModel Performance:")
        logger.info("  F1 Score: %.4f", metrics['f1'])
        logger.info("  ROC AUC: %.4f", metrics['roc_auc'])
        logger.info("  Status: %s", 
                   "ACCEPTED ✓" if metrics.get('model_accepted', False) else "NEEDS IMPROVEMENT ⚠")
        
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error("\n" + "=" * 80)
        logger.error("PIPELINE FAILED")
        logger.error("=" * 80)
        logger.error("Error: %s", str(e))
        logger.exception("Full traceback:")
        sys.exit(1)


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description="Bank Marketing Prediction ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python pipeline.py --data path/to/data.csv
  
  # Run without packaging
  python pipeline.py --data path/to/data.csv --skip-packaging
  
  # Use custom config
  python pipeline.py --data path/to/data.csv --config custom_config.yaml
        """
    )
    
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to the raw data CSV file'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration YAML file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--skip-packaging',
        action='store_true',
        help='Skip model packaging step'
    )
    
    args = parser.parse_args()
    
    # Validate data file exists
    if not Path(args.data).exists():
        logger.error("Data file not found: %s", args.data)
        sys.exit(1)
    
    # Validate config file exists
    if not Path(args.config).exists():
        logger.error("Config file not found: %s", args.config)
        sys.exit(1)
    
    # Run pipeline
    run_pipeline(
        data_path=args.data,
        config_path=args.config,
        skip_packaging=args.skip_packaging
    )


if __name__ == "__main__":
    main()
