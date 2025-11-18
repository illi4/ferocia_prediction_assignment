"""
Main Pipeline for the prediction model

This script orchestrates the complete ML pipeline:
1. Data loading
2. Preprocessing
3. Model training
4. Model evaluation
5. Model packaging
"""

import argparse
import sys
import shutil
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
    """
    logger.info("=" * 80)
    logger.info("BANK MARKETING PREDICTION - ML PIPELINE")
    logger.info("=" * 80)
    
    temp_dir = Path("temp_artifacts")
    
    try:
        # Create temp directory for intermediate artifacts
        temp_dir.mkdir(exist_ok=True)
        
        # ============================================================================
        # STEP 1: Data Loading
        # ============================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: DATA LOADING")
        logger.info("=" * 80)
        
        df = load_data(data_path)
        logger.info("✓ Data loaded successfully")
        
        # ============================================================================
        # STEP 2: Data Preprocessing
        # ============================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: DATA PREPROCESSING")
        logger.info("=" * 80)
        
        preprocessor = BankMarketingPreprocessor(config_path)
        X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)
        
        logger.info("✓ Preprocessing completed")
        
        # Save preprocessor to temp
        preprocessor_path = temp_dir / "preprocessor.pkl"
        preprocessor.save(str(preprocessor_path))
        logger.info("✓ Preprocessor saved to temp location")
        
        # ============================================================================
        # STEP 3: Model Training
        # ============================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: MODEL TRAINING")
        logger.info("=" * 80)
        
        trainer = BankMarketingTrainer(config_path)
        model = trainer.train(X_train, y_train)
        
        logger.info("✓ Model training completed")
        
        # Save model to temp
        model_path = temp_dir / "xgboost_model.pkl"
        trainer.save_model(str(model_path))
        logger.info("✓ Model saved to temp location")
        
        # ============================================================================
        # STEP 4: Model Evaluation
        # ============================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: MODEL EVALUATION")
        logger.info("=" * 80)
        
        metrics = trainer.evaluate(X_test, y_test, save_results=True)
        
        logger.info("✓ Model evaluation completed")
        
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
        else:
            package_dir = "Skipped"
        
        # ============================================================================
        # Pipeline Summary
        # ============================================================================
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info("=" * 80)
        
        if not skip_packaging:
            logger.info(f"✓ Model Version: {package_dir.name if hasattr(package_dir, 'name') else package_dir}")
            logger.info(f"✓ Artifacts Location: {package_dir}")
        
        logger.info(f"✓ F1 Score: {metrics['f1']:.4f}")
        logger.info(f"✓ Status: {'ACCEPTED' if metrics.get('model_accepted') else 'NEEDS IMPROVEMENT'}")
        
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
    finally:
        # Cleanup temp directory
        if temp_dir.exists():
            logger.info("Cleaning up temporary artifacts...")
            try:
                shutil.rmtree(temp_dir)
                logger.info("✓ Temp cleanup complete")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp directory: {e}")


def main():
    parser = argparse.ArgumentParser(description="Bank Marketing Prediction ML Pipeline")
    parser.add_argument('--data', type=str, required=True, help='Path to raw data CSV')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--skip-packaging', action='store_true', help='Skip model packaging')
    
    args = parser.parse_args()
    
    if not Path(args.data).exists():
        logger.error(f"Data file not found: {args.data}")
        sys.exit(1)
    
    if not Path(args.config).exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)
    
    run_pipeline(args.data, args.config, args.skip_packaging)


if __name__ == "__main__":
    main()