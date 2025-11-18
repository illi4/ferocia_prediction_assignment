""" 
This module handles API configuration settings using config.yaml.

# For improvement:
# Pass the specific model path via Environment Variables or a strict config path instead of deriving latest
"""

import yaml
import logging
from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    """
    API Configuration Settings.
    """
    
    # API Settings
    APP_NAME: str = "Marketing Prediction API"
    VERSION: str = "1.0.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = True
    
    # Paths
    CONFIG_PATH: str = "config.yaml"
    
    # These will be determined from config.yaml
    MODEL_PATH: str = ""
    PREPROCESSOR_PATH: str = ""
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    # CORS
    ALLOW_ORIGINS: list = ["*"]
    
    def _load_from_yaml(self):
        """Load configuration from YAML file to determine model paths."""
        config_file = Path(self.CONFIG_PATH)
        if not config_file.exists():
            logger.warning(f"Config file {self.CONFIG_PATH} not found, using defaults")
            return

        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Get API settings from config
            api_config = config.get('api', {})
            pkg_config = config.get('packaging', {})
            
            model_name = api_config.get('model_name', pkg_config.get('model_name', 'marketing_model'))
            model_version = api_config.get('model_version', 'latest')
            models_root = Path(pkg_config.get('model_dir', 'models'))
            
            # Determine specific model directory
            model_dir = None
            
            if model_version == 'latest':
                # Find latest version directory
                if models_root.exists():
                    # Look for directories starting with model_name
                    candidates = sorted([
                        d for d in models_root.iterdir() 
                        if d.is_dir() and d.name.startswith(model_name)
                    ])
                    if candidates:
                        model_dir = candidates[-1]
                        logger.info(f"Selected latest model version: {model_dir.name}")
            else:
                # Construct specific version path (assuming folder naming convention)
                # format is usually {model_name}_{version}
                target_name = f"{model_name}_{model_version}"
                if (models_root / target_name).exists():
                    model_dir = models_root / target_name
            
            if model_dir:
                self.MODEL_PATH = str(model_dir / "xgboost_model.pkl")
                self.PREPROCESSOR_PATH = str(model_dir / "preprocessor.pkl")
            else:
                logger.error(f"Could not locate model for {model_name} version {model_version}")

        except Exception as e:
            logger.error(f"Error parsing config: {e}")

    def model_post_init(self, __context):
        """Called after initialization."""
        self._load_from_yaml()


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()