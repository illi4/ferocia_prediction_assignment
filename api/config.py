""" 
This module handles API configuration settings.
"""

import os
from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    API Configuration Settings.
    
    All settings can be overridden via environment variables.
    For example: export MODEL_PATH="/path/to/model.pkl"
    """
    
    # API Settings
    APP_NAME: str = "Bank Marketing Prediction API"
    VERSION: str = "1.0.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = True  # Auto-reload in development
    
    # Model Paths
    MODEL_PATH: str = "model.pkl"
    PREPROCESSOR_PATH: str = "preprocessor.pkl"
    CONFIG_PATH: str = "config.yaml"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    # CORS
    ALLOW_ORIGINS: list = ["*"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    This function is cached to avoid reloading settings on every request.
    """
    return Settings()
