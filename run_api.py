"""
Startup script for Marketing Prediction API
"""

import argparse
import sys
import uvicorn
from pathlib import Path
from api.config import get_settings

def check_api_ready():
    """Check if settings resolve to valid files."""
    settings = get_settings()
    
    print("Configuration Check:")
    print(f"  Config File: {settings.CONFIG_PATH}")
    print(f"  Resolved Model Path: {settings.MODEL_PATH}")
    print(f"  Resolved Preprocessor Path: {settings.PREPROCESSOR_PATH}")
    
    if not settings.MODEL_PATH or not Path(settings.MODEL_PATH).exists():
        print("\n✗ Error: Model file not found.")
        print("  Please run 'python pipeline.py --data ...' to generate a model")
        print("  or check config.yaml 'api' and 'packaging' sections.")
        return False
        
    if not settings.PREPROCESSOR_PATH or not Path(settings.PREPROCESSOR_PATH).exists():
        print("\n✗ Error: Preprocessor file not found.")
        return False
        
    print("\n✓ Model artifacts found")
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    
    if not check_api_ready():
        sys.exit(1)
        
    print(f"\nStarting API on http://{args.host}:{args.port}")
    uvicorn.run("api.main:app", host=args.host, port=args.port, reload=args.reload)

if __name__ == "__main__":
    main()