"""
Startup script for Bank Marketing Prediction API

This script starts the FastAPI server with appropriate settings.

Usage:
    python run_api.py [--port PORT] [--host HOST] [--reload]

Author: ML Engineer
Date: 2024-11-18
"""

import argparse
import sys
import os
from pathlib import Path


def check_requirements():
    """Check if model files exist."""
    required_files = [
        "model.pkl",
        "preprocessor.pkl"
    ]
    
    missing = []
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)
    
    if missing:
        print("✗ Missing required files:")
        for file in missing:
            print(f"  - {file}")
        print("\nPlease ensure model.pkl and preprocessor.pkl exist in the project root.")
        print("These files should be generated from Part A of the assignment.")
        return False
    
    print("✓ All required files found")
    return True


def check_api_package():
    """Check if api package exists."""
    if not Path("api").exists():
        print("✗ 'api' package not found")
        print("Please ensure the api directory with all modules exists.")
        return False
    
    required_modules = [
        "api/__init__.py",
        "api/main.py",
        "api/models.py",
        "api/predictor.py",
        "api/config.py"
    ]
    
    missing = []
    for module in required_modules:
        if not Path(module).exists():
            missing.append(module)
    
    if missing:
        print("✗ Missing API modules:")
        for module in missing:
            print(f"  - {module}")
        return False
    
    print("✓ API package found")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Start the Bank Marketing Prediction API"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on code changes (for development)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("BANK MARKETING PREDICTION API - STARTUP")
    print("=" * 80)
    
    # Check requirements
    print("\nChecking requirements...")
    if not check_requirements():
        sys.exit(1)
    
    if not check_api_package():
        sys.exit(1)
    
    print("\n✓ All checks passed")
    print("\nStarting API server...")
    print(f"  Host: {args.host}")
    print(f"  Port: {args.port}")
    print(f"  Reload: {args.reload}")
    print("\nAPI will be available at:")
    print(f"  - http://localhost:{args.port}")
    print(f"  - http://localhost:{args.port}/docs (Swagger UI)")
    print(f"  - http://localhost:{args.port}/redoc (ReDoc)")
    print("\nPress CTRL+C to stop the server")
    print("=" * 80)
    print()
    
    # Import and run uvicorn
    try:
        import uvicorn
        
        uvicorn.run(
            "api.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="info"
        )
    except ImportError:
        print("✗ uvicorn not installed")
        print("Please install requirements: pip install -r requirements-api.txt")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nShutting down gracefully...")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
