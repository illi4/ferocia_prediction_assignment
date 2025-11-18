#!/usr/bin/env python3
"""
Setup and Validation Script for Bank Marketing ML Pipeline

This script helps with:
- Environment validation
- Dependency checking
- Directory setup
- Configuration validation

Author: ML Engineering Team
Date: November 2025
"""

import sys
import subprocess
from pathlib import Path
import importlib.util


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def check_python_version():
    """Check if Python version is compatible."""
    print_section("Checking Python Version")
    
    required_version = (3, 8)
    current_version = sys.version_info[:2]
    
    print(f"Current Python version: {sys.version}")
    
    if current_version >= required_version:
        print(f"✓ Python version is compatible (>= {required_version[0]}.{required_version[1]})")
        return True
    else:
        print(f"✗ Python version is incompatible (requires >= {required_version[0]}.{required_version[1]})")
        return False


def check_dependencies():
    """Check if all required packages are installed."""
    print_section("Checking Dependencies")
    
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'xgboost': 'xgboost',
        'torch': 'torch',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'yaml': 'pyyaml'
    }
    
    all_installed = True
    
    for import_name, package_name in required_packages.items():
        if importlib.util.find_spec(import_name) is not None:
            # Get version if possible
            try:
                module = __import__(import_name)
                version = getattr(module, '__version__', 'unknown')
                print(f"✓ {package_name}: {version}")
            except:
                print(f"✓ {package_name}: installed")
        else:
            print(f"✗ {package_name}: NOT INSTALLED")
            all_installed = False
    
    return all_installed


def install_dependencies():
    """Install dependencies from requirements.txt."""
    print_section("Installing Dependencies")
    
    requirements_file = Path("requirements.txt")
    
    if not requirements_file.exists():
        print("✗ requirements.txt not found!")
        return False
    
    print("Installing packages from requirements.txt...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("✗ Failed to install dependencies")
        return False


def create_directories():
    """Create necessary directories."""
    print_section("Creating Project Directories")
    
    directories = [
        "logs",
        "models",
        "data"
    ]
    
    for dir_name in directories:
        dir_path = Path(dir_name)
        dir_path.mkdir(exist_ok=True)
        print(f"✓ Created/verified: {dir_name}/")
    
    return True


def validate_config():
    """Validate configuration file."""
    print_section("Validating Configuration")
    
    config_file = Path("config.yaml")
    
    if not config_file.exists():
        print("✗ config.yaml not found!")
        return False
    
    try:
        import yaml
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check for required sections
        required_sections = [
            'general',
            'data',
            'feature_engineering',
            'outlier_removal',
            'model',
            'evaluation'
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in config:
                missing_sections.append(section)
        
        if missing_sections:
            print(f"✗ Missing configuration sections: {', '.join(missing_sections)}")
            return False
        
        print("✓ Configuration file is valid")
        print(f"  Model: {config['model']['algorithm']}")
        print(f"  Random seed: {config['general']['random_seed']}")
        print(f"  F1 threshold: {config['evaluation']['f1_threshold']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error validating config: {e}")
        return False


def validate_modules():
    """Validate that all pipeline modules are present."""
    print_section("Validating Pipeline Modules")
    
    required_modules = [
        'config.yaml',
        'preprocessing.py',
        'train.py',
        'package_model.py',
        'pipeline.py',
        'predict.py',
        'requirements.txt',
        'README.md'
    ]
    
    all_present = True
    
    for module in required_modules:
        module_path = Path(module)
        if module_path.exists():
            print(f"✓ {module}")
        else:
            print(f"✗ {module} - NOT FOUND")
            all_present = False
    
    return all_present


def print_usage_instructions():
    """Print usage instructions."""
    print_section("Setup Complete - Usage Instructions")
    
    print("""
Your environment is ready! Here's how to use the pipeline:

1. PREPARE YOUR DATA
   - Place your bank marketing CSV file in the data/ directory
   - Example: data/bank-marketing.csv

2. RUN THE PIPELINE
   python pipeline.py --data data/bank-marketing.csv

3. CHECK RESULTS
   - Logs: logs/training_*.log
   - Metrics: logs/metrics_*.json
   - Model: model.pkl
   - Packaged model: models/bank_marketing_model_v*/

4. MAKE PREDICTIONS
   python predict.py --model models/bank_marketing_model_v* --data data/new_data.csv

For more information, see README.md
    """)


def main():
    """Main setup flow."""
    print("=" * 80)
    print("Bank Marketing ML Pipeline - Setup & Validation")
    print("=" * 80)
    
    # Track overall status
    all_checks_passed = True
    
    # Check Python version
    if not check_python_version():
        all_checks_passed = False
        print("\n⚠️  Please upgrade to Python 3.8 or higher")
    
    # Validate modules
    if not validate_modules():
        all_checks_passed = False
        print("\n⚠️  Some pipeline modules are missing")
    
    # Check dependencies
    if not check_dependencies():
        print("\n⚠️  Some dependencies are missing")
        response = input("\nWould you like to install dependencies now? (y/n): ")
        
        if response.lower() == 'y':
            if not install_dependencies():
                all_checks_passed = False
        else:
            all_checks_passed = False
            print("Skipping dependency installation")
    
    # Create directories
    create_directories()
    
    # Validate configuration
    if not validate_config():
        all_checks_passed = False
        print("\n⚠️  Configuration validation failed")
    
    # Final status
    print_section("Setup Status")
    
    if all_checks_passed:
        print("✓ ALL CHECKS PASSED")
        print("Your environment is ready to use!")
        print_usage_instructions()
    else:
        print("✗ SOME CHECKS FAILED")
        print("Please address the issues above before running the pipeline")
        sys.exit(1)


if __name__ == "__main__":
    main()
