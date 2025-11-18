# QUICK START GUIDE

## 🚀 Get Started in 5 Minutes

### Step 1: Download All Files
Download all files from this conversation to a single directory on your machine.

### Step 2: Set Up Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Validate setup
python setup.py
```

### Step 3: Prepare Your Data
Place your bank marketing CSV file in the project directory.
- Expected format: CSV with columns matching the data dictionary
- Example filename: `bank-marketing.csv`

### Step 4: Run the Pipeline
```bash
# Run complete pipeline
python pipeline.py --data bank-marketing.csv

# This will:
# ✓ Preprocess the data
# ✓ Train XGBoost model
# ✓ Evaluate with F1 score
# ✓ Package for deployment
# ✓ Save all logs and artifacts
```

### Step 5: Check Results
After running, you'll find:
- **Model**: `model.pkl`
- **Preprocessor**: `preprocessor.pkl`
- **Logs**: `logs/training_*.log`
- **Metrics**: `logs/metrics_*.json`
- **Plots**: `logs/*.png`
- **Package**: `models/bank_marketing_model_v*/`

## 📋 File Checklist

Make sure you have all these files:
- [ ] `config.yaml` - Configuration with all thresholds
- [ ] `preprocessing.py` - Data preprocessing module
- [ ] `train.py` - Training and evaluation module
- [ ] `package_model.py` - Model packaging module
- [ ] `pipeline.py` - Main orchestration script
- [ ] `predict.py` - Prediction utility
- [ ] `setup.py` - Setup validation script
- [ ] `requirements.txt` - Dependencies
- [ ] `README.md` - Full documentation
- [ ] `gitignore.txt` - Git ignore patterns (rename to .gitignore)
- [ ] `PROJECT_SUMMARY.md` - This summary

## 🎯 What Each File Does

| File | Purpose |
|------|---------|
| `config.yaml` | **ALL thresholds** (IQR, max values, F1 threshold, model params) |
| `preprocessing.py` | **Reusable** preprocessing (feature engineering, outlier removal, encoding) |
| `train.py` | **Training** with XGBoost + **evaluation** with F1 score + **logging** |
| `package_model.py` | **PyTorch wrapper** + **packaging** for production |
| `pipeline.py` | **Orchestrates** everything (preprocessing → training → evaluation → packaging) |
| `predict.py` | Make **predictions** with packaged models |
| `setup.py` | Validate environment and dependencies |
| `requirements.txt` | Python package dependencies |
| `README.md` | Complete documentation |
| `PROJECT_SUMMARY.md` | Detailed summary of what was built |

## ⚡ Quick Commands

```bash
# Full pipeline
python pipeline.py --data data.csv

# Pipeline without packaging (faster)
python pipeline.py --data data.csv --skip-packaging

# Make predictions
python predict.py --model models/bank_marketing_model_v* --data new_data.csv

# Validate environment
python setup.py
```

## 🔍 Key Features

✅ **Config-driven**: All thresholds in `config.yaml` (no code changes needed)  
✅ **Modular**: Each component can be used independently  
✅ **Reproducible**: Fixed random seeds, saved states  
✅ **Logged**: Complete training logs with F1 evaluation conclusion  
✅ **PyTorch-ready**: TorchScript wrapper for serving  
✅ **Production-grade**: Professional structure and documentation  

## 📊 Understanding the Config

The `config.yaml` file contains ALL configurable parameters:

**Outlier Removal Thresholds** (as specified in next_steps.txt):
- IQR multiplier: 3.0
- Features: age, balance, duration, campaign
- Max previous: 50
- Max days_since_contact: 800

**Model Parameters**:
- XGBoost hyperparameters
- scale_pos_weight: 7.5 (handles class imbalance)

**Evaluation**:
- F1 threshold: 0.40 (model acceptance criterion)
- All metrics tracked

## 🎉 You're Ready!

Just run:
```bash
python pipeline.py --data your_data.csv
```

Check `logs/` for results and `models/` for the packaged model!

For detailed information, see:
- `README.md` - Complete documentation
- `PROJECT_SUMMARY.md` - Design decisions and architecture
