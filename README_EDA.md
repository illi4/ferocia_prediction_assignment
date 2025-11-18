# Jupyter venv setup and usage (macOS)

This guide covers the minimal steps to create a virtual environment (`venv`), install libraries, and use it as a kernel in Jupyter Notebook.

### 1. Create environment
Navigate to your project folder in Terminal and run:
```bash
python3 -m venv venv
```

### 2. Activate environment

```bash
source venv/bin/activate
```

### 3. Install libraries

```bash
pip install -r requirements.txt
```

### 4. Add venv to Jupyter
This makes your venv selectable as a kernel:
```bash
python -m ipykernel install --user --name="dataset_eda" --display-name="Dataset EDA"
```

### 5. Run Jupyter
Execute: 
```bash
jupyter notebook
```

Inside Jupyter, open the EDA notebook, then go to Kernel > Change kernel and select the one you created. 