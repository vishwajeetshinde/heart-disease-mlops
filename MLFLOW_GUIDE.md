# MLflow Experiment Tracking Guide

## Overview
This project uses **MLflow** for comprehensive experiment tracking, including:
- ✅ Parameters (hyperparameters, dataset info, model configs)
- ✅ Metrics (accuracy, precision, recall, F1, ROC-AUC)
- ✅ Artifacts (models, plots, reports)
- ✅ Model signatures for deployment

---

## 🚀 Quick Start

### Method 1: Using Python Launcher (Recommended)
```bash
cd notebooks
python run_mlflow_experiments.py
```

This interactive script lets you:
1. Install requirements
2. Run experiments with MLflow tracking
3. Launch MLflow UI
4. Run everything in sequence

### Method 2: Direct Execution
```bash
# Install dependencies
pip install -r requirements.txt

# Run experiments
cd notebooks
python feature_engineering_modeling_mlflow.py

# View results in MLflow UI
mlflow ui
```

Then open your browser to: **http://127.0.0.1:5000**

---

## 📊 What Gets Logged to MLflow

### For Each Model Run:

#### 1. **Parameters** 📝
- Dataset information (samples, features)
- Train-test split ratio
- Cross-validation folds
- Scaler type (StandardScaler)
- Hyperparameter grid
- Best hyperparameters from GridSearchCV

#### 2. **Metrics** 📈
- Cross-validation accuracy (mean & std)
- Cross-validation ROC-AUC
- Train accuracy
- Test accuracy
- Test precision
- Test recall
- Test F1-score
- Test ROC-AUC

#### 3. **Artifacts** 📦
- **Model files** (.pkl format)
- **Confusion matrix** (PNG)
- **ROC curve** (PNG)
- **Classification report** (TXT)
- **Feature importance** (PNG & CSV) - Random Forest only
- **Scaler object** (.pkl)

#### 4. **Models** 🤖
- Registered models with signatures
- Model versioning
- Model metadata

---

## 📂 Project Structure

```
heart-disease-mlops/
├── notebooks/
│   ├── feature_engineering_modeling_mlflow.py  # Main script with MLflow
│   ├── run_mlflow_experiments.py               # Python launcher
│   └── data/
│       └── heart_cleaned.csv                    # Cleaned dataset
├── models/                                       # Saved model files
├── results/                                      # Reports and CSVs
├── screenshots/                                  # Plots and visualizations
├── mlruns/                                       # MLflow tracking data
└── requirements.txt                              # Python dependencies
```

---

## 🎯 Experiments Tracked

### 1. Logistic Regression
- **Hyperparameter Tuning:** C, penalty (L1/L2), solver
- **Cross-validation:** 5-fold stratified
- **Metrics:** Accuracy, Precision, Recall, F1, ROC-AUC
- **Artifacts:** Confusion matrix, ROC curve, classification report

### 2. Random Forest
- **Hyperparameter Tuning:** n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features
- **Cross-validation:** 5-fold stratified
- **Metrics:** Accuracy, Precision, Recall, F1, ROC-AUC
- **Artifacts:** Confusion matrix, ROC curve, feature importance, classification report

---

## 🔍 Using MLflow UI

Once you run `mlflow ui`, you can:

### Compare Experiments
- View all runs side-by-side
- Sort by metrics (accuracy, ROC-AUC, etc.)
- Filter by parameters
- Compare model performance

### View Run Details
- Click any run to see full details
- Review all logged parameters
- Examine metrics over time
- Download artifacts

### Model Registry
- View registered models
- Manage model versions
- Transition models to production
- Add model descriptions and tags

---

## 💡 Alternative Methods to Integrate MLflow

### 1. **Python Script** (Current Method) ✅
```python
import mlflow

with mlflow.start_run():
    mlflow.log_param("param", value)
    mlflow.log_metric("metric", score)
    mlflow.sklearn.log_model(model, "model")
```

**Pros:** Cross-platform, easy to version control, no shell dependencies

### 2. **Jupyter Notebook**
```python
# In notebook cells
%load_ext mlflow
mlflow.set_experiment("my_experiment")
# ... your ML code with mlflow.start_run()
```

**Pros:** Interactive, great for exploration

### 3. **MLflow Projects** (Advanced)
Create `MLproject` file:
```yaml
name: heart-disease

conda_env: conda.yaml

entry_points:
  main:
    parameters:
      max_depth: {type: int, default: 10}
    command: "python train.py --max-depth {max_depth}"
```

**Pros:** Reproducible, environment management

### 4. **Command Line Wrapper**
```bash
mlflow run . -P max_depth=10
```

### 5. **Python Package with Click**
```python
import click
import mlflow

@click.command()
@click.option('--model-type', default='rf')
def train(model_type):
    with mlflow.start_run():
        # training code
        pass
```

---

## 🔧 Configuration Options

### Change Tracking Server
```python
# Remote server
mlflow.set_tracking_uri("http://remote-server:5000")

# Database backend
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Default (local mlruns directory)
mlflow.set_tracking_uri("mlruns")
```

### Custom Experiment Location
```bash
mlflow ui --backend-store-uri /custom/path/to/mlruns
```

### Different Port
```bash
mlflow ui --port 8080
```

---

## 📊 Example MLflow Commands

```bash
# View experiments
mlflow experiments list

# Search runs
mlflow runs list --experiment-id 0

# Compare runs
mlflow runs compare run-id-1 run-id-2

# Download artifacts
mlflow artifacts download --run-id <run-id>

# Serve model
mlflow models serve -m runs:/<run-id>/model -p 5001
```

---

## 🎓 Assignment Requirements Checklist

- ✅ **Experiment Tracking**: MLflow integrated
- ✅ **Log Parameters**: Dataset info, hyperparameters, best params
- ✅ **Log Metrics**: Accuracy, precision, recall, F1, ROC-AUC
- ✅ **Log Artifacts**: Models, plots, reports
- ✅ **Multiple Runs**: Logistic Regression & Random Forest tracked separately
- ✅ **Model Comparison**: Easy comparison in MLflow UI
- ✅ **Reproducibility**: All parameters and models saved

---

## 🐛 Troubleshooting

### MLflow UI won't start
```bash
# Check if port is in use
lsof -i :5000

# Try different port
mlflow ui --port 5001
```

### Can't find mlruns directory
```bash
# Specify path explicitly
mlflow ui --backend-store-uri ./mlruns
```

### Import errors
```bash
# Reinstall requirements
pip install -r requirements.txt --upgrade
```

---

## 📚 Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Tracking API](https://mlflow.org/docs/latest/tracking.html)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)

---

**Author:** Heart Disease MLOps Project  
**Date:** January 2026  
**Purpose:** MLOps Assignment - Task 3 (Experiment Tracking)
