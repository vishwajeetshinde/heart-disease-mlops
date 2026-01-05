# MLflow Experiment Tracking Setup Guide

## Overview
This guide covers MLflow integration for comprehensive experiment tracking in the Heart Disease Classification project (Task 3 - 5 marks).

## What is Logged to MLflow

### 1. **Parameters**
- Dataset information (train/test split, number of samples)
- Model hyperparameters (C, penalty, n_estimators, max_depth, etc.)
- Cross-validation settings (folds, strategy)
- Random state and reproducibility settings
- Preprocessing methods (scaler type)

### 2. **Metrics**
- Training accuracy
- Test accuracy, precision, recall, F1-score, ROC-AUC
- Cross-validation scores (mean, std, per-fold)
- Overfitting gap (train_acc - test_acc)

### 3. **Artifacts**
- Trained models (.pkl files)
- Confusion matrices (PNG plots)
- ROC curves (PNG plots)
- Feature importance plots (for tree-based models)
- Classification reports (TXT files)
- Scaler objects

### 4. **Models**
- Registered models with signatures
- Model versioning
- Model metadata and lineage

## Installation

### Install MLflow and dependencies:
```bash
pip install mlflow
# OR install all requirements
pip install -r requirements.txt
```

## Running Experiments

### Option 1: Main Feature Engineering Script (with MLflow)
```bash
cd notebooks
python feature_engineering_modeling.py
```
This script runs:
- Logistic Regression with hyperparameter tuning
- Random Forest with hyperparameter tuning
- Both logged to MLflow experiment: `heart-disease-classification`

### Option 2: Comprehensive MLflow Experiments
```bash
cd notebooks
python mlflow_experiments.py
```
This script runs 8 different experiments:
- Logistic Regression (L1, L2, Strong Regularization)
- Random Forest (Small, Medium, Large)
- Gradient Boosting
- Support Vector Machine

All logged to MLflow experiment: `heart-disease-mlops-comprehensive`

## Viewing Results in MLflow UI

### 1. Start MLflow UI:
```bash
cd /Users/VSE18/Desktop/Projects/heart-disease-mlops/notebooks
mlflow ui
```

### 2. Open browser:
```
http://localhost:5000
```

### 3. In the UI you can:
- **Compare runs** side-by-side
- **View all metrics** and parameters
- **Download artifacts** (plots, models, reports)
- **Search and filter** experiments
- **Visualize metric trends**
- **Register models** for deployment

## MLflow Directory Structure

After running experiments, you'll see:
```
notebooks/
├── mlruns/                          # MLflow tracking directory
│   ├── 0/                          # Default experiment
│   ├── <experiment_id>/            # Your experiments
│   │   ├── <run_id>/               # Individual run
│   │   │   ├── artifacts/          # Models, plots, files
│   │   │   ├── metrics/            # Metric values
│   │   │   ├── params/             # Parameter values
│   │   │   └── tags/               # Run metadata
│   └── models/                     # Registered models
```

## Key Features Demonstrated

### ✅ **Parameter Logging**
```python
mlflow.log_param("model_type", "Random Forest")
mlflow.log_param("n_estimators", 100)
mlflow.log_param("max_depth", 10)
```

### ✅ **Metric Logging**
```python
mlflow.log_metric("test_accuracy", 0.85)
mlflow.log_metric("test_roc_auc", 0.92)
mlflow.log_metric("cv_fold_1", 0.84)
```

### ✅ **Artifact Logging**
```python
mlflow.log_artifact("confusion_matrix.png")
mlflow.log_artifact("model.pkl")
mlflow.sklearn.log_model(model, "model")
```

### ✅ **Model Registry**
```python
mlflow.sklearn.log_model(
    model, 
    "model",
    registered_model_name="heart_disease_random_forest"
)
```

## Experiment Comparison

In MLflow UI, you can:

1. **Select multiple runs** from the experiment list
2. Click **"Compare"** button
3. View side-by-side comparison of:
   - All parameters
   - All metrics
   - Confusion matrices
   - ROC curves
   - Feature importance

## Advanced MLflow Commands

### Query Best Run:
```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
experiment = client.get_experiment_by_name("heart-disease-classification")
runs = client.search_runs(
    experiment.experiment_id,
    order_by=["metrics.test_roc_auc DESC"],
    max_results=1
)
best_run = runs[0]
print(f"Best Run ID: {best_run.info.run_id}")
print(f"Best ROC-AUC: {best_run.data.metrics['test_roc_auc']}")
```

### Load Model from MLflow:
```python
import mlflow.sklearn

# Load by run ID
model = mlflow.sklearn.load_model(f"runs:/<run_id>/model")

# Load by model name and version
model = mlflow.sklearn.load_model("models:/heart_disease_random_forest/1")
```

### Export Experiment Data:
```bash
mlflow experiments csv -e <experiment_id> -o results.csv
```

## Assignment Requirements Checklist

✅ **Integrate MLflow** - Done  
✅ **Log parameters** - All hyperparameters logged  
✅ **Log metrics** - Accuracy, precision, recall, ROC-AUC, CV scores  
✅ **Log artifacts** - Models, plots (confusion matrix, ROC curves), reports  
✅ **Log plots** - All visualizations saved and logged  
✅ **Multiple runs** - 2+ models tracked (up to 8 in comprehensive script)  
✅ **Run comparison** - Available via MLflow UI  

## Troubleshooting

### Issue: MLflow UI not starting
```bash
# Check if port 5000 is already in use
lsof -ti:5000

# Use different port
mlflow ui --port 5001
```

### Issue: Cannot find experiment
```bash
# List all experiments
mlflow experiments list

# Search for specific experiment
mlflow experiments search --view all
```

### Issue: Artifacts not logging
- Check write permissions in `mlruns/` directory
- Ensure OUTPUT_DIR exists before logging
- Verify file paths are correct

## Best Practices

1. **Organize experiments** - Use descriptive experiment names
2. **Tag runs** - Add tags for easy filtering (`mlflow.set_tag()`)
3. **Document parameters** - Log all relevant hyperparameters
4. **Version models** - Use model registry for production models
5. **Clean up** - Periodically archive old experiments

## Additional Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Tracking Guide](https://mlflow.org/docs/latest/tracking.html)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)

---

**For Assignment Task 3**: Run either script above and access MLflow UI to demonstrate:
- Parameter logging for all model configurations
- Metric tracking across training runs
- Artifact storage (models, plots, reports)
- Experiment comparison capabilities
