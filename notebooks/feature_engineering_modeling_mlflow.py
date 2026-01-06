"""
Feature Engineering & Model Development with MLflow Tracking
Heart Disease UCI Dataset
MLOps Assignment - Task 2 & 3

This script performs:
- Feature scaling and encoding
- Model training (Logistic Regression & Random Forest)
- Hyperparameter tuning with GridSearchCV
- Cross-validation
- Model evaluation with comprehensive metrics
- MLflow experiment tracking (parameters, metrics, artifacts, plots)
"""

import os

import joblib
import matplotlib.pyplot as plt

# MLflow imports
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

# -----------------------------------
# 1. CONFIGURATION
# -----------------------------------
DATA_PATH = "notebooks/data/heart_cleaned.csv"
OUTPUT_DIR = "screenshots"
MODELS_DIR = "models"
RESULTS_DIR = "results"

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# MLflow Configuration
EXPERIMENT_NAME = "Heart-Disease-Classification"
mlflow.set_experiment(EXPERIMENT_NAME)

print("=" * 60)
print("FEATURE ENGINEERING & MODEL DEVELOPMENT WITH MLFLOW")
print("=" * 60)
print(f"\nMLflow Experiment: {EXPERIMENT_NAME}")
print(f"Tracking URI: {mlflow.get_tracking_uri()}")

# -----------------------------------
# 2. LOAD CLEANED DATA
# -----------------------------------
print("\n[1] Loading cleaned dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nClass distribution:")
print(df['target'].value_counts())

# -----------------------------------
# 3. FEATURE ENGINEERING
# -----------------------------------
print("\n[2] Feature Engineering...")

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split data into train and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Feature Scaling using StandardScaler
print("\n[3] Scaling features using StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler for future use
scaler_path = f"{MODELS_DIR}/scaler.pkl"
joblib.dump(scaler, scaler_path)
print(f"Scaler saved to {scaler_path}")

# Convert back to DataFrame for better readability
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

print("Feature scaling completed.")

# Cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Dictionary to store results
results = {}

# -----------------------------------
# 4. HELPER FUNCTIONS
# -----------------------------------


def plot_confusion_matrix(y_true, y_pred, title, filename):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(title, fontsize=14)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    return filename


def plot_roc_curve(y_true, y_proba, model_name, filename):
    """Plot and save ROC curve"""
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {model_name}', fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    return filename


def plot_feature_importance(model, feature_names, filename):
    """Plot and save feature importance"""
    plt.figure(figsize=(10, 8))
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
    plt.title('Feature Importance', fontsize=14)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    return filename


# -----------------------------------
# 5. LOGISTIC REGRESSION WITH MLFLOW
# -----------------------------------
print("\n" + "=" * 60)
print("MODEL 1: LOGISTIC REGRESSION")
print("=" * 60)

with mlflow.start_run(run_name="Logistic_Regression") as run:
    print(f"\nMLflow Run ID: {run.info.run_id}")

    # Log dataset info
    mlflow.log_param("dataset_name", "Heart Disease UCI")
    mlflow.log_param("n_samples", len(df))
    mlflow.log_param("n_features", X.shape[1])
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_param("cv_folds", 5)
    mlflow.log_param("scaler", "StandardScaler")

    # Define hyperparameter grid
    lr_param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear'],
        'max_iter': [1000]
    }

    # Log hyperparameter grid
    mlflow.log_param("param_grid", str(lr_param_grid))

    # Initialize model
    lr = LogisticRegression(random_state=RANDOM_STATE)

    # Grid Search with Cross-Validation
    print("Performing Grid Search with Cross-Validation...")
    lr_grid = GridSearchCV(
        lr, lr_param_grid, cv=cv, scoring='roc_auc',
        n_jobs=-1, verbose=1
    )
    lr_grid.fit(X_train_scaled, y_train)

    # Best model
    best_lr = lr_grid.best_estimator_

    # Log best parameters
    for param, value in lr_grid.best_params_.items():
        mlflow.log_param(f"best_{param}", value)

    # Cross-validation scores
    cv_scores = cross_val_score(best_lr, X_train_scaled, y_train, cv=cv, scoring='accuracy')

    # Train final model
    best_lr.fit(X_train_scaled, y_train)

    # Predictions
    y_train_pred = best_lr.predict(X_train_scaled)
    y_test_pred = best_lr.predict(X_test_scaled)
    y_test_proba = best_lr.predict_proba(X_test_scaled)[:, 1]

    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_roc_auc = roc_auc_score(y_test, y_test_proba)

    # Log metrics
    mlflow.log_metric("cv_accuracy_mean", cv_scores.mean())
    mlflow.log_metric("cv_accuracy_std", cv_scores.std())
    mlflow.log_metric("cv_roc_auc", lr_grid.best_score_)
    mlflow.log_metric("train_accuracy", train_accuracy)
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("test_precision", test_precision)
    mlflow.log_metric("test_recall", test_recall)
    mlflow.log_metric("test_f1_score", test_f1)
    mlflow.log_metric("test_roc_auc", test_roc_auc)

    # Print results
    print(f"\nBest parameters: {lr_grid.best_params_}")
    print(f"Best CV ROC-AUC: {lr_grid.best_score_:.4f}")
    print("\nResults:")
    print(f"  CV Accuracy:    {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"  Train Accuracy: {train_accuracy:.4f}")
    print(f"  Test Accuracy:  {test_accuracy:.4f}")
    print(f"  Precision:      {test_precision:.4f}")
    print(f"  Recall:         {test_recall:.4f}")
    print(f"  F1-Score:       {test_f1:.4f}")
    print(f"  ROC-AUC:        {test_roc_auc:.4f}")

    # Generate and log plots
    print("\nGenerating plots...")

    # Confusion Matrix
    cm_path = f"{OUTPUT_DIR}/lr_confusion_matrix.png"
    plot_confusion_matrix(y_test, y_test_pred,
                          "Logistic Regression - Confusion Matrix", cm_path)
    mlflow.log_artifact(cm_path)

    # ROC Curve
    roc_path = f"{OUTPUT_DIR}/lr_roc_curve.png"
    plot_roc_curve(y_test, y_test_proba, "Logistic Regression", roc_path)
    mlflow.log_artifact(roc_path)

    # Classification Report
    report = classification_report(y_test, y_test_pred,
                                   target_names=['No Disease', 'Disease'])
    report_path = f"{RESULTS_DIR}/lr_classification_report.txt"
    with open(report_path, 'w') as f:
        f.write("Logistic Regression - Classification Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(report)
    mlflow.log_artifact(report_path)

    # Log model with signature
    signature = infer_signature(X_train_scaled, best_lr.predict(X_train_scaled))
    mlflow.sklearn.log_model(
        best_lr,
        "model",
        signature=signature,
        registered_model_name="logistic_regression_heart_disease"
    )

    # Save model locally
    model_path = f"{MODELS_DIR}/logistic_regression.pkl"
    joblib.dump(best_lr, model_path)
    mlflow.log_artifact(model_path)

    print("✓ Model and artifacts logged to MLflow")

    # Store results
    results['logistic_regression'] = {
        'run_id': run.info.run_id,
        'best_params': lr_grid.best_params_,
        'test_accuracy': test_accuracy,
        'test_roc_auc': test_roc_auc
    }

# -----------------------------------
# 6. RANDOM FOREST WITH MLFLOW
# -----------------------------------
print("\n" + "=" * 60)
print("MODEL 2: RANDOM FOREST")
print("=" * 60)

with mlflow.start_run(run_name="Random_Forest") as run:
    print(f"\nMLflow Run ID: {run.info.run_id}")

    # Log dataset info
    mlflow.log_param("dataset_name", "Heart Disease UCI")
    mlflow.log_param("n_samples", len(df))
    mlflow.log_param("n_features", X.shape[1])
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_param("cv_folds", 5)
    mlflow.log_param("scaler", "StandardScaler")

    # Define hyperparameter grid
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    # Log hyperparameter grid
    mlflow.log_param("param_grid", str(rf_param_grid))

    # Initialize model
    rf = RandomForestClassifier(random_state=RANDOM_STATE)

    # Grid Search with Cross-Validation
    print("Performing Grid Search with Cross-Validation...")
    rf_grid = GridSearchCV(
        rf, rf_param_grid, cv=cv, scoring='roc_auc',
        n_jobs=-1, verbose=1
    )
    rf_grid.fit(X_train_scaled, y_train)

    # Best model
    best_rf = rf_grid.best_estimator_

    # Log best parameters
    for param, value in rf_grid.best_params_.items():
        mlflow.log_param(f"best_{param}", value)

    # Cross-validation scores
    cv_scores = cross_val_score(best_rf, X_train_scaled, y_train, cv=cv, scoring='accuracy')

    # Train final model
    best_rf.fit(X_train_scaled, y_train)

    # Predictions
    y_train_pred = best_rf.predict(X_train_scaled)
    y_test_pred = best_rf.predict(X_test_scaled)
    y_test_proba = best_rf.predict_proba(X_test_scaled)[:, 1]

    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_roc_auc = roc_auc_score(y_test, y_test_proba)

    # Log metrics
    mlflow.log_metric("cv_accuracy_mean", cv_scores.mean())
    mlflow.log_metric("cv_accuracy_std", cv_scores.std())
    mlflow.log_metric("cv_roc_auc", rf_grid.best_score_)
    mlflow.log_metric("train_accuracy", train_accuracy)
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("test_precision", test_precision)
    mlflow.log_metric("test_recall", test_recall)
    mlflow.log_metric("test_f1_score", test_f1)
    mlflow.log_metric("test_roc_auc", test_roc_auc)

    # Print results
    print(f"\nBest parameters: {rf_grid.best_params_}")
    print(f"Best CV ROC-AUC: {rf_grid.best_score_:.4f}")
    print("\nResults:")
    print("  CV Accuracy:    {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"  Train Accuracy: {train_accuracy:.4f}")
    print(f"  Test Accuracy:  {test_accuracy:.4f}")
    print(f"  Precision:      {test_precision:.4f}")
    print(f"  Recall:         {test_recall:.4f}")
    print(f"  F1-Score:       {test_f1:.4f}")
    print(f"  ROC-AUC:        {test_roc_auc:.4f}")

    # Generate and log plots
    print("\nGenerating plots...")

    # Confusion Matrix
    cm_path = f"{OUTPUT_DIR}/rf_confusion_matrix.png"
    plot_confusion_matrix(y_test, y_test_pred,
                          "Random Forest - Confusion Matrix", cm_path)
    mlflow.log_artifact(cm_path)

    # ROC Curve
    roc_path = f"{OUTPUT_DIR}/rf_roc_curve.png"
    plot_roc_curve(y_test, y_test_proba, "Random Forest", roc_path)
    mlflow.log_artifact(roc_path)

    # Feature Importance
    fi_path = f"{OUTPUT_DIR}/rf_feature_importance.png"
    plot_feature_importance(best_rf, X.columns, fi_path)
    mlflow.log_artifact(fi_path)

    # Classification Report
    report = classification_report(y_test, y_test_pred,
                                   target_names=['No Disease', 'Disease'])
    report_path = f"{RESULTS_DIR}/rf_classification_report.txt"
    with open(report_path, 'w') as f:
        f.write("Random Forest - Classification Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(report)
    mlflow.log_artifact(report_path)

    # Log model with signature
    signature = infer_signature(X_train_scaled, best_rf.predict(X_train_scaled))
    mlflow.sklearn.log_model(
        best_rf,
        "model",
        signature=signature,
        registered_model_name="random_forest_heart_disease"
    )

    # Save model locally
    model_path = f"{MODELS_DIR}/random_forest.pkl"
    joblib.dump(best_rf, model_path)
    mlflow.log_artifact(model_path)

    # Log feature importance data
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': best_rf.feature_importances_
    }).sort_values('importance', ascending=False)
    importance_path = f"{RESULTS_DIR}/feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    mlflow.log_artifact(importance_path)

    print("✓ Model and artifacts logged to MLflow")

    # Store results
    results['random_forest'] = {
        'run_id': run.info.run_id,
        'best_params': rf_grid.best_params_,
        'test_accuracy': test_accuracy,
        'test_roc_auc': test_roc_auc
    }

# -----------------------------------
# 7. MODEL COMPARISON
# -----------------------------------
print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

comparison_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest'],
    'Run ID': [results['logistic_regression']['run_id'],
               results['random_forest']['run_id']],
    'Test Accuracy': [results['logistic_regression']['test_accuracy'],
                      results['random_forest']['test_accuracy']],
    'Test ROC-AUC': [results['logistic_regression']['test_roc_auc'],
                     results['random_forest']['test_roc_auc']]
})

print("\n", comparison_df.to_string(index=False))

# Determine best model
best_model = 'Logistic Regression' if results['logistic_regression']['test_roc_auc'] > \
             results['random_forest']['test_roc_auc'] else 'Random Forest'

print(f"\nBest Model: {best_model}")
print("   Based on Test ROC-AUC Score")

# -----------------------------------
# 8. SUMMARY
# -----------------------------------
print("\n" + "=" * 60)
print("EXPERIMENT TRACKING COMPLETE")
print("=" * 60)
print("\nAll experiments logged to MLflow")
print(f"Experiment Name: {EXPERIMENT_NAME}")
print("Total Runs: 2 (Logistic Regression + Random Forest)")
print("\nLogged Items per Run:")
print("   - Parameters (dataset info, hyperparameters, best params)")
print("   - Metrics (accuracy, precision, recall, F1, ROC-AUC)")
print("   - Artifacts (models, plots, reports)")
print("   - Model signatures")
print("\nView Results:")
print("   Run: mlflow ui")
print("   Then open: http://127.0.0.1:5000")

print("\n" + "=" * 60)
