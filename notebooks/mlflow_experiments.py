"""
MLflow Experiment Tracking Script
Heart Disease UCI Dataset
MLOps Assignment - Task 3

This script demonstrates comprehensive MLflow experiment tracking including:
- Multiple experiment runs
- Parameter logging
- Metric logging
- Artifact logging (models, plots, reports)
- Model versioning
- Run comparison
"""

import os
import warnings

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from mlflow.models.signature import infer_signature
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings('ignore')

# -----------------------------------
# CONFIGURATION
# -----------------------------------
DATA_PATH = "data/heart_cleaned.csv"
OUTPUT_DIR = "screenshots"
MODELS_DIR = "models"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# MLflow setup
EXPERIMENT_NAME = "heart-disease-mlops-comprehensive"
mlflow.set_experiment(EXPERIMENT_NAME)

print("=" * 70)
print("MLFLOW COMPREHENSIVE EXPERIMENT TRACKING")
print("Heart Disease Classification")
print("=" * 70)
print(f"\nExperiment: {EXPERIMENT_NAME}")
print(f"Tracking URI: {mlflow.get_tracking_uri()}")
print(f"Artifact Location: {mlflow.get_experiment_by_name(EXPERIMENT_NAME).artifact_location}")

# -----------------------------------
# LOAD AND PREPARE DATA
# -----------------------------------
print("\n[1] Loading and preparing data...")
df = pd.read_csv(DATA_PATH)

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# Cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# -----------------------------------
# HELPER FUNCTIONS
# -----------------------------------


def log_confusion_matrix(y_true, y_pred, model_name, run_id):
    """Generate and log confusion matrix to MLflow"""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'{model_name} - Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

    cm_path = f"{OUTPUT_DIR}/{run_id}_confusion_matrix.png"
    plt.savefig(cm_path, dpi=200, bbox_inches='tight')
    plt.close()
    mlflow.log_artifact(cm_path)

    # Clean up
    if os.path.exists(cm_path):
        os.remove(cm_path)


def log_roc_curve(y_true, y_proba, model_name, run_id):
    """Generate and log ROC curve to MLflow"""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f'ROC (AUC = {auc:.3f})', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{model_name} - ROC Curve')
    ax.legend()
    ax.grid(alpha=0.3)

    roc_path = f"{OUTPUT_DIR}/{run_id}_roc_curve.png"
    plt.savefig(roc_path, dpi=200, bbox_inches='tight')
    plt.close()
    mlflow.log_artifact(roc_path)

    # Clean up
    if os.path.exists(roc_path):
        os.remove(roc_path)


def train_and_log_model(model, model_name, X_train, X_test, y_train, y_test, params=None):
    """
    Train a model and log everything to MLflow

    Returns:
        dict: Model results including metrics
    """
    with mlflow.start_run(run_name=model_name) as run:
        print(f"\n{'=' * 70}")
        print(f"Training: {model_name}")
        print(f"Run ID: {run.info.run_id}")
        print(f"{'=' * 70}")

        # Log basic information
        mlflow.set_tag("model_type", model_name)
        mlflow.log_param("train_samples", X_train.shape[0])
        mlflow.log_param("test_samples", X_test.shape[0])
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("random_state", RANDOM_STATE)

        # Log model-specific parameters
        if params:
            for key, value in params.items():
                mlflow.log_param(key, value)

        # Train model
        model.fit(X_train, y_train)

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        # Log CV scores
        mlflow.log_metric("cv_accuracy_mean", cv_mean)
        mlflow.log_metric("cv_accuracy_std", cv_std)
        for i, score in enumerate(cv_scores):
            mlflow.log_metric(f"cv_fold_{i + 1}", score)

        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Get probabilities (if available)
        if hasattr(model, 'predict_proba'):
            y_test_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_test_proba = model.decision_function(X_test)

        # Calculate metrics
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_prec = precision_score(y_test, y_test_pred)
        test_rec = recall_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        test_auc = roc_auc_score(y_test, y_test_proba)

        # Log metrics
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_precision", test_prec)
        mlflow.log_metric("test_recall", test_rec)
        mlflow.log_metric("test_f1_score", test_f1)
        mlflow.log_metric("test_roc_auc", test_auc)

        # Calculate and log overfitting metric
        overfitting = train_acc - test_acc
        mlflow.log_metric("overfitting_gap", overfitting)

        print("\nResults:")
        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Test Accuracy:  {test_acc:.4f}")
        print(f"  Precision:      {test_prec:.4f}")
        print(f"  Recall:         {test_rec:.4f}")
        print(f"  F1-Score:       {test_f1:.4f}")
        print(f"  ROC-AUC:        {test_auc:.4f}")
        print(f"  CV Accuracy:    {cv_mean:.4f} (+/- {cv_std:.4f})")

        # Log artifacts
        log_confusion_matrix(y_test, y_test_pred, model_name, run.info.run_id)
        log_roc_curve(y_test, y_test_proba, model_name, run.info.run_id)

        # Log model with signature
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            model,
            "model",
            signature=signature,
            registered_model_name=f"heart_disease_{model_name.lower().replace(' ', '_')}"
        )

        # Log feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            feature_imp = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            fig, ax = plt.subplots(figsize=(10, 8))
            sns.barplot(data=feature_imp, x='importance', y='feature', palette='viridis', ax=ax)
            ax.set_title(f'{model_name} - Feature Importance')

            imp_path = f"{OUTPUT_DIR}/{run.info.run_id}_feature_importance.png"
            plt.tight_layout()
            plt.savefig(imp_path, dpi=200, bbox_inches='tight')
            plt.close()
            mlflow.log_artifact(imp_path)

            # Clean up
            if os.path.exists(imp_path):
                os.remove(imp_path)

        print("✓ Logged: parameters, metrics, artifacts, and model")

        return {
            'run_id': run.info.run_id,
            'model_name': model_name,
            'test_accuracy': test_acc,
            'test_roc_auc': test_auc,
            'test_f1': test_f1,
            'cv_mean': cv_mean
        }


# -----------------------------------
# EXPERIMENT 1: LOGISTIC REGRESSION VARIANTS
# -----------------------------------
print("\n" + "=" * 70)
print("EXPERIMENT SET 1: Logistic Regression with Different Regularizations")
print("=" * 70)

results = []

# L1 Regularization
result = train_and_log_model(
    LogisticRegression(penalty='l1', C=1.0, solver='liblinear', random_state=RANDOM_STATE, max_iter=1000),
    "Logistic Regression (L1)",
    X_train_scaled, X_test_scaled, y_train, y_test,
    {'penalty': 'l1', 'C': 1.0, 'solver': 'liblinear'}
)
results.append(result)

# L2 Regularization
result = train_and_log_model(
    LogisticRegression(penalty='l2', C=1.0, solver='liblinear', random_state=RANDOM_STATE, max_iter=1000),
    "Logistic Regression (L2)",
    X_train_scaled, X_test_scaled, y_train, y_test,
    {'penalty': 'l2', 'C': 1.0, 'solver': 'liblinear'}
)
results.append(result)

# Strong Regularization
result = train_and_log_model(
    LogisticRegression(penalty='l2', C=0.1, solver='liblinear', random_state=RANDOM_STATE, max_iter=1000),
    "Logistic Regression (Strong Reg)",
    X_train_scaled, X_test_scaled, y_train, y_test,
    {'penalty': 'l2', 'C': 0.1, 'solver': 'liblinear'}
)
results.append(result)

# -----------------------------------
# EXPERIMENT 2: RANDOM FOREST VARIANTS
# -----------------------------------
print("\n" + "=" * 70)
print("EXPERIMENT SET 2: Random Forest with Different Configurations")
print("=" * 70)

# Small Forest
result = train_and_log_model(
    RandomForestClassifier(n_estimators=50, max_depth=5, random_state=RANDOM_STATE),
    "Random Forest (Small)",
    X_train_scaled, X_test_scaled, y_train, y_test,
    {'n_estimators': 50, 'max_depth': 5}
)
results.append(result)

# Medium Forest
result = train_and_log_model(
    RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE),
    "Random Forest (Medium)",
    X_train_scaled, X_test_scaled, y_train, y_test,
    {'n_estimators': 100, 'max_depth': 10}
)
results.append(result)

# Large Forest
result = train_and_log_model(
    RandomForestClassifier(n_estimators=200, max_depth=None, random_state=RANDOM_STATE),
    "Random Forest (Large)",
    X_train_scaled, X_test_scaled, y_train, y_test,
    {'n_estimators': 200, 'max_depth': None}
)
results.append(result)

# -----------------------------------
# EXPERIMENT 3: OTHER CLASSIFIERS
# -----------------------------------
print("\n" + "=" * 70)
print("EXPERIMENT SET 3: Other Classification Algorithms")
print("=" * 70)

# Gradient Boosting
result = train_and_log_model(
    GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=RANDOM_STATE),
    "Gradient Boosting",
    X_train_scaled, X_test_scaled, y_train, y_test,
    {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3}
)
results.append(result)

# Support Vector Machine
result = train_and_log_model(
    SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=RANDOM_STATE),
    "Support Vector Machine",
    X_train_scaled, X_test_scaled, y_train, y_test,
    {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'}
)
results.append(result)

# -----------------------------------
# RESULTS COMPARISON
# -----------------------------------
print("\n" + "=" * 70)
print("EXPERIMENT RESULTS COMPARISON")
print("=" * 70)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('test_roc_auc', ascending=False)

print("\n", results_df[['model_name', 'test_accuracy', 'test_roc_auc', 'test_f1', 'cv_mean']].to_string(index=False))

# Save comparison
comparison_path = f"{MODELS_DIR}/mlflow_experiment_comparison.csv"
results_df.to_csv(comparison_path, index=False)
print(f"\n✓ Comparison saved to: {comparison_path}")

# -----------------------------------
# SUMMARY
# -----------------------------------
print("\n" + "=" * 70)
print("MLFLOW EXPERIMENT TRACKING SUMMARY")
print("=" * 70)

best_model = results_df.iloc[0]
print(f"\n🏆 Best Model: {best_model['model_name']}")
print(f"   Test ROC-AUC: {best_model['test_roc_auc']:.4f}")
print(f"   Test Accuracy: {best_model['test_accuracy']:.4f}")
print(f"   Run ID: {best_model['run_id']}")

print(f"\n📊 Total Experiments Run: {len(results)}")
print(f"   Experiment Name: {EXPERIMENT_NAME}")
print("\n💡 To view results in MLflow UI, run:")
print("   cd {os.getcwd()}")
print("   mlflow ui")
print("   Then open: http://localhost:5000")

print("\n✓ All experiments logged to MLflow successfully!")
print("=" * 70)
