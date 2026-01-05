"""
Feature Engineering & Model Development Script
Heart Disease UCI Dataset
MLOps Assignment - Task 2 & Task 3

This script performs:
- Feature scaling and encoding
- Model training (Logistic Regression & Random Forest)
- Hyperparameter tuning with GridSearchCV
- Cross-validation
- Model evaluation (accuracy, precision, recall, ROC-AUC)
- MLflow experiment tracking (parameters, metrics, artifacts, plots)
"""

import json
import os

import joblib
import matplotlib.pyplot as plt
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
DATA_PATH = "data/heart_cleaned.csv"
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
MLFLOW_EXPERIMENT_NAME = "heart-disease-classification"
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

print("=" * 60)
print("FEATURE ENGINEERING & MODEL DEVELOPMENT")
print("WITH MLFLOW EXPERIMENT TRACKING")
print("=" * 60)
print("\nMLflow Experiment: {MLFLOW_EXPERIMENT_NAME}")
print("Tracking URI: {mlflow.get_tracking_uri()}")

# -----------------------------------
# 2. LOAD CLEANED DATA
# -----------------------------------
print("\n[1] Loading cleaned dataset...")
df = pd.read_csv(DATA_PATH)
print("Dataset shape: {df.shape}")
print("Columns: {list(df.columns)}")
print("\nClass distribution:")
print(df['target'].value_counts())

# -----------------------------------
# 3. FEATURE ENGINEERING
# -----------------------------------
print("\n[2] Feature Engineering...")

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

print("Features shape: {X.shape}")
print("Target shape: {y.shape}")

# Split data into train and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print("\nTrain set: {X_train.shape[0]} samples")
print("Test set: {X_test.shape[0]} samples")

# Feature Scaling using StandardScaler
print("\n[3] Scaling features using StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler for future use
scaler_path = "{MODELS_DIR}/scaler.pkl"
joblib.dump(scaler, scaler_path)
print("Scaler saved to {scaler_path}")

# Convert back to DataFrame for better readability (optional)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

print("Feature scaling completed.")

# -----------------------------------
# 4. MODEL DEVELOPMENT
# -----------------------------------
print("\n" + "=" * 60)
print("MODEL TRAINING & HYPERPARAMETER TUNING")
print("=" * 60)

# Cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Dictionary to store results
results = {}

# -----------------------------------
# 4.1 LOGISTIC REGRESSION
# -----------------------------------
print("\n[4.1] Logistic Regression")
print("-" * 40)

# Start MLflow run for Logistic Regression
with mlflow.start_run(run_name="logistic_regression"):

    # Log dataset information
    mlflow.log_param("dataset_name", "heart_disease_cleveland")
    mlflow.log_param("train_samples", X_train.shape[0])
    mlflow.log_param("test_samples", X_test.shape[0])
    mlflow.log_param("n_features", X_train.shape[1])
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("scaler", "StandardScaler")

    # Define hyperparameter grid
    lr_param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear'],
        'max_iter': [1000]
    }

    # Log hyperparameter search space
    mlflow.log_param("param_grid", str(lr_param_grid))
    mlflow.log_param("cv_folds", 5)
    mlflow.log_param("cv_strategy", "StratifiedKFold")

    # Initialize model
    lr = LogisticRegression(random_state=RANDOM_STATE)

    # Grid Search with Cross-Validation
    print("Performing Grid Search...")
    lr_grid = GridSearchCV(
        lr, lr_param_grid, cv=cv, scoring='roc_auc',
        n_jobs=-1, verbose=1
    )
    lr_grid.fit(X_train_scaled, y_train)

    # Best model
    best_lr = lr_grid.best_estimator_
    print("\nBest parameters: {lr_grid.best_params_}")
    print("Best CV ROC-AUC score: {lr_grid.best_score_:.4f}")

    # Log best hyperparameters
    for param, value in lr_grid.best_params_.items():
        mlflow.log_param(f"best_{param}", value)

    # Cross-validation scores
    cv_scores_lr = cross_val_score(best_lr, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    print("Cross-validation accuracy: {cv_scores_lr.mean():.4f} (+/- {cv_scores_lr.std():.4f})")

    # Train final model
    best_lr.fit(X_train_scaled, y_train)

    # Predictions
    y_train_pred_lr = best_lr.predict(X_train_scaled)
    y_test_pred_lr = best_lr.predict(X_test_scaled)
    y_test_proba_lr = best_lr.predict_proba(X_test_scaled)[:, 1]

    # Evaluate
    lr_results = {
        'model_name': 'Logistic Regression',
        'best_params': lr_grid.best_params_,
        'best_cv_score': lr_grid.best_score_,
        'train_accuracy': accuracy_score(y_train, y_train_pred_lr),
        'test_accuracy': accuracy_score(y_test, y_test_pred_lr),
        'test_precision': precision_score(y_test, y_test_pred_lr),
        'test_recall': recall_score(y_test, y_test_pred_lr),
        'test_f1': f1_score(y_test, y_test_pred_lr),
        'test_roc_auc': roc_auc_score(y_test, y_test_proba_lr),
        'cv_accuracy_mean': cv_scores_lr.mean(),
        'cv_accuracy_std': cv_scores_lr.std()
    }

    # Log all metrics to MLflow
    mlflow.log_metric("best_cv_roc_auc", lr_results['best_cv_score'])
    mlflow.log_metric("train_accuracy", lr_results['train_accuracy'])
    mlflow.log_metric("test_accuracy", lr_results['test_accuracy'])
    mlflow.log_metric("test_precision", lr_results['test_precision'])
    mlflow.log_metric("test_recall", lr_results['test_recall'])
    mlflow.log_metric("test_f1_score", lr_results['test_f1'])
    mlflow.log_metric("test_roc_auc", lr_results['test_roc_auc'])
    mlflow.log_metric("cv_accuracy_mean", lr_results['cv_accuracy_mean'])
    mlflow.log_metric("cv_accuracy_std", lr_results['cv_accuracy_std'])

    # Log cross-validation scores
    for i, score in enumerate(cv_scores_lr):
        mlflow.log_metric(f"cv_fold_{i + 1}_accuracy", score)

    results['logistic_regression'] = lr_results

    print("\nLogistic Regression Results:")
    print("  Train Accuracy: {lr_results['train_accuracy']:.4f}")
    print("  Test Accuracy:  {lr_results['test_accuracy']:.4f}")
    print("  Precision:      {lr_results['test_precision']:.4f}")
    print("  Recall:         {lr_results['test_recall']:.4f}")
    print("  F1-Score:       {lr_results['test_f1']:.4f}")
    print("  ROC-AUC:        {lr_results['test_roc_auc']:.4f}")

    # Generate and log confusion matrix
    cm_lr = confusion_matrix(y_test, y_test_pred_lr)
    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_title('Logistic Regression - Confusion Matrix')
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('Actual')
    cm_path = "{OUTPUT_DIR}/lr_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    mlflow.log_artifact(cm_path)

    # Generate and log ROC curve
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_test_proba_lr)
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    ax_roc.plot(fpr_lr, tpr_lr, label='ROC (AUC = {lr_results["test_roc_auc"]:.3f})', linewidth=2)
    ax_roc.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    ax_roc.set_xlabel('False Positive Rate', fontsize=12)
    ax_roc.set_ylabel('True Positive Rate', fontsize=12)
    ax_roc.set_title('Logistic Regression - ROC Curve', fontsize=14)
    ax_roc.legend(loc='lower right')
    ax_roc.grid(alpha=0.3)
    roc_path = "{OUTPUT_DIR}/lr_roc_curve.png"
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    mlflow.log_artifact(roc_path)

    # Log classification report
    report = classification_report(y_test, y_test_pred_lr, target_names=['No Disease', 'Disease'])
    report_path = "{RESULTS_DIR}/lr_classification_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    mlflow.log_artifact(report_path)

    # Save and log model
    lr_model_path = f"{MODELS_DIR}/logistic_regression.pkl"
    joblib.dump(best_lr, lr_model_path)

    # Log model with signature
    signature = infer_signature(X_train_scaled, best_lr.predict(X_train_scaled))
    mlflow.sklearn.log_model(best_lr, "model", signature=signature)
    mlflow.log_artifact(lr_model_path)

    # Log scaler
    mlflow.log_artifact(scaler_path)

    print("\nModel saved to {lr_model_path}")
    print("✓ Logged to MLflow: parameters, metrics, artifacts, and model")

# -----------------------------------
# 4.2 RANDOM FOREST
# -----------------------------------
print("\n[4.2] Random Forest Classifier")
print("-" * 40)

# Start MLflow run for Random Forest
with mlflow.start_run(run_name="random_forest"):

    # Log dataset information
    mlflow.log_param("dataset_name", "heart_disease_cleveland")
    mlflow.log_param("train_samples", X_train.shape[0])
    mlflow.log_param("test_samples", X_test.shape[0])
    mlflow.log_param("n_features", X_train.shape[1])
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("scaler", "StandardScaler")

    # Define hyperparameter grid
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    # Log hyperparameter search space
    mlflow.log_param("param_grid", str(rf_param_grid))
    mlflow.log_param("cv_folds", 5)
    mlflow.log_param("cv_strategy", "StratifiedKFold")

    # Initialize model
    rf = RandomForestClassifier(random_state=RANDOM_STATE)

    # Grid Search with Cross-Validation
    print("Performing Grid Search...")
    rf_grid = GridSearchCV(
        rf, rf_param_grid, cv=cv, scoring='roc_auc',
        n_jobs=-1, verbose=1
    )
    rf_grid.fit(X_train_scaled, y_train)

    # Best model
    best_rf = rf_grid.best_estimator_
    print(f"\nBest parameters: {rf_grid.best_params_}")
    print(f"Best CV ROC-AUC score: {rf_grid.best_score_:.4f}")

    # Log best hyperparameters
    for param, value in rf_grid.best_params_.items():
        mlflow.log_param(f"best_{param}", value)

    # Cross-validation scores
    cv_scores_rf = cross_val_score(best_rf, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    print(f"Cross-validation accuracy: {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std():.4f})")

    # Train final model
    best_rf.fit(X_train_scaled, y_train)

    # Predictions
    y_train_pred_rf = best_rf.predict(X_train_scaled)
    y_test_pred_rf = best_rf.predict(X_test_scaled)
    y_test_proba_rf = best_rf.predict_proba(X_test_scaled)[:, 1]

    # Evaluate
    rf_results = {
        'model_name': 'Random Forest',
        'best_params': rf_grid.best_params_,
        'best_cv_score': rf_grid.best_score_,
        'train_accuracy': accuracy_score(y_train, y_train_pred_rf),
        'test_accuracy': accuracy_score(y_test, y_test_pred_rf),
        'test_precision': precision_score(y_test, y_test_pred_rf),
        'test_recall': recall_score(y_test, y_test_pred_rf),
        'test_f1': f1_score(y_test, y_test_pred_rf),
        'test_roc_auc': roc_auc_score(y_test, y_test_proba_rf),
        'cv_accuracy_mean': cv_scores_rf.mean(),
        'cv_accuracy_std': cv_scores_rf.std()
    }

    # Log all metrics to MLflow
    mlflow.log_metric("best_cv_roc_auc", rf_results['best_cv_score'])
    mlflow.log_metric("train_accuracy", rf_results['train_accuracy'])
    mlflow.log_metric("test_accuracy", rf_results['test_accuracy'])
    mlflow.log_metric("test_precision", rf_results['test_precision'])
    mlflow.log_metric("test_recall", rf_results['test_recall'])
    mlflow.log_metric("test_f1_score", rf_results['test_f1'])
    mlflow.log_metric("test_roc_auc", rf_results['test_roc_auc'])
    mlflow.log_metric("cv_accuracy_mean", rf_results['cv_accuracy_mean'])
    mlflow.log_metric("cv_accuracy_std", rf_results['cv_accuracy_std'])

    # Log cross-validation scores
    for i, score in enumerate(cv_scores_rf):
        mlflow.log_metric(f"cv_fold_{i + 1}_accuracy", score)

    results['random_forest'] = rf_results

    print("\nRandom Forest Results:")
    print(f"  Train Accuracy: {rf_results['train_accuracy']:.4f}")
    print(f"  Test Accuracy:  {rf_results['test_accuracy']:.4f}")
    print(f"  Precision:      {rf_results['test_precision']:.4f}")
    print(f"  Recall:         {rf_results['test_recall']:.4f}")
    print(f"  F1-Score:       {rf_results['test_f1']:.4f}")
    print(f"  ROC-AUC:        {rf_results['test_roc_auc']:.4f}")

    # Generate and log confusion matrix
    cm_rf = confusion_matrix(y_test, y_test_pred_rf)
    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=ax_cm)
    ax_cm.set_title('Random Forest - Confusion Matrix')
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('Actual')
    cm_path = f"{OUTPUT_DIR}/rf_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    mlflow.log_artifact(cm_path)

    # Generate and log ROC curve
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_test_proba_rf)
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    ax_roc.plot(fpr_rf, tpr_rf, label=f'ROC (AUC = {rf_results["test_roc_auc"]:.3f})', linewidth=2)
    ax_roc.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    ax_roc.set_xlabel('False Positive Rate', fontsize=12)
    ax_roc.set_ylabel('True Positive Rate', fontsize=12)
    ax_roc.set_title('Random Forest - ROC Curve', fontsize=14)
    ax_roc.legend(loc='lower right')
    ax_roc.grid(alpha=0.3)
    roc_path = f"{OUTPUT_DIR}/rf_roc_curve.png"
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    mlflow.log_artifact(roc_path)

    # Generate and log feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_rf.feature_importances_
    }).sort_values('importance', ascending=False)

    fig_imp, ax_imp = plt.subplots(figsize=(10, 8))
    sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis', ax=ax_imp)
    ax_imp.set_title('Random Forest - Feature Importance', fontsize=14)
    ax_imp.set_xlabel('Importance', fontsize=12)
    ax_imp.set_ylabel('Feature', fontsize=12)
    imp_path = f"{OUTPUT_DIR}/rf_feature_importance.png"
    plt.tight_layout()
    plt.savefig(imp_path, dpi=300, bbox_inches='tight')
    plt.close()
    mlflow.log_artifact(imp_path)

    # Save and log feature importance CSV
    feature_importance_path = f"{RESULTS_DIR}/rf_feature_importance.csv"
    feature_importance.to_csv(feature_importance_path, index=False)
    mlflow.log_artifact(feature_importance_path)

    # Log classification report
    report = classification_report(y_test, y_test_pred_rf, target_names=['No Disease', 'Disease'])
    report_path = f"{RESULTS_DIR}/rf_classification_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    mlflow.log_artifact(report_path)

    # Save and log model
    rf_model_path = f"{MODELS_DIR}/random_forest.pkl"
    joblib.dump(best_rf, rf_model_path)

    # Log model with signature
    signature = infer_signature(X_train_scaled, best_rf.predict(X_train_scaled))
    mlflow.sklearn.log_model(best_rf, "model", signature=signature)
    mlflow.log_artifact(rf_model_path)

    # Log scaler
    mlflow.log_artifact(scaler_path)

    print("\nModel saved to {rf_model_path}")
    print("✓ Logged to MLflow: parameters, metrics, artifacts, and model")

# -----------------------------------
# 5. MODEL COMPARISON
# -----------------------------------
print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

comparison_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest'],
    'CV ROC-AUC': [lr_results['best_cv_score'], rf_results['best_cv_score']],
    'Test Accuracy': [lr_results['test_accuracy'], rf_results['test_accuracy']],
    'Test Precision': [lr_results['test_precision'], rf_results['test_precision']],
    'Test Recall': [lr_results['test_recall'], rf_results['test_recall']],
    'Test F1-Score': [lr_results['test_f1'], rf_results['test_f1']],
    'Test ROC-AUC': [lr_results['test_roc_auc'], rf_results['test_roc_auc']]
})

print("\n", comparison_df.to_string(index=False))

# Save comparison
comparison_path = "{RESULTS_DIR}/model_comparison.csv"
comparison_df.to_csv(comparison_path, index=False)
print("\nComparison saved to {comparison_path}")

# -----------------------------------
# 6. DETAILED EVALUATION & VISUALIZATION
# -----------------------------------
print("\n[5] Generating evaluation visualizations...")

# Start MLflow run for comparison plots
with mlflow.start_run(run_name="model_comparison_plots"):

    mlflow.log_param("comparison_type", "logistic_regression_vs_random_forest")

    # 6.1 Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cm_lr = confusion_matrix(y_test, y_test_pred_lr)
    sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Logistic Regression - Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')

    cm_rf = confusion_matrix(y_test, y_test_pred_rf)
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[1])
    axes[1].set_title('Random Forest - Confusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')

    plt.tight_layout()
    cm_comparison_path = f"{OUTPUT_DIR}/confusion_matrices.png"
    plt.savefig(cm_comparison_path, dpi=300)
    plt.close()
    mlflow.log_artifact(cm_comparison_path)
    print("✓ Confusion matrices saved")

    # 6.2 ROC Curves
    plt.figure(figsize=(10, 8))

    # Logistic Regression ROC
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_test_proba_lr)
    plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {lr_results["test_roc_auc"]:.3f})', linewidth=2)

    # Random Forest ROC
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_test_proba_rf)
    plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {rf_results["test_roc_auc"]:.3f})', linewidth=2)

    # Diagonal line
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)

    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Model Comparison', fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    roc_comparison_path = f"{OUTPUT_DIR}/roc_curves.png"
    plt.savefig(roc_comparison_path, dpi=300)
    plt.close()
    mlflow.log_artifact(roc_comparison_path)
    print("✓ ROC curves saved")

    # 6.3 Feature Importance (Random Forest)
    plt.figure(figsize=(10, 8))
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_rf.feature_importances_
    }).sort_values('importance', ascending=False)

    sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis')
    plt.title('Random Forest - Feature Importance', fontsize=14)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    feature_imp_path = f"{OUTPUT_DIR}/feature_importance.png"
    plt.savefig(feature_imp_path, dpi=300)
    plt.close()
    mlflow.log_artifact(feature_imp_path)
    print("✓ Feature importance plot saved")

    # Save feature importance
    feature_importance.to_csv(f"{RESULTS_DIR}/feature_importance.csv", index=False)

    # 6.4 Model Performance Metrics Comparison
    metrics = ['Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1-Score', 'Test ROC-AUC']
    lr_scores = [lr_results['test_accuracy'], lr_results['test_precision'],
                 lr_results['test_recall'], lr_results['test_f1'], lr_results['test_roc_auc']]
    rf_scores = [rf_results['test_accuracy'], rf_results['test_precision'],
                 rf_results['test_recall'], rf_results['test_f1'], rf_results['test_roc_auc']]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, lr_scores, width, label='Logistic Regression', color='skyblue')
    bars2 = ax.bar(x + width / 2, rf_scores, width, label='Random Forest', color='lightgreen')

    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=15, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    perf_comparison_path = f"{OUTPUT_DIR}/model_performance_comparison.png"
    plt.savefig(perf_comparison_path, dpi=300)
    plt.close()
    mlflow.log_artifact(perf_comparison_path)
    print("✓ Performance comparison plot saved")

    # Log comparison metrics
    mlflow.log_metric("lr_vs_rf_accuracy_diff", rf_results['test_accuracy'] - lr_results['test_accuracy'])
    mlflow.log_metric("lr_vs_rf_roc_auc_diff", rf_results['test_roc_auc'] - lr_results['test_roc_auc'])

# 6.5 Classification Reports
print("\n[6] Classification Reports:")
print("\nLogistic Regression:")
print("-" * 40)
print(classification_report(y_test, y_test_pred_lr, target_names=['No Disease', 'Disease']))

print("\nRandom Forest:")
print("-" * 40)
print(classification_report(y_test, y_test_pred_rf, target_names=['No Disease', 'Disease']))

# Save classification reports
with open(f"{RESULTS_DIR}/classification_reports.txt", 'w') as f:
    f.write("CLASSIFICATION REPORTS\n")
    f.write("=" * 60 + "\n\n")
    f.write("Logistic Regression:\n")
    f.write("-" * 40 + "\n")
    f.write(classification_report(y_test, y_test_pred_lr, target_names=['No Disease', 'Disease']))
    f.write("\n\nRandom Forest:\n")
    f.write("-" * 40 + "\n")
    f.write(classification_report(y_test, y_test_pred_rf, target_names=['No Disease', 'Disease']))

# -----------------------------------
# 7. SAVE RESULTS
# -----------------------------------
print("\n[7] Saving results...")

# Save all results as JSON
results_json_path = f"{RESULTS_DIR}/model_results.json"
with open(results_json_path, 'w') as f:
    # Convert numpy types to native Python types for JSON serialization
    results_serializable = {}
    for model, res in results.items():
        results_serializable[model] = {}
        for key, value in res.items():
            if isinstance(value, (np.integer, np.floating)):
                results_serializable[model][key] = float(value)
            elif isinstance(value, dict):
                results_serializable[model][key] = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                                                    for k, v in value.items()}
            else:
                results_serializable[model][key] = value
    json.dump(results_serializable, f, indent=4)

print(f"✓ Results saved to {results_json_path}")

# -----------------------------------
# 8. SUMMARY
# -----------------------------------
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("\n✓ Feature engineering completed (StandardScaler)")
print("✓ Two models trained and tuned:")
print("  - Logistic Regression")
print("  - Random Forest Classifier")
print("✓ Hyperparameter tuning with GridSearchCV")
print("✓ 5-fold stratified cross-validation")
print("✓ Comprehensive evaluation metrics computed")
print("✓ MLflow experiment tracking enabled")
print("\n📊 MLflow UI: Run 'mlflow ui' to view experiments")
print("   Experiment: {MLFLOW_EXPERIMENT_NAME}")
print("\nFiles saved:")
print("  - Models: {MODELS_DIR}/")
print("  - Results: {RESULTS_DIR}/")
print("  - Plots: {OUTPUT_DIR}/")
print("  - MLflow: mlruns/ directory")

# Determine best model
best_model_name = 'Logistic Regression' if lr_results['test_roc_auc'] > rf_results['test_roc_auc'] else 'Random Forest'
print("\n🏆 Best performing model: {best_model_name}")
print("   (Based on Test ROC-AUC score)")

print("\n" + "=" * 60)
print("FEATURE ENGINEERING & MODEL DEVELOPMENT COMPLETE!")
print("WITH MLFLOW EXPERIMENT TRACKING")
print("=" * 60)
