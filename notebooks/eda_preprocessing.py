
"""
EDA and Preprocessing Script
Heart Disease UCI Dataset
MLOps Assignment - Task 1
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# -----------------------------------
# 1. LOAD DATA
# -----------------------------------
DATA_PATH = "/Users/VSE18/Desktop/Projects/heart-disease-mlops/data/processed.cleveland.data"      # change name if needed
OUTPUT_DIR = "screenshots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading dataset...")

# Define column names as the dataset does not have a header
column_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
    'ca', 'thal', 'target'
]
df = pd.read_csv(DATA_PATH, header=None, names=column_names)
print(f"Dataset loaded successfully with shape: {df.shape}")

# -----------------------------------
# 2. DATA CLEANING & PREPROCESSING
# -----------------------------------

print("\nCleaning data...")

# Replace '?' with NaN (if present)
df.replace("?", np.nan, inplace=True)

# Convert all columns to numeric where possible
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Handle missing values using median
df.fillna(df.median(), inplace=True)

# Ensure target is binary
# (UCI dataset sometimes has values 1–4 for disease)
df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

# Save cleaned dataset
CLEAN_PATH = "data/heart_cleaned.csv"
os.makedirs(os.path.dirname(CLEAN_PATH), exist_ok=True)
df.to_csv(CLEAN_PATH, index=False)
print(f"Cleaned dataset saved to {CLEAN_PATH}")

# -----------------------------------
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# -----------------------------------

sns.set(style="whitegrid")

# 3.1 Feature Distributions
print("Generating feature distribution plots...")
df.hist(figsize=(16, 12))
plt.suptitle("Feature Distributions", fontsize=16)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/feature_distributions.png")
plt.close()

# 3.2 Correlation Heatmap
print("Generating correlation heatmap...")
plt.figure(figsize=(12, 10))
corr_matrix = df.corr()
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True
)
plt.title("Feature Correlation Heatmap", fontsize=14)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png")
plt.close()

# 3.3 Class Balance Plot
print("Generating class balance plot...")
plt.figure(figsize=(6, 4))
sns.countplot(x="target", data=df)
plt.title("Class Balance (Heart Disease)")
plt.xlabel("Target (0 = No Disease, 1 = Disease)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/class_balance.png")
plt.close()

print("\nEDA completed successfully.")
print("Plots saved in 'screenshots/' directory.")
