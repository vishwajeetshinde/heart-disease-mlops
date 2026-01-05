# Heart Disease MLOps Project

This project demonstrates a complete MLOps workflow for heart disease prediction using the UCI Heart Disease dataset.

## Features
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering (scaling, encoding)
- Model development (Logistic Regression, Random Forest)
- Hyperparameter tuning and cross-validation
- Experiment tracking with MLflow
- Model evaluation and visualization

## Getting Started
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the main scripts in the `notebooks/` folder:
   - `eda_preprocessing.py` for data cleaning and EDA
   - `feature_engineering_modeling_mlflow.py` for model training and MLflow tracking

## Code Quality & Testing
- **Check linting issues:** `python check_linting.py`
- **Fix linting automatically:** `python fix_linting.py`
- **Run all tests:** `python run_tests.py`
- **See guides:** `LINTING_FIXES.md` and `CI_CD_GUIDE.md`

## Experiment Tracking
- Start MLflow UI:
  ```bash
  mlflow ui
  ```
- Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser to view experiment results.

## Project Structure
```
heart-disease-mlops/
├── data/
├── notebooks/
├── models/
├── results/
├── screenshots/
├── requirements.txt
├── Makefile
└── README.md
```

## Author
- MLOps Assignment, January 2026
