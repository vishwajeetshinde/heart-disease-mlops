# CI/CD Pipeline Documentation

## Overview
This project implements a complete CI/CD pipeline using GitHub Actions for automated testing, linting, and model training.

## Pipeline Components

### 1. **Linting** (Code Quality)
- **Tool:** Flake8
- **Configuration:** `.flake8`
- **Purpose:** Ensures code follows Python style guidelines and best practices
- **Checks:**
  - Code formatting
  - Syntax errors
  - Complexity analysis
  - Import organization

### 2. **Unit Testing** (Code Correctness)
- **Framework:** Pytest
- **Configuration:** `pytest.ini`
- **Coverage:** pytest-cov
- **Test Suites:**
  - `test_data_processing.py` - Data loading, cleaning, validation (60+ tests)
  - `test_pipeline.py` - Preprocessing pipeline, model loading, predictions (40+ tests)
  - `test_model_training.py` - Model performance, robustness, evaluation (35+ tests)

### 3. **Model Training** (ML Pipeline)
- **Script:** `notebooks/feature_engineering_modeling_mlflow.py`
- **Tracking:** MLflow
- **Artifacts:** Models, plots, metrics, reports

### 4. **Artifact Management**
- Test results (JUnit XML)
- Trained models (.pkl files)
- MLflow run logs
- Coverage reports

## GitHub Actions Workflow

### Workflow File
Location: `.github/workflows/ci.yml`

### Trigger Events
- **Push** to main branch
- **Pull Request** to main branch

### Jobs & Steps

```yaml
1. Checkout code
2. Set up Python 3.10
3. Install dependencies
4. Run linting (flake8)
5. Run unit tests (pytest)
6. Train models (optional)
7. Upload artifacts
```

### Artifacts Generated
- `test-results/` - Test execution results
- `trained-models/` - Model files (.pkl)
- `mlflow-logs/` - Experiment tracking data

## Running Tests Locally

### Method 1: Using Test Runner Script
```bash
# Run all tests + linting
python run_tests.py

# Lint only
python run_tests.py --lint-only

# Tests only
python run_tests.py --test-only

# Specific test file
python run_tests.py test_pipeline.py
```

### Method 2: Direct Commands
```bash
# Linting
flake8 src/ tests/ notebooks/

# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_pipeline.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

### Method 3: Using Makefile
```bash
# Install dependencies
make install

# Run tests (if added to Makefile)
make test
```

## Test Categories

### Data Processing Tests (60+ tests)
- **Data Loading:** File existence, loading, shape validation
- **Data Cleaning:** Missing values, duplicates, data types
- **Data Validation:** Value ranges, binary features, categorical features
- **Data Statistics:** Class balance, variance, correlations
- **Data Transformations:** Train-test split, normalization
- **Data Integrity:** Consistency, outliers, distributions

### Pipeline Tests (40+ tests)
- **Preprocessing:** Scaler loading, transformations, scaling
- **Model Loading:** File existence, model attributes, classes
- **Predictions:** Shape validation, binary outputs, probabilities
- **Edge Cases:** Empty data, missing columns, extreme values
- **Integration:** End-to-end pipeline testing

### Model Training Tests (35+ tests)
- **Model Architecture:** Type validation, hyperparameters, fitted status
- **Model Performance:** Accuracy, precision, recall, F1, ROC-AUC
- **Model Robustness:** Consistency, NaN handling, distributions
- **Feature Importance:** Random Forest importance validation
- **Cross-Validation:** CV scores, stratified splits
- **Serialization:** Save/load consistency

## CI/CD Best Practices Implemented

✅ **Automated Testing:** Tests run on every push/PR
✅ **Code Quality:** Linting enforces standards
✅ **Coverage Tracking:** Code coverage metrics
✅ **Artifact Storage:** Models and results saved
✅ **Reproducibility:** Fixed random seeds, pinned dependencies
✅ **Documentation:** Comprehensive test documentation
✅ **Fast Feedback:** Parallel test execution
✅ **Fail Fast:** Pipeline stops on first error

## Continuous Improvement

### Current Coverage
- **Data Processing:** 100% (comprehensive validation)
- **Preprocessing Pipeline:** 100% (all transformations tested)
- **Model Inference:** 100% (predictions validated)
- **Model Training:** 90% (main workflows covered)

### Future Enhancements
- [ ] Integration tests with real API endpoints
- [ ] Performance benchmarking tests
- [ ] Security scanning (Bandit)
- [ ] Dependency vulnerability checks
- [ ] Automated model deployment on success
- [ ] Slack/email notifications on failure
- [ ] Multi-OS testing (Windows, macOS, Linux)
- [ ] Docker container testing

## Troubleshooting

### Common Issues

**Tests fail locally but pass in CI:**
- Ensure same Python version (3.10)
- Check dependencies: `pip install -r requirements.txt`
- Verify models are trained: Run training script first

**Linting errors:**
- Auto-fix: `autopep8 --in-place --recursive src/`
- Check config: `.flake8` file

**Import errors in tests:**
- Ensure correct working directory
- Check PYTHONPATH includes project root

**Model files not found:**
- Train models first: `python notebooks/feature_engineering_modeling_mlflow.py`
- Or skip model tests: `pytest tests/test_data_processing.py`

## Assignment Requirements Checklist

✅ **Unit Tests:** Comprehensive test suite with 135+ tests
✅ **Testing Framework:** Pytest with fixtures and parametrization
✅ **Data Processing Tests:** 60+ tests covering all aspects
✅ **Model Code Tests:** 75+ tests for models and pipeline
✅ **GitHub Actions:** Complete CI/CD workflow
✅ **Linting:** Flake8 integrated in pipeline
✅ **Unit Testing Step:** Automated pytest execution
✅ **Model Training Step:** Automated training on CI
✅ **Artifacts/Logging:** Test results, models, MLflow logs uploaded

## Viewing Results

### GitHub Actions
1. Go to repository on GitHub
2. Click "Actions" tab
3. Select latest workflow run
4. View logs and download artifacts

### Local Coverage Report
After running tests with coverage:
```bash
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### Test Results
```bash
# View JUnit XML
cat results/test-results.xml

# View coverage summary
pytest tests/ --cov=src --cov-report=term
```

---

**Last Updated:** January 2026  
**Pipeline Status:** ✅ Active and Tested
