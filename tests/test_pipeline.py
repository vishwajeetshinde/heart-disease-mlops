"""
Unit Tests for Heart Disease MLOps Pipeline
Tests cover data processing, preprocessing, model loading, and predictions
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.preprocessing_pipeline import load_preprocessing_pipeline, preprocess_input

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Fixtures for reusable test data
@pytest.fixture
def sample_columns():
    """Returns the expected feature columns"""
    return ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
            'ca', 'thal']


@pytest.fixture
def sample_data(sample_columns):
    """Returns valid sample data for testing"""
    return pd.DataFrame({
        'age': [63, 67, 37, 41, 56],
        'sex': [1, 1, 1, 0, 1],
        'cp': [3, 0, 2, 1, 1],
        'trestbps': [145, 160, 130, 130, 120],
        'chol': [233, 286, 250, 204, 236],
        'fbs': [1, 0, 0, 0, 0],
        'restecg': [0, 0, 1, 0, 1],
        'thalach': [150, 108, 187, 172, 178],
        'exang': [0, 1, 0, 0, 0],
        'oldpeak': [2.3, 1.5, 3.5, 1.4, 0.8],
        'slope': [0, 1, 0, 2, 2],
        'ca': [0, 3, 0, 0, 0],
        'thal': [1, 2, 2, 2, 2]
    })


@pytest.fixture
def invalid_data():
    """Returns invalid data with wrong columns for negative testing"""
    return pd.DataFrame({
        'wrong_col1': [1, 2, 3],
        'wrong_col2': [4, 5, 6]
    })


# ============================================
# DATA VALIDATION TESTS
# ============================================

class TestDataValidation:
    """Test data validation and structure"""

    def test_data_shape(self, sample_data):
        """Test if sample data has correct shape"""
        assert sample_data.shape == (5, 13), "Sample data should have 5 rows and 13 columns"

    def test_data_columns(self, sample_data, sample_columns):
        """Test if data has all required columns"""
        assert list(sample_data.columns) == sample_columns, "Data columns don't match expected columns"

    def test_data_types(self, sample_data):
        """Test if data types are numeric"""
        for col in sample_data.columns:
            assert pd.api.types.is_numeric_dtype(sample_data[col]), f"Column {col} should be numeric"

    def test_no_missing_values(self, sample_data):
        """Test that sample data has no missing values"""
        assert sample_data.isnull().sum().sum() == 0, "Sample data should not have missing values"

    def test_data_ranges(self, sample_data):
        """Test if data values are within reasonable ranges"""
        assert (sample_data['age'] >= 0).all() and (sample_data['age'] <= 120).all(), "Age should be between 0-120"
        assert sample_data['sex'].isin([0, 1]).all(), "Sex should be binary (0 or 1)"
        assert (sample_data['trestbps'] > 0).all(), "Blood pressure should be positive"
        assert (sample_data['chol'] > 0).all(), "Cholesterol should be positive"


# ============================================
# PREPROCESSING PIPELINE TESTS
# ============================================

class TestPreprocessingPipeline:
    """Test preprocessing pipeline functionality"""

    def test_scaler_exists(self):
        """Test if scaler file exists"""
        scaler_path = "models/scaler.pkl"
        assert os.path.exists(scaler_path), f"Scaler file not found at {scaler_path}"

    def test_scaler_load(self):
        """Test if scaler loads correctly"""
        scaler = load_preprocessing_pipeline("models/scaler.pkl")
        assert scaler is not None, "Scaler should not be None"
        assert isinstance(scaler, StandardScaler), "Scaler should be StandardScaler instance"

    def test_scaler_has_required_attributes(self):
        """Test if scaler has required methods and attributes"""
        scaler = load_preprocessing_pipeline("models/scaler.pkl")
        assert hasattr(scaler, 'transform'), "Scaler should have transform method"
        assert hasattr(scaler, 'mean_'), "Scaler should have mean_ attribute"
        assert hasattr(scaler, 'scale_'), "Scaler should have scale_ attribute"

    def test_preprocess_input_shape(self, sample_data):
        """Test if preprocessing maintains data shape"""
        scaler = load_preprocessing_pipeline("models/scaler.pkl")
        X_scaled = preprocess_input(sample_data, scaler)
        assert X_scaled.shape == sample_data.shape, "Scaled data should have same shape as input"

    def test_preprocess_output_type(self, sample_data):
        """Test if preprocessing returns numpy array"""
        scaler = load_preprocessing_pipeline("models/scaler.pkl")
        X_scaled = preprocess_input(sample_data, scaler)
        assert isinstance(X_scaled, np.ndarray), "Preprocessed output should be numpy array"

    def test_preprocess_scaling_mean(self, sample_data):
        """Test if scaled data has approximately zero mean"""
        scaler = load_preprocessing_pipeline("models/scaler.pkl")
        X_scaled = preprocess_input(sample_data, scaler)
        # Mean should be close to zero (within tolerance due to small sample)
        assert np.abs(X_scaled.mean()) < 2.0, "Scaled data mean should be close to zero"

    def test_preprocess_no_nan(self, sample_data):
        """Test that preprocessing doesn't introduce NaN values"""
        scaler = load_preprocessing_pipeline("models/scaler.pkl")
        X_scaled = preprocess_input(sample_data, scaler)
        assert not np.isnan(X_scaled).any(), "Scaled data should not contain NaN values"

    def test_preprocess_wrong_columns_fails(self, invalid_data):
        """Test that preprocessing fails with wrong columns"""
        scaler = load_preprocessing_pipeline("models/scaler.pkl")
        with pytest.raises(Exception):
            preprocess_input(invalid_data, scaler)


# ============================================
# MODEL LOADING TESTS
# ============================================

class TestModelLoading:
    """Test model loading and validation"""

    def test_logistic_regression_exists(self):
        """Test if Logistic Regression model file exists"""
        model_path = "models/logistic_regression.pkl"
        assert os.path.exists(model_path), f"Logistic Regression model not found at {model_path}"

    def test_random_forest_exists(self):
        """Test if Random Forest model file exists"""
        model_path = "models/random_forest.pkl"
        assert os.path.exists(model_path), f"Random Forest model not found at {model_path}"

    def test_logistic_regression_load(self):
        """Test if Logistic Regression model loads correctly"""
        model = joblib.load("models/logistic_regression.pkl")
        assert model is not None, "Model should not be None"
        assert hasattr(model, 'predict'), "Model should have predict method"
        assert hasattr(model, 'predict_proba'), "Model should have predict_proba method"

    def test_random_forest_load(self):
        """Test if Random Forest model loads correctly"""
        model = joblib.load("models/random_forest.pkl")
        assert model is not None, "Model should not be None"
        assert hasattr(model, 'predict'), "Model should have predict method"
        assert hasattr(model, 'predict_proba'), "Model should have predict_proba method"

    def test_model_has_classes(self):
        """Test if model has classes_ attribute"""
        model = joblib.load("models/logistic_regression.pkl")
        assert hasattr(model, 'classes_'), "Model should have classes_ attribute"
        assert len(model.classes_) == 2, "Binary classification should have 2 classes"

    def test_model_classes_are_valid(self):
        """Test if model classes are 0 and 1"""
        model = joblib.load("models/logistic_regression.pkl")
        assert set(model.classes_) == {0, 1}, "Classes should be 0 and 1"


# ============================================
# MODEL PREDICTION TESTS
# ============================================

class TestModelPredictions:
    """Test model prediction functionality"""

    def test_lr_prediction_shape(self, sample_data):
        """Test if Logistic Regression predictions have correct shape"""
        model = joblib.load("models/logistic_regression.pkl")
        scaler = load_preprocessing_pipeline("models/scaler.pkl")
        X_scaled = preprocess_input(sample_data, scaler)
        predictions = model.predict(X_scaled)
        assert predictions.shape == (len(sample_data),), "Predictions should match number of samples"

    def test_rf_prediction_shape(self, sample_data):
        """Test if Random Forest predictions have correct shape"""
        model = joblib.load("models/random_forest.pkl")
        scaler = load_preprocessing_pipeline("models/scaler.pkl")
        X_scaled = preprocess_input(sample_data, scaler)
        predictions = model.predict(X_scaled)
        assert predictions.shape == (len(sample_data),), "Predictions should match number of samples"

    def test_prediction_values_are_binary(self, sample_data):
        """Test if predictions are binary (0 or 1)"""
        model = joblib.load("models/logistic_regression.pkl")
        scaler = load_preprocessing_pipeline("models/scaler.pkl")
        X_scaled = preprocess_input(sample_data, scaler)
        predictions = model.predict(X_scaled)
        assert set(predictions).issubset({0, 1}), "Predictions should be 0 or 1"

    def test_predict_proba_shape(self, sample_data):
        """Test if predict_proba returns correct shape"""
        model = joblib.load("models/logistic_regression.pkl")
        scaler = load_preprocessing_pipeline("models/scaler.pkl")
        X_scaled = preprocess_input(sample_data, scaler)
        probabilities = model.predict_proba(X_scaled)
        assert probabilities.shape == (len(sample_data), 2), "Probabilities should be (n_samples, 2)"

    def test_predict_proba_sums_to_one(self, sample_data):
        """Test if probabilities sum to 1 for each sample"""
        model = joblib.load("models/logistic_regression.pkl")
        scaler = load_preprocessing_pipeline("models/scaler.pkl")
        X_scaled = preprocess_input(sample_data, scaler)
        probabilities = model.predict_proba(X_scaled)
        row_sums = probabilities.sum(axis=1)
        assert np.allclose(row_sums, 1.0), "Probabilities should sum to 1 for each sample"

    def test_predict_proba_range(self, sample_data):
        """Test if probabilities are between 0 and 1"""
        model = joblib.load("models/logistic_regression.pkl")
        scaler = load_preprocessing_pipeline("models/scaler.pkl")
        X_scaled = preprocess_input(sample_data, scaler)
        probabilities = model.predict_proba(X_scaled)
        assert (probabilities >= 0).all() and (probabilities <= 1).all(), "Probabilities should be between 0 and 1"

    def test_single_sample_prediction(self, sample_data):
        """Test prediction on single sample"""
        model = joblib.load("models/logistic_regression.pkl")
        scaler = load_preprocessing_pipeline("models/scaler.pkl")
        single_sample = sample_data.iloc[[0]]
        X_scaled = preprocess_input(single_sample, scaler)
        prediction = model.predict(X_scaled)
        assert len(prediction) == 1, "Single sample should return single prediction"
        assert prediction[0] in [0, 1], "Prediction should be 0 or 1"

    def test_batch_prediction_consistency(self, sample_data):
        """Test if batch prediction equals individual predictions"""
        model = joblib.load("models/logistic_regression.pkl")
        scaler = load_preprocessing_pipeline("models/scaler.pkl")

        # Batch prediction
        X_scaled_batch = preprocess_input(sample_data, scaler)
        batch_predictions = model.predict(X_scaled_batch)

        # Individual predictions
        individual_predictions = []
        for idx in range(len(sample_data)):
            single = sample_data.iloc[[idx]]
            X_scaled_single = preprocess_input(single, scaler)
            pred = model.predict(X_scaled_single)[0]
            individual_predictions.append(pred)

        assert np.array_equal(batch_predictions, individual_predictions), \
            "Batch predictions should match individual predictions"


# ============================================
# MODEL COMPARISON TESTS
# ============================================

class TestModelComparison:
    """Test comparing different models"""

    def test_both_models_predict_same_shape(self, sample_data):
        """Test if both models return same prediction shape"""
        lr_model = joblib.load("models/logistic_regression.pkl")
        rf_model = joblib.load("models/random_forest.pkl")
        scaler = load_preprocessing_pipeline("models/scaler.pkl")

        X_scaled = preprocess_input(sample_data, scaler)
        lr_preds = lr_model.predict(X_scaled)
        rf_preds = rf_model.predict(X_scaled)

        assert lr_preds.shape == rf_preds.shape, "Both models should return same shape"

    def test_models_have_same_classes(self):
        """Test if both models have same classes"""
        lr_model = joblib.load("models/logistic_regression.pkl")
        rf_model = joblib.load("models/random_forest.pkl")

        assert np.array_equal(lr_model.classes_, rf_model.classes_), \
            "Both models should have same classes"


# ============================================
# EDGE CASE TESTS
# ============================================

class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_dataframe_fails(self):
        """Test that empty dataframe raises error"""
        scaler = load_preprocessing_pipeline("models/scaler.pkl")
        empty_df = pd.DataFrame()
        with pytest.raises(Exception):
            preprocess_input(empty_df, scaler)

    def test_missing_column_fails(self, sample_columns):
        """Test that missing columns raise error"""
        scaler = load_preprocessing_pipeline("models/scaler.pkl")
        incomplete_df = pd.DataFrame({col: [1] for col in sample_columns[:-1]})  # Missing last column
        with pytest.raises(Exception):
            preprocess_input(incomplete_df, scaler)

    def test_extra_column_handling(self, sample_data):
        """Test behavior with extra columns"""
        scaler = load_preprocessing_pipeline("models/scaler.pkl")
        data_with_extra = sample_data.copy()
        data_with_extra['extra_col'] = [1, 2, 3, 4, 5]
        # This should ideally raise an error or ignore extra columns
        # Depending on implementation, adjust assertion
        with pytest.raises(Exception):
            preprocess_input(data_with_extra, scaler)

    def test_extreme_values(self, sample_columns):
        """Test model with extreme but valid values"""
        model = joblib.load("models/logistic_regression.pkl")
        scaler = load_preprocessing_pipeline("models/scaler.pkl")

        extreme_data = pd.DataFrame({
            'age': [120],
            'sex': [1],
            'cp': [3],
            'trestbps': [200],
            'chol': [600],
            'fbs': [1],
            'restecg': [2],
            'thalach': [200],
            'exang': [1],
            'oldpeak': [6.0],
            'slope': [2],
            'ca': [4],
            'thal': [3]
        })

        X_scaled = preprocess_input(extreme_data, scaler)
        prediction = model.predict(X_scaled)
        assert prediction[0] in [0, 1], "Should handle extreme values and return valid prediction"


# ============================================
# INTEGRATION TESTS
# ============================================

class TestEndToEndPipeline:
    """Test complete end-to-end pipeline"""

    def test_full_pipeline_lr(self, sample_data):
        """Test complete pipeline from raw data to prediction (Logistic Regression)"""
        # Load components
        scaler = load_preprocessing_pipeline("models/scaler.pkl")
        model = joblib.load("models/logistic_regression.pkl")

        # Preprocess
        X_scaled = preprocess_input(sample_data, scaler)

        # Predict
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)

        # Validate
        assert predictions.shape == (len(sample_data),), "Predictions shape incorrect"
        assert probabilities.shape == (len(sample_data), 2), "Probabilities shape incorrect"
        assert set(predictions).issubset({0, 1}), "Invalid prediction values"

    def test_full_pipeline_rf(self, sample_data):
        """Test complete pipeline from raw data to prediction (Random Forest)"""
        # Load components
        scaler = load_preprocessing_pipeline("models/scaler.pkl")
        model = joblib.load("models/random_forest.pkl")

        # Preprocess
        X_scaled = preprocess_input(sample_data, scaler)

        # Predict
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)

        # Validate
        assert predictions.shape == (len(sample_data),), "Predictions shape incorrect"
        assert probabilities.shape == (len(sample_data), 2), "Probabilities shape incorrect"
        assert set(predictions).issubset({0, 1}), "Invalid prediction values"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
