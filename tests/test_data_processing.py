"""
Unit Tests for Data Processing
Tests cover data loading, cleaning, and validation
"""

import os

import numpy as np
import pandas as pd
import pytest


class TestDataLoading:
    """Test data loading functionality"""

    def test_cleaned_data_exists(self):
        """Test if cleaned data file exists"""
        data_path = "notebooks/data/heart_cleaned.csv"
        assert os.path.exists(data_path), f"Cleaned data not found at {data_path}"

    def test_data_loads_correctly(self):
        """Test if data loads without errors"""
        df = pd.read_csv("notebooks/data/heart_cleaned.csv")
        assert df is not None, "DataFrame should not be None"
        assert len(df) > 0, "DataFrame should not be empty"

    def test_data_shape(self):
        """Test if data has expected shape"""
        df = pd.read_csv("notebooks/data/heart_cleaned.csv")
        assert df.shape[1] == 14, "Should have 14 columns (13 features + 1 target)"
        assert df.shape[0] > 100, "Should have more than 100 samples"

    def test_required_columns_present(self):
        """Test if all required columns are present"""
        df = pd.read_csv("notebooks/data/heart_cleaned.csv")
        required_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                         'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
                         'ca', 'thal', 'target']
        for col in required_cols:
            assert col in df.columns, f"Column {col} is missing"

    def test_no_missing_values(self):
        """Test that cleaned data has no missing values"""
        df = pd.read_csv("notebooks/data/heart_cleaned.csv")
        assert df.isnull().sum().sum() == 0, "Cleaned data should have no missing values"

    def test_all_numeric_columns(self):
        """Test that all columns are numeric"""
        df = pd.read_csv("notebooks/data/heart_cleaned.csv")
        for col in df.columns:
            assert pd.api.types.is_numeric_dtype(df[col]), f"Column {col} should be numeric"


class TestDataCleaning:
    """Test data cleaning operations"""

    def test_target_is_binary(self):
        """Test that target variable is binary (0 or 1)"""
        df = pd.read_csv("notebooks/data/heart_cleaned.csv")
        assert set(df['target'].unique()).issubset({0, 1}), "Target should be binary (0 or 1)"

    def test_no_duplicates(self):
        """Test that there are no duplicate rows"""
        df = pd.read_csv("notebooks/data/heart_cleaned.csv")
        duplicates = df.duplicated().sum()
        # Some duplicates might be acceptable, but shouldn't be too many
        assert duplicates < len(df) * 0.1, "More than 10% duplicates found"

    def test_no_negative_values_in_positive_features(self):
        """Test that features that should be positive are positive"""
        df = pd.read_csv("notebooks/data/heart_cleaned.csv")
        positive_features = ['age', 'trestbps', 'chol', 'thalach']
        for feature in positive_features:
            assert (df[feature] > 0).all(), f"{feature} should be positive"

    def test_age_range(self):
        """Test that age is in reasonable range"""
        df = pd.read_csv("notebooks/data/heart_cleaned.csv")
        assert (df['age'] >= 0).all() and (df['age'] <= 120).all(), \
            "Age should be between 0 and 120"

    def test_binary_features(self):
        """Test that binary features only have 0 or 1"""
        df = pd.read_csv("notebooks/data/heart_cleaned.csv")
        binary_features = ['sex', 'fbs', 'exang']
        for feature in binary_features:
            unique_vals = set(df[feature].unique())
            assert unique_vals.issubset({0, 1}), f"{feature} should be binary (0 or 1)"

    def test_categorical_feature_ranges(self):
        """Test that categorical features are in expected ranges"""
        df = pd.read_csv("notebooks/data/heart_cleaned.csv")

        # cp (chest pain type): 0-4
        assert df["cp"].isin([0, 1, 2, 3, 4]).all(), "cp should be in range 0-4"

        # restecg: 0-2
        assert df['restecg'].isin([0, 1, 2]).all(), "restecg should be in range 0-2"

        # slope: 0-2
        assert df['slope'].isin([0, 1, 2]).all(), "slope should be in range 0-2"

        # ca: 0-4
        assert df['ca'].isin([0, 1, 2, 3, 4]).all(), "ca should be in range 0-4"

        # thal: 0-3
        assert df['thal'].isin([0, 1, 2, 3]).all(), "thal should be in range 0-3"


class TestDataStatistics:
    """Test data statistics and distributions"""

    def test_class_balance(self):
        """Test that classes are somewhat balanced (not severely imbalanced)"""
        df = pd.read_csv("notebooks/data/heart_cleaned.csv")
        class_counts = df['target'].value_counts()
        ratio = min(class_counts) / max(class_counts)
        assert ratio > 0.3, "Classes are severely imbalanced (ratio < 0.3)"

    def test_feature_variance(self):
        """Test that features have non-zero variance"""
        df = pd.read_csv("notebooks/data/heart_cleaned.csv")
        for col in df.columns:
            if col != 'target':
                variance = df[col].var()
                assert variance > 0, f"Feature {col} has zero variance"

    def test_no_constant_features(self):
        """Test that no features are constant"""
        df = pd.read_csv("notebooks/data/heart_cleaned.csv")
        for col in df.columns:
            n_unique = df[col].nunique()
            assert n_unique > 1, f"Feature {col} is constant (only {n_unique} unique value)"

    def test_reasonable_correlations(self):
        """Test that not all features are perfectly correlated"""
        df = pd.read_csv("notebooks/data/heart_cleaned.csv")
        corr_matrix = df.corr().abs()
        # Remove diagonal
        np.fill_diagonal(corr_matrix.values, 0)
        # Check for perfect correlations
        perfect_corr = (corr_matrix > 0.99).sum().sum()
        assert perfect_corr == 0, "Found perfectly correlated features"


class TestDataTransformations:
    """Test data transformation operations"""

    def test_data_can_be_split(self):
        """Test that data can be split into features and target"""
        df = pd.read_csv("notebooks/data/heart_cleaned.csv")
        X = df.drop('target', axis=1)
        y = df['target']

        assert X.shape[0] == y.shape[0], "Features and target should have same number of rows"
        assert X.shape[1] == 13, "Features should have 13 columns"

    def test_train_test_split_possible(self):
        """Test that data can be split into train and test sets"""
        from sklearn.model_selection import train_test_split

        df = pd.read_csv("notebooks/data/heart_cleaned.csv")
        X = df.drop('target', axis=1)
        y = df['target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        assert len(X_train) > len(X_test), "Train set should be larger than test set"
        assert len(X_train) + len(X_test) == len(X), "Split should account for all samples"

    def test_data_normalization_possible(self):
        """Test that data can be normalized"""
        from sklearn.preprocessing import StandardScaler

        df = pd.read_csv("notebooks/data/heart_cleaned.csv")
        X = df.drop('target', axis=1)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        assert X_scaled.shape == X.shape, "Normalized data should have same shape"
        assert not np.isnan(X_scaled).any(), "Normalized data should not contain NaN"
        assert not np.isinf(X_scaled).any(), "Normalized data should not contain infinity"


class TestDataIntegrity:
    """Test data integrity and consistency"""

    def test_data_types_consistent(self):
        """Test that data types are consistent"""
        df = pd.read_csv("notebooks/data/heart_cleaned.csv")
        # All columns should be numeric
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        assert len(numeric_cols) == len(df.columns), "All columns should be numeric"

    def test_no_infinite_values(self):
        """Test that there are no infinite values"""
        df = pd.read_csv("notebooks/data/heart_cleaned.csv")
        assert not np.isinf(df.values).any(), "Data should not contain infinite values"

    def test_feature_distributions(self):
        """Test that features have reasonable distributions"""
        df = pd.read_csv("notebooks/data/heart_cleaned.csv")

        # Check that features don't have extreme skewness
        for col in df.columns:
            if col != 'target':
                skew = df[col].skew()
                assert abs(skew) < 10, f"Feature {col} has extreme skewness: {skew}"

    def test_outlier_detection(self):
        """Test basic outlier detection (should have some but not too many)"""
        df = pd.read_csv("notebooks/data/heart_cleaned.csv")

        for col in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < Q1 - 3 * IQR) | (df[col] > Q3 + 3 * IQR)).sum()
            outlier_ratio = outliers / len(df)
            # Should not have more than 10% outliers
            assert outlier_ratio < 0.1, f"Feature {col} has too many outliers: {outlier_ratio:.2%}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
