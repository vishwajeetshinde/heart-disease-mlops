"""
Unit Tests for Model Training and Evaluation
Tests cover model performance, training, and evaluation metrics
"""


import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


@pytest.fixture
def test_data():
    """Load test data for model evaluation"""
    df = pd.read_csv("notebooks/data/heart_cleaned.csv")
    return df


@pytest.fixture
def models():
    """Load trained models"""
    lr_model = joblib.load("models/logistic_regression.pkl")
    rf_model = joblib.load("models/random_forest.pkl")
    return {'lr': lr_model, 'rf': rf_model}


@pytest.fixture
def scaler():
    """Load scaler"""
    return joblib.load("models/scaler.pkl")


class TestModelArchitecture:
    """Test model architecture and properties"""

    def test_lr_model_type(self, models):
        """Test Logistic Regression model type"""
        from sklearn.linear_model import LogisticRegression
        assert isinstance(models['lr'], LogisticRegression), \
            "Model should be LogisticRegression instance"

    def test_rf_model_type(self, models):
        """Test Random Forest model type"""
        from sklearn.ensemble import RandomForestClassifier
        assert isinstance(models['rf'], RandomForestClassifier), \
            "Model should be RandomForestClassifier instance"

    def test_lr_has_hyperparameters(self, models):
        """Test that Logistic Regression has tuned hyperparameters"""
        lr = models['lr']
        assert hasattr(lr, 'C'), "Model should have C parameter"
        assert hasattr(lr, 'penalty'), "Model should have penalty parameter"
        assert lr.C > 0, "C should be positive"

    def test_rf_has_hyperparameters(self, models):
        """Test that Random Forest has tuned hyperparameters"""
        rf = models['rf']
        assert hasattr(rf, 'n_estimators'), "Model should have n_estimators"
        assert hasattr(rf, 'max_depth'), "Model should have max_depth"
        assert rf.n_estimators > 0, "n_estimators should be positive"

    def test_rf_ensemble_properties(self, models):
        """Test Random Forest ensemble properties"""
        rf = models['rf']
        assert hasattr(rf, 'estimators_'), "Random Forest should have estimators"
        assert len(rf.estimators_) == rf.n_estimators, \
            "Number of estimators should match n_estimators parameter"

    def test_models_are_fitted(self, models):
        """Test that models are fitted"""
        for name, model in models.items():
            # Fitted models should have these attributes
            assert hasattr(model, 'classes_'), f"{name} should be fitted (missing classes_)"
            if hasattr(model, 'coef_'):  # For linear models
                assert model.coef_ is not None


class TestModelPerformance:
    """Test model performance metrics"""

    def test_lr_minimum_accuracy(self, models, test_data, scaler):
        """Test that Logistic Regression achieves minimum accuracy"""
        from sklearn.model_selection import train_test_split

        X = test_data.drop('target', axis=1)
        y = test_data['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_test_scaled = scaler.transform(X_test)
        y_pred = models['lr'].predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        assert accuracy > 0.6, f"Logistic Regression accuracy too low: {accuracy:.3f}"

    def test_rf_minimum_accuracy(self, models, test_data, scaler):
        """Test that Random Forest achieves minimum accuracy"""
        from sklearn.model_selection import train_test_split

        X = test_data.drop('target', axis=1)
        y = test_data['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_test_scaled = scaler.transform(X_test)
        y_pred = models['rf'].predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        assert accuracy > 0.6, f"Random Forest accuracy too low: {accuracy:.3f}"

    def test_lr_precision_recall(self, models, test_data, scaler):
        """Test Logistic Regression precision and recall"""
        from sklearn.model_selection import train_test_split

        X = test_data.drop('target', axis=1)
        y = test_data['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_test_scaled = scaler.transform(X_test)
        y_pred = models['lr'].predict(X_test_scaled)

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        assert precision > 0.5, f"Precision too low: {precision:.3f}"
        assert recall > 0.5, f"Recall too low: {recall:.3f}"

    def test_rf_precision_recall(self, models, test_data, scaler):
        """Test Random Forest precision and recall"""
        from sklearn.model_selection import train_test_split

        X = test_data.drop('target', axis=1)
        y = test_data['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_test_scaled = scaler.transform(X_test)
        y_pred = models['rf'].predict(X_test_scaled)

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        assert precision > 0.5, f"Precision too low: {precision:.3f}"
        assert recall > 0.5, f"Recall too low: {recall:.3f}"

    def test_lr_roc_auc(self, models, test_data, scaler):
        """Test Logistic Regression ROC-AUC score"""
        from sklearn.model_selection import train_test_split

        X = test_data.drop('target', axis=1)
        y = test_data['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_test_scaled = scaler.transform(X_test)
        y_proba = models['lr'].predict_proba(X_test_scaled)[:, 1]

        roc_auc = roc_auc_score(y_test, y_proba)
        assert roc_auc > 0.65, f"ROC-AUC too low: {roc_auc:.3f}"

    def test_rf_roc_auc(self, models, test_data, scaler):
        """Test Random Forest ROC-AUC score"""
        from sklearn.model_selection import train_test_split

        X = test_data.drop('target', axis=1)
        y = test_data['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_test_scaled = scaler.transform(X_test)
        y_proba = models['rf'].predict_proba(X_test_scaled)[:, 1]

        roc_auc = roc_auc_score(y_test, y_proba)
        assert roc_auc > 0.65, f"ROC-AUC too low: {roc_auc:.3f}"

    def test_models_better_than_random(self, models, test_data, scaler):
        """Test that models perform better than random guessing"""
        from sklearn.model_selection import train_test_split

        X = test_data.drop('target', axis=1)
        y = test_data['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_test_scaled = scaler.transform(X_test)

        for name, model in models.items():
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            # Should be significantly better than random (0.5)
            assert accuracy > 0.55, f"{name} accuracy not better than random: {accuracy:.3f}"


class TestModelRobustness:
    """Test model robustness and stability"""

    def test_consistent_predictions(self, models, test_data, scaler):
        """Test that models give consistent predictions on same input"""
        from sklearn.model_selection import train_test_split

        X = test_data.drop('target', axis=1)
        y = test_data['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_test_scaled = scaler.transform(X_test)

        for name, model in models.items():
            pred1 = model.predict(X_test_scaled)
            pred2 = model.predict(X_test_scaled)
            assert np.array_equal(pred1, pred2), f"{name} gives inconsistent predictions"

    def test_no_nan_predictions(self, models, test_data, scaler):
        """Test that models don't produce NaN predictions"""
        from sklearn.model_selection import train_test_split

        X = test_data.drop('target', axis=1)
        y = test_data['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_test_scaled = scaler.transform(X_test)

        for name, model in models.items():
            predictions = model.predict(X_test_scaled)
            assert not np.isnan(predictions).any(), f"{name} produces NaN predictions"

    def test_prediction_distribution(self, models, test_data, scaler):
        """Test that predictions have reasonable distribution"""
        from sklearn.model_selection import train_test_split

        X = test_data.drop('target', axis=1)
        y = test_data['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_test_scaled = scaler.transform(X_test)

        for name, model in models.items():
            predictions = model.predict(X_test_scaled)
            # Should predict both classes (not all 0 or all 1)
            unique_preds = set(predictions)
            assert len(unique_preds) > 1, f"{name} predicts only one class"


class TestFeatureImportance:
    """Test feature importance (for Random Forest)"""

    def test_rf_has_feature_importance(self, models):
        """Test that Random Forest has feature importance"""
        rf = models['rf']
        assert hasattr(rf, 'feature_importances_'), "Random Forest should have feature_importances_"

    def test_feature_importance_sum(self, models):
        """Test that feature importances sum to 1"""
        rf = models['rf']
        importance_sum = rf.feature_importances_.sum()
        assert np.isclose(importance_sum, 1.0), f"Feature importances should sum to 1, got {importance_sum}"

    def test_feature_importance_non_negative(self, models):
        """Test that all feature importances are non-negative"""
        rf = models['rf']
        assert (rf.feature_importances_ >= 0).all(), "Feature importances should be non-negative"

    def test_feature_importance_length(self, models, test_data):
        """Test that feature importance length matches number of features"""
        rf = models['rf']
        n_features = test_data.shape[1] - 1  # Excluding target
        assert len(rf.feature_importances_) == n_features, \
            f"Feature importance length should match number of features ({n_features})"


class TestCrossValidation:
    """Test cross-validation behavior"""

    def test_cross_validation_scores(self, models, test_data, scaler):
        """Test cross-validation produces reasonable scores"""
        from sklearn.model_selection import cross_val_score

        X = test_data.drop('target', axis=1)
        y = test_data['target']
        X_scaled = scaler.fit_transform(X)

        for name, model in models.items():
            cv_scores = cross_val_score(model, X_scaled, y, cv=3, scoring='accuracy')

            # Check that all CV scores are reasonable
            assert (cv_scores > 0.5).all(), f"{name} has poor CV scores"
            assert cv_scores.std() < 0.3, f"{name} has high variance in CV scores"

    def test_stratified_cv_maintains_class_balance(self, test_data):
        """Test that stratified CV maintains class balance"""
        from sklearn.model_selection import StratifiedKFold

        X = test_data.drop('target', axis=1)
        y = test_data['target']

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        original_ratio = y.mean()

        for train_idx, test_idx in skf.split(X, y):
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            train_ratio = y_train.mean()
            test_ratio = y_test.mean()

            # Ratios should be similar to original
            assert abs(train_ratio - original_ratio) < 0.15, "Train set class ratio deviates too much"
            assert abs(test_ratio - original_ratio) < 0.15, "Test set class ratio deviates too much"


class TestModelSerialization:
    """Test model serialization and loading"""

    def test_models_can_be_saved_and_loaded(self, models, tmp_path):
        """Test that models can be saved and loaded correctly"""
        for name, model in models.items():
            # Save to temporary file
            temp_file = tmp_path / f"{name}_temp.pkl"
            joblib.dump(model, temp_file)

            # Load from file
            loaded_model = joblib.load(temp_file)

            # Check that loaded model has same attributes
            assert isinstance(loaded_model, type(model)), f"Loaded {name} has different type"
            assert np.array_equal(loaded_model.classes_, model.classes_), \
                f"Loaded {name} has different classes"

    def test_predictions_same_after_reload(self, models, test_data, scaler, tmp_path):
        """Test that predictions are same after saving and reloading"""
        from sklearn.model_selection import train_test_split

        X = test_data.drop('target', axis=1)
        y = test_data['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_test_scaled = scaler.transform(X_test)

        for name, model in models.items():
            # Get original predictions
            original_preds = model.predict(X_test_scaled)

            # Save and reload
            temp_file = tmp_path / f"{name}_temp.pkl"
            joblib.dump(model, temp_file)
            loaded_model = joblib.load(temp_file)

            # Get new predictions
            new_preds = loaded_model.predict(X_test_scaled)

            # Should be identical
            assert np.array_equal(original_preds, new_preds), \
                f"Predictions changed after reload for {name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
