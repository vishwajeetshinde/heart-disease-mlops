import joblib


def load_preprocessing_pipeline(scaler_path):
    """Load the saved StandardScaler object."""
    return joblib.load(scaler_path)


def preprocess_input(input_df, scaler):
    """Apply the saved scaler to new input data (expects DataFrame with same columns as training)."""
    return scaler.transform(input_df)


if __name__ == "__main__":
    # Example usage
    scaler = load_preprocessing_pipeline("../models/scaler.pkl")
    # Suppose you have new data in a DataFrame 'new_data'
    # new_data_scaled = preprocess_input(new_data, scaler)
    print("Preprocessing pipeline loaded. Ready for inference.")
