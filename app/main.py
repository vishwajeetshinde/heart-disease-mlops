"""
Heart Disease Prediction API
FastAPI application for serving ML model predictions

Endpoints:
- GET /: Health check
- POST /predict: Get heart disease prediction
- GET /model-info: Get model information
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Heart Disease Prediction API",
    description="API for predicting heart disease using Machine Learning",
    version="1.0.0"
)

# Model paths
MODEL_PATH = "models/random_forest.pkl"
SCALER_PATH = "models/scaler.pkl"

# Global variables for model and scaler
model = None
scaler = None
feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                 'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
                 'ca', 'thal']


class PredictionInput(BaseModel):
    """Input schema for prediction"""
    age: float = Field(..., ge=0, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex (0 = female, 1 = male)")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: float = Field(..., gt=0, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., gt=0, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (0 or 1)")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: float = Field(..., gt=0, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina (0 or 1)")
    oldpeak: float = Field(..., ge=0, description="ST depression induced by exercise")
    slope: int = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment (0-2)")
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels colored by fluoroscopy (0-4)")
    thal: int = Field(..., ge=0, le=3, description="Thalassemia (0-3)")

    class Config:
        schema_extra = {
            "example": {
                "age": 63,
                "sex": 1,
                "cp": 3,
                "trestbps": 145,
                "chol": 233,
                "fbs": 1,
                "restecg": 0,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 2.3,
                "slope": 0,
                "ca": 0,
                "thal": 1
            }
        }


class PredictionOutput(BaseModel):
    """Output schema for prediction"""
    prediction: int = Field(..., description="Predicted class (0 = No Disease, 1 = Disease)")
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    probability_no_disease: float = Field(..., description="Probability of no disease")
    probability_disease: float = Field(..., description="Probability of disease")
    risk_level: str = Field(..., description="Risk level (Low, Medium, High)")


class BatchPredictionInput(BaseModel):
    """Input schema for batch prediction"""
    instances: List[PredictionInput]


def load_model():
    """Load trained model and scaler"""
    global model, scaler
    
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"Scaler file not found at {SCALER_PATH}")
        
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        
        logger.info("Model and scaler loaded successfully")
        logger.info(f"Model type: {type(model).__name__}")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()
    logger.info("API startup complete")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Heart Disease Prediction API is running",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    model_loaded = model is not None
    scaler_loaded = scaler is not None
    
    return {
        "status": "healthy" if (model_loaded and scaler_loaded) else "unhealthy",
        "model_loaded": model_loaded,
        "scaler_loaded": scaler_loaded,
        "model_type": type(model).__name__ if model_loaded else None
    }


@app.get("/model-info")
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    info = {
        "model_type": type(model).__name__,
        "features": feature_names,
        "n_features": len(feature_names),
        "classes": model.classes_.tolist() if hasattr(model, 'classes_') else None
    }
    
    # Add model-specific info
    if hasattr(model, 'n_estimators'):
        info["n_estimators"] = model.n_estimators
    if hasattr(model, 'max_depth'):
        info["max_depth"] = model.max_depth
    
    return info


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """
    Predict heart disease for a single patient
    
    Returns prediction, confidence, probabilities, and risk level
    """
    try:
        if model is None or scaler is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Convert input to DataFrame
        input_dict = input_data.dict()
        input_df = pd.DataFrame([input_dict], columns=feature_names)
        
        logger.info(f"Received prediction request: {input_dict}")
        
        # Scale features
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        
        # Calculate confidence (max probability)
        confidence = float(np.max(probabilities))
        
        # Determine risk level
        disease_prob = float(probabilities[1])
        if disease_prob < 0.3:
            risk_level = "Low"
        elif disease_prob < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        result = {
            "prediction": int(prediction),
            "confidence": confidence,
            "probability_no_disease": float(probabilities[0]),
            "probability_disease": float(probabilities[1]),
            "risk_level": risk_level
        }
        
        logger.info(f"Prediction result: {result}")
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict-batch")
async def predict_batch(input_data: BatchPredictionInput):
    """
    Predict heart disease for multiple patients
    
    Returns predictions for all instances
    """
    try:
        if model is None or scaler is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        results = []
        
        for instance in input_data.instances:
            # Convert to DataFrame
            input_dict = instance.dict()
            input_df = pd.DataFrame([input_dict], columns=feature_names)
            
            # Scale and predict
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            probabilities = model.predict_proba(input_scaled)[0]
            
            confidence = float(np.max(probabilities))
            disease_prob = float(probabilities[1])
            
            if disease_prob < 0.3:
                risk_level = "Low"
            elif disease_prob < 0.7:
                risk_level = "Medium"
            else:
                risk_level = "High"
            
            results.append({
                "prediction": int(prediction),
                "confidence": confidence,
                "probability_no_disease": float(probabilities[0]),
                "probability_disease": float(probabilities[1]),
                "risk_level": risk_level
            })
        
        return {"predictions": results, "count": len(results)}
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
