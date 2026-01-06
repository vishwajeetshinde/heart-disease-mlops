#!/usr/bin/env python3
"""
Test script for Heart Disease Prediction API
Tests the /predict endpoint with sample data
"""

import requests
import json

# API endpoint
API_URL = "http://localhost:8000"

# Sample test cases
test_cases = [
    {
        "name": "Patient 1 - High Risk",
        "data": {
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
    },
    {
        "name": "Patient 2 - Low Risk",
        "data": {
            "age": 37,
            "sex": 1,
            "cp": 2,
            "trestbps": 130,
            "chol": 250,
            "fbs": 0,
            "restecg": 1,
            "thalach": 187,
            "exang": 0,
            "oldpeak": 3.5,
            "slope": 0,
            "ca": 0,
            "thal": 2
        }
    },
    {
        "name": "Patient 3 - Medium Risk",
        "data": {
            "age": 56,
            "sex": 0,
            "cp": 1,
            "trestbps": 140,
            "chol": 294,
            "fbs": 0,
            "restecg": 0,
            "thalach": 153,
            "exang": 0,
            "oldpeak": 1.3,
            "slope": 1,
            "ca": 0,
            "thal": 2
        }
    }
]


def test_health_check():
    """Test health check endpoint"""
    print("=" * 60)
    print("Testing Health Check Endpoint")
    print("=" * 60)
    
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    print()


def test_model_info():
    """Test model info endpoint"""
    print("=" * 60)
    print("Testing Model Info Endpoint")
    print("=" * 60)
    
    try:
        response = requests.get(f"{API_URL}/model-info")
        if response.status_code == 200:
            print("✅ Model info retrieved")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"❌ Model info failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    print()


def test_prediction(case):
    """Test prediction endpoint"""
    print("=" * 60)
    print(f"Testing Prediction: {case['name']}")
    print("=" * 60)
    
    print("\nInput Data:")
    print(json.dumps(case['data'], indent=2))
    
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=case['data'],
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ Prediction successful")
            print("\nResult:")
            print(json.dumps(result, indent=2))
            
            # Interpretation
            print("\nInterpretation:")
            if result['prediction'] == 0:
                print("  Prediction: No Heart Disease")
            else:
                print("  Prediction: Heart Disease Detected")
            
            print(f"  Confidence: {result['confidence']:.2%}")
            print(f"  Risk Level: {result['risk_level']}")
            
        else:
            print(f"\n❌ Prediction failed: {response.status_code}")
            print(response.text)
    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
    
    print()


def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("=" * 60)
    print("Testing Batch Prediction Endpoint")
    print("=" * 60)
    
    batch_data = {
        "instances": [case['data'] for case in test_cases]
    }
    
    try:
        response = requests.post(
            f"{API_URL}/predict-batch",
            json=batch_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ Batch prediction successful")
            print(f"\nProcessed {result['count']} patients")
            
            for i, pred in enumerate(result['predictions']):
                print(f"\nPatient {i+1}:")
                print(f"  Prediction: {'Disease' if pred['prediction'] == 1 else 'No Disease'}")
                print(f"  Confidence: {pred['confidence']:.2%}")
                print(f"  Risk Level: {pred['risk_level']}")
        else:
            print(f"\n❌ Batch prediction failed: {response.status_code}")
            print(response.text)
    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
    
    print()


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("HEART DISEASE PREDICTION API - TEST SUITE")
    print("=" * 60)
    print(f"\nAPI URL: {API_URL}")
    print()
    
    # Test endpoints
    test_health_check()
    test_model_info()
    
    # Test individual predictions
    for case in test_cases:
        test_prediction(case)
    
    # Test batch prediction
    test_batch_prediction()
    
    print("=" * 60)
    print("TEST SUITE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
