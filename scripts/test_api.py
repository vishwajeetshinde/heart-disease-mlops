#!/usr/bin/env python3
"""
Manual smoke-test script for Heart Disease Prediction API.

Run this AFTER starting the API locally, e.g.
  uvicorn src.api:app --host 0.0.0.0 --port 8000

It calls:
- GET  /health           (optional)
- GET  /model-info       (optional)
- POST /predict          (required by assignment)
- POST /predict-batch    (optional)
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")
TIMEOUT = 10  # seconds


TEST_CASES = [
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
            "thal": 1,
        },
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
            "thal": 2,
        },
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
            "thal": 2,
        },
    },
]


def _request(method: str, path: str, **kwargs) -> requests.Response:
    url = f"{API_URL}{path}"
    return requests.request(method, url, timeout=TIMEOUT, **kwargs)


def _print_response(resp: requests.Response) -> None:
    print(f"Status: {resp.status_code}")
    content_type = resp.headers.get("content-type", "")

    if "application/json" in content_type:
        try:
            print(json.dumps(resp.json(), indent=2))
        except Exception:
            print(resp.text)
    else:
        print(resp.text)


def health_check() -> None:
    print("=" * 60)
    print("GET /health")
    print("=" * 60)

    try:
        resp = _request("GET", "/health")
        _print_response(resp)
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API. Is it running?")
    except Exception as e:
        print(f"Error: {e}")
    print()


def model_info() -> None:
    print("=" * 60)
    print("GET /model-info (optional)")
    print("=" * 60)

    try:
        resp = _request("GET", "/model-info")
        _print_response(resp)
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API. Is it running?")
    except Exception as e:
        print(f"Error: {e}")
    print()


def predict_one(case: Dict[str, Any]) -> None:
    print("=" * 60)
    print(f"POST /predict :: {case['name']}")
    print("=" * 60)

    print("Input:")
    print(json.dumps(case["data"], indent=2))

    try:
        resp = _request(
            "POST",
            "/predict",
            json=case["data"],
            headers={"Content-Type": "application/json"},
        )
        print("\nResponse:")
        _print_response(resp)

        if resp.ok and "application/json" in resp.headers.get("content-type", ""):
            result = resp.json()
            if "prediction" in result:
                print("\nInterpretation:")
                pred = result["prediction"]
                print(f"prediction: {pred} (0=No Disease, 1=Disease)")
                if "confidence" in result:
                    print(f"confidence: {result['confidence']}")
                if "risk_level" in result:
                    print(f"risk_level: {result['risk_level']}")

    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API. Is it running?")
    except Exception as e:
        print(f"Error: {e}")

    print()


def predict_batch() -> None:
    print("=" * 60)
    print("POST /predict-batch (optional)")
    print("=" * 60)

    payload = {"instances": [c["data"] for c in TEST_CASES]}

    try:
        resp = _request(
            "POST",
            "/predict-batch",
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        _print_response(resp)
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API. Is it running?")
    except Exception as e:
        print(f"Error: {e}")
    print()


def main() -> None:
    print("\n" + "=" * 60)
    print("HEART DISEASE API - SMOKE TEST")
    print("=" * 60)
    print(f"API_URL = {API_URL}\n")

    # Optional endpoints (won't fail the script if they return 404)
    health_check()
    model_info()

    # Required endpoint for assignment
    for case in TEST_CASES:
        predict_one(case)

    # Optional
    predict_batch()

    print("=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
