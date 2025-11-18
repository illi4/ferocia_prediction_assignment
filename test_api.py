"""
Test script for Marketing Prediction API

This script tests all API endpoints to ensure the API is working correctly.

Usage:
    python test_api.py
"""

import requests
import json
import sys
from typing import Dict, Any


# API base URL
BASE_URL = "http://localhost:8000"


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_response(response: requests.Response, show_body: bool = True):
    """Print response details."""
    print(f"Status Code: {response.status_code}")
    if show_body:
        print(f"Response Body:")
        try:
            print(json.dumps(response.json(), indent=2))
        except:
            print(response.text)


def test_root():
    """Test the root endpoint."""
    print_section("TEST 1: Root Endpoint (GET /)")
    
    try:
        response = requests.get(f"{BASE_URL}/")
        print_response(response)
        
        if response.status_code == 200:
            print("✓ Root endpoint working")
            return True
        else:
            print("✗ Root endpoint failed")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to API. Make sure it's running at", BASE_URL)
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_health():
    """Test the health check endpoint."""
    print_section("TEST 2: Health Check (GET /health)")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        print_response(response)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "healthy" and data.get("model_loaded"):
                print("✓ Health check passed - Model is loaded and ready")
                return True
            else:
                print("✗ Model not ready")
                return False
        else:
            print("✗ Health check failed")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_predict_valid():
    """Test the predict endpoint with valid data."""
    print_section("TEST 3: Valid Prediction (POST /predict)")
    
    # Valid test data
    data = {
        "age": 30,
        "job": "technician",
        "marital": "single",
        "education": "secondary",
        "default": "no",
        "balance": 1500,
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "day": 15,
        "month": "may",
        "duration": 180,
        "campaign": 2,
        "pdays": -1,
        "previous": 0,
        "poutcome": "unknown"
    }
    
    print("Input data:")
    print(json.dumps(data, indent=2))
    print()
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=data)
        print_response(response)
        
        if response.status_code == 200:
            result = response.json()
            
            # Validate response structure
            if all(key in result for key in ["prediction", "probability", "confidence"]):
                print("✓ Valid prediction received")
                print(f"  Prediction: {result['prediction']}")
                print(f"  Probability: {result['probability']:.4f}")
                print(f"  Confidence: {result['confidence']}")
                return True
            else:
                print("✗ Response missing required fields")
                return False
        else:
            print("✗ Prediction failed")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_predict_likely_yes():
    """Test prediction with data likely to result in 'yes'."""
    print_section("TEST 4: Likely Positive Prediction")
    
    # Data more likely to subscribe (high duration, good outcome before)
    data = {
        "age": 35,
        "job": "management",
        "marital": "married",
        "education": "tertiary",
        "default": "no",
        "balance": 5000,
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "day": 20,
        "month": "may",
        "duration": 600,  # Long duration
        "campaign": 1,
        "pdays": 30,
        "previous": 2,
        "poutcome": "success"  # Previous success
    }
    
    print("Input data (likely to subscribe):")
    print(json.dumps(data, indent=2))
    print()
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=data)
        print_response(response)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Prediction: {result['prediction']}")
            print(f"  Probability: {result['probability']:.4f}")
            return True
        else:
            print("✗ Prediction failed")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_predict_invalid_job():
    """Test prediction with invalid job category."""
    print_section("TEST 5: Invalid Input - Bad Job Category (Expect Error)")
    
    data = {
        "age": 30,
        "job": "invalid_job",  # Invalid job
        "marital": "single",
        "education": "secondary",
        "default": "no",
        "balance": 1500,
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "day": 15,
        "month": "may",
        "duration": 180,
        "campaign": 2,
        "pdays": -1,
        "previous": 0,
        "poutcome": "unknown"
    }
    
    print("Input data (invalid job):")
    print(json.dumps(data, indent=2))
    print()
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=data)
        print_response(response)
        
        if response.status_code == 422:
            print("✓ Correctly rejected invalid input (422)")
            return True
        else:
            print("✗ Should have returned 422 for invalid input")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_predict_missing_field():
    """Test prediction with missing required field."""
    print_section("TEST 6: Missing Required Field (Expect Error)")
    
    data = {
        "age": 30,
        "job": "technician",
        # Missing 'marital' field
        "education": "secondary",
        "default": "no",
        "balance": 1500,
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "day": 15,
        "month": "may",
        "duration": 180,
        "campaign": 2,
        "pdays": -1,
        "previous": 0,
        "poutcome": "unknown"
    }
    
    print("Input data (missing 'marital'):")
    print(json.dumps(data, indent=2))
    print()
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=data)
        print_response(response)
        
        if response.status_code == 422:
            print("✓ Correctly rejected missing field (422)")
            return True
        else:
            print("✗ Should have returned 422 for missing field")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_predict_multiple():
    """Test multiple predictions in sequence."""
    print_section("TEST 7: Multiple Sequential Predictions")
    
    test_cases = [
        {
            "name": "Young student",
            "data": {
                "age": 22,
                "job": "student",
                "marital": "single",
                "education": "tertiary",
                "default": "no",
                "balance": 100,
                "housing": "no",
                "loan": "no",
                "contact": "cellular",
                "day": 10,
                "month": "may",
                "duration": 120,
                "campaign": 1,
                "pdays": -1,
                "previous": 0,
                "poutcome": "unknown"
            }
        },
        {
            "name": "Retired with savings",
            "data": {
                "age": 65,
                "job": "retired",
                "marital": "married",
                "education": "secondary",
                "default": "no",
                "balance": 10000,
                "housing": "no",
                "loan": "no",
                "contact": "cellular",
                "day": 25,
                "month": "jun",
                "duration": 300,
                "campaign": 1,
                "pdays": -1,
                "previous": 0,
                "poutcome": "unknown"
            }
        },
        {
            "name": "Blue collar worker",
            "data": {
                "age": 40,
                "job": "blue-collar",
                "marital": "married",
                "education": "primary",
                "default": "no",
                "balance": 500,
                "housing": "yes",
                "loan": "yes",
                "contact": "telephone",
                "day": 5,
                "month": "nov",
                "duration": 90,
                "campaign": 3,
                "pdays": -1,
                "previous": 0,
                "poutcome": "unknown"
            }
        }
    ]
    
    success_count = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest case {i}: {test_case['name']}")
        print("-" * 40)
        
        try:
            response = requests.post(f"{BASE_URL}/predict", json=test_case['data'])
            
            if response.status_code == 200:
                result = response.json()
                print(f"  Prediction: {result['prediction']}")
                print(f"  Probability: {result['probability']:.4f}")
                print(f"  Confidence: {result['confidence']}")
                success_count += 1
            else:
                print(f"  Failed with status code: {response.status_code}")
        except Exception as e:
            print(f"  Error: {e}")
    
    print(f"\n✓ {success_count}/{len(test_cases)} predictions successful")
    return success_count == len(test_cases)


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 80)
    print("BANK MARKETING API TEST SUITE")
    print("=" * 80)
    print(f"Testing API at: {BASE_URL}")
    
    tests = [
        ("Root Endpoint", test_root),
        ("Health Check", test_health),
        ("Valid Prediction", test_predict_valid),
        ("Likely Yes Prediction", test_predict_likely_yes),
        ("Invalid Job Category", test_predict_invalid_job),
        ("Missing Required Field", test_predict_missing_field),
        ("Multiple Predictions", test_predict_multiple),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print_section("TEST SUMMARY")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print("\n" + "=" * 80)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 80)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
