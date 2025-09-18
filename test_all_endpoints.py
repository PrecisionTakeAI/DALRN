#!/usr/bin/env python
"""Test all DALRN gateway endpoints"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_endpoint(method, path, data=None, expected_status=200):
    """Test a single endpoint"""
    url = f"{BASE_URL}{path}"
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)

        status = "PASS" if response.status_code == expected_status else "FAIL"
        print(f"{status} {method:4} {path:30} - Status: {response.status_code}")

        if response.status_code == expected_status and response.status_code == 200:
            try:
                data = response.json()
                if isinstance(data, dict):
                    keys = list(data.keys())[:3]
                    print(f"     Response keys: {keys}")
            except:
                pass

        return response.status_code == expected_status
    except Exception as e:
        print(f"FAIL {method:4} {path:30} - Error: {str(e)}")
        return False

def main():
    print("=" * 60)
    print("DALRN Gateway API Endpoint Test")
    print("=" * 60)

    results = []

    # Test all GET endpoints
    get_endpoints = [
        ("/", 200),
        ("/health", 200),
        ("/healthz", 200),
        ("/agents-fast", 200),
        ("/metrics-fast", 200),
        ("/perf-test", 200),
        ("/status/test_id", 404),  # Expected 404 for non-existent ID
        ("/docs", 200),
    ]

    print("\nGET Endpoints:")
    print("-" * 60)
    for path, expected in get_endpoints:
        result = test_endpoint("GET", path, expected_status=expected)
        results.append(result)

    # Test POST endpoint with valid data
    print("\nPOST Endpoints:")
    print("-" * 60)

    valid_dispute = {
        "parties": ["party1", "party2"],
        "jurisdiction": "US",
        "cid": "QmTestCid12345678901234567890",
        "enc_meta": {"test": "data"}
    }

    result = test_endpoint("POST", "/submit-dispute", data=valid_dispute, expected_status=201)
    results.append(result)

    # Test with invalid data (should fail validation)
    invalid_dispute = {
        "parties": ["party1"],  # Only 1 party (minimum is 2)
        "jurisdiction": "us",  # lowercase (should be uppercase)
        "cid": "invalid",  # Too short
    }

    result = test_endpoint("POST", "/submit-dispute", data=invalid_dispute, expected_status=422)
    results.append(result)

    # Summary
    print("\n" + "=" * 60)
    total = len(results)
    passed = sum(results)
    print(f"Test Summary: {passed}/{total} passed ({100*passed/total:.1f}%)")

    if passed == total:
        print("SUCCESS: All endpoints working correctly!")
    else:
        print(f"FAILED: {total - passed} endpoints failed")

    print("=" * 60)

if __name__ == "__main__":
    main()