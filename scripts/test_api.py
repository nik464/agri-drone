#!/usr/bin/env python
"""
test_api.py - Test the detection API endpoint.

Usage:
    # Start the server in one terminal:
    uvicorn agridrone.api.app:app --reload --host 127.0.0.1 --port 8000

    # In another terminal, run this test:
    python scripts/test_api.py
"""

import sys
from pathlib import Path

import requests
from loguru import logger

# Setup logging
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Test configuration
API_BASE_URL = "http://127.0.0.1:8000"
DETECT_ENDPOINT = f"{API_BASE_URL}/api/detect/"


def test_health_check():
    """Test health check endpoint."""
    logger.info("Testing health check endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        logger.info(f"Status: {response.status_code}")
        logger.info(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False


def test_detector_health():
    """Test detector health endpoint."""
    logger.info("Testing detector health endpoint...")
    try:
        response = requests.get(f"{DETECT_ENDPOINT}health")
        logger.info(f"Status: {response.status_code}")
        logger.info(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Detector health check failed: {e}")
        return False


def test_detect_with_synthetic_image():
    """Test detection endpoint with synthetic image."""
    logger.info("Testing detection endpoint with synthetic image...")

    try:
        # Generate synthetic test image
        import cv2
        import numpy as np

        # Create a test image
        image = np.ones((640, 480, 3), dtype=np.uint8) * 200

        # Add some colored regions to simulate detections
        cv2.rectangle(image, (100, 100), (200, 200), (0, 255, 0), -1)  # Green box
        cv2.rectangle(image, (300, 150), (400, 250), (0, 0, 255), -1)   # Red box
        cv2.rectangle(image, (50, 350), (150, 450), (255, 0, 0), -1)    # Blue box

        # Encode to JPEG
        success, image_bytes = cv2.imencode('.jpg', image)

        if not success:
            logger.error("Failed to encode synthetic image")
            return False

        logger.info(f"Generated synthetic image: {image.shape}")

        # Send to API
        files = {'file': ('test_image.jpg', image_bytes.tobytes(), 'image/jpeg')}
        params = {'confidence_threshold': 0.5}

        logger.info(f"Sending request to {DETECT_ENDPOINT}")
        response = requests.post(DETECT_ENDPOINT, files=files, params=params, timeout=30)

        logger.info(f"Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            logger.info(f"Response: {result}")

            if result.get('status') == 'success':
                num_dets = result.get('num_detections', 0)
                proc_time = result.get('processing_time_ms', 'N/A')
                logger.info(f"SUCCESS: Found {num_dets} detections in {proc_time}ms")
                return True
            else:
                logger.error(f"Detection failed: {result}")
                return False
        else:
            logger.error(f"Error response: {response.text}")
            return False

    except Exception as e:
        logger.error(f"Detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("AgriDrone Detection API Tests")
    logger.info("=" * 60)

    tests = [
        ("Health Check", test_health_check),
        ("Detector Health", test_detector_health),
        ("Synthetic Image Detection", test_detect_with_synthetic_image),
    ]

    results = []
    for test_name, test_func in tests:
        logger.info("")
        logger.info(f"Running: {test_name}")
        logger.info("-" * 60)

        try:
            success = test_func()
            results.append((test_name, "PASSED" if success else "FAILED"))
        except Exception as e:
            logger.exception(f"Test {test_name} crashed: {e}")
            results.append((test_name, "ERROR"))

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)

    for test_name, status in results:
        status_icon = "✓" if status == "PASSED" else "✗"
        logger.info(f"{status_icon} {test_name}: {status}")

    passed = sum(1 for _, status in results if status == "PASSED")
    total = len(results)
    logger.info(f"\nTotal: {passed}/{total} passed")

    return all(status == "PASSED" for _, status in results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
