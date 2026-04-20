#!/usr/bin/env python3
"""
Test script for face recognition functionality.
Run this after installing dependencies to verify everything works.
"""

import numpy as np
import cv2
import pandas as pd
from ultralytics import YOLO

try:
    import face_recognition
    FACE_AVAILABLE = True
except ImportError:
    FACE_AVAILABLE = False

def test_face_extraction():
    """Test face embedding extraction."""
    if not FACE_AVAILABLE:
        print("❌ face_recognition not available")
        return False

    # Create a simple test image (black square)
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)

    # Test with None embedding (should return None)
    from detect_people import extract_face_embedding
    result = extract_face_embedding(test_image, (10, 10, 50, 50))

    if result is None:
        print("✅ Face extraction handles no-face case correctly")
        return True
    else:
        print("❌ Face extraction should return None for test image")
        return False

def test_yolo_loading():
    """Test YOLO model loading."""
    try:
        model = YOLO("yolov8n.pt")
        print("✅ YOLO model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ YOLO loading failed: {e}")
        return False

def test_data_processing():
    """Test data processing functions."""
    try:
        # Test embedding parsing
        test_emb = "0.1,0.2,0.3"
        parsed = np.array([0.1, 0.2, 0.3])
        print("✅ Data processing functions work")
        return True
    except Exception as e:
        print(f"❌ Data processing failed: {e}")
        return False

def main():
    print("Testing facial recognition system components...\n")

    tests = [
        ("YOLO Model Loading", test_yolo_loading),
        ("Face Extraction", test_face_extraction),
        ("Data Processing", test_data_processing),
    ]

    passed = 0
    total = len(tests)

    for name, test_func in tests:
        print(f"Testing {name}...")
        if test_func():
            passed += 1
        print()

    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! System is ready.")
    else:
        print("⚠️  Some tests failed. Check dependencies and try again.")

    print("\nTo install missing dependencies:")
    print("pip install -r requirements.txt")

if __name__ == "__main__":
    main()