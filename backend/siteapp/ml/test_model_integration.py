#!/usr/bin/env python3
"""
Test script to verify the model integration with Django
"""

import os
import sys
import django
from django.conf import settings

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'backend'))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'project.settings')
django.setup()

from siteapp.views import _get_prediction_service
from PIL import Image
import numpy as np

def test_model_loading():
    """Test if the model loads correctly"""
    print("Testing model loading...")
    
    try:
        service = _get_prediction_service()
        if service and service.model is not None:
            print("SUCCESS: Model loaded successfully!")
            print(f"   - Model type: {type(service.model)}")
            print(f"   - Number of classes: {len(service.class_names)}")
            print(f"   - Classes: {service.class_names}")
            return True
        else:
            print("ERROR: Model failed to load")
            return False
    except Exception as e:
        print(f"ERROR: Error loading model: {e}")
        return False

def test_prediction():
    """Test prediction with a dummy image"""
    print("\nTesting prediction...")
    
    try:
        service = _get_prediction_service()
        if not service or service.model is None:
            print("ERROR: No model available for testing")
            return False
        
        # Create a dummy image for testing
        dummy_image = Image.new('RGB', (224, 224), color='red')
        
        # Test prediction
        predictions = service.predict(dummy_image, top_k=3)
        
        if predictions:
            print("SUCCESS: Prediction successful!")
            print("   Top predictions:")
            for i, pred in enumerate(predictions, 1):
                print(f"   {i}. {pred['disease']} (confidence: {pred['confidence']:.3f})")
            return True
        else:
            print("ERROR: No predictions returned")
            return False
            
    except Exception as e:
        print(f"ERROR: Error during prediction: {e}")
        return False

def test_disease_info():
    """Test disease information retrieval"""
    print("\nTesting disease information...")
    
    try:
        service = _get_prediction_service()
        if not service:
            print("ERROR: No service available")
            return False
        
        diseases = service.get_all_diseases()
        print(f"SUCCESS: Found {len(diseases)} diseases in database")
        
        # Test a specific disease
        acne_info = service.get_disease_info('Acne')
        if acne_info:
            print("SUCCESS: Disease information retrieval working")
            print(f"   Acne description: {acne_info.get('description', 'N/A')[:50]}...")
            return True
        else:
            print("ERROR: Could not retrieve disease information")
            return False
            
    except Exception as e:
        print(f"ERROR: Error retrieving disease info: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing Model Integration with Django")
    print("=" * 50)
    
    tests = [
        test_model_loading,
        test_prediction,
        test_disease_info
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("SUCCESS: All tests passed! The system is ready to use.")
        return True
    else:
        print("WARNING: Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
