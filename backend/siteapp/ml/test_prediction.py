#!/usr/bin/env python3
"""
Test script for the skin disease prediction service
This script tests the prediction service with sample images
"""

import os
import sys
import logging
from disease_prediction_service import DiseasePredictionService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_prediction_service():
    """Test the prediction service with sample images"""
    
    # Initialize the service
    logger.info("Initializing prediction service...")
    service = DiseasePredictionService(dataset_path="Dataset")
    
    # Check if we have any models to load
    model_paths = [
        "my_model.h5",
        "skin_disease_cnn_densenet121_balanced.h5",
        "best_model_densenet121.h5",
        "enhanced_skin_disease_model.h5"
    ]
    
    model_loaded = False
    for model_path in model_paths:
        if os.path.exists(model_path):
            logger.info(f"Found model: {model_path}")
            if service.load_model(model_path):
                model_loaded = True
                logger.info(f"Successfully loaded model from {model_path}")
                break
    
    if not model_loaded:
        logger.warning("No compatible model found. Service will use fallback mode.")
    
    # Test with sample images from the dataset
    logger.info("Testing prediction service...")
    
    # Find sample images from each disease class
    sample_images = []
    for disease in service.class_names[:5]:  # Test first 5 diseases
        disease_path = os.path.join(service.dataset_path, disease)
        if os.path.exists(disease_path):
            for img_file in os.listdir(disease_path)[:1]:  # Take first image
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    sample_images.append({
                        'path': os.path.join(disease_path, img_file),
                        'disease': disease
                    })
                    break
    
    if not sample_images:
        logger.error("No sample images found in dataset")
        return False
    
    logger.info(f"Found {len(sample_images)} sample images to test")
    
    # Test each sample image
    success_count = 0
    for i, sample in enumerate(sample_images, 1):
        logger.info(f"\n--- Test {i}: {sample['disease']} ---")
        logger.info(f"Image: {sample['path']}")
        
        try:
            # Validate image quality
            quality = service.validate_image_quality(sample['path'])
            logger.info(f"Quality check: {quality['message']}")
            logger.info(f"  - Size OK: {quality.get('size_ok', 'N/A')}")
            logger.info(f"  - Contrast OK: {quality.get('contrast_ok', 'N/A')}")
            logger.info(f"  - Brightness OK: {quality.get('brightness_ok', 'N/A')}")
            
            # Make prediction
            predictions = service.predict(sample['path'], top_k=3)
            
            if predictions:
                logger.info("Predictions:")
                for j, pred in enumerate(predictions, 1):
                    logger.info(f"  {j}. {pred['disease']} (confidence: {pred['confidence']:.3f})")
                    logger.info(f"     Description: {pred['description'][:100]}...")
                    logger.info(f"     Urgency: {pred['urgency']}")
                
                # Check if the correct disease is in top 3 predictions
                predicted_diseases = [p['disease'] for p in predictions]
                if sample['disease'] in predicted_diseases:
                    logger.info("✓ Correct disease found in predictions!")
                    success_count += 1
                else:
                    logger.warning("✗ Correct disease not in top 3 predictions")
            else:
                logger.warning("No predictions returned")
                
        except Exception as e:
            logger.error(f"Error testing {sample['path']}: {e}")
    
    # Summary
    logger.info(f"\n--- Test Summary ---")
    logger.info(f"Total tests: {len(sample_images)}")
    logger.info(f"Successful predictions: {success_count}")
    logger.info(f"Success rate: {success_count/len(sample_images)*100:.1f}%")
    
    return success_count > 0

def test_disease_info():
    """Test the disease information database"""
    logger.info("\n--- Testing Disease Information ---")
    
    service = DiseasePredictionService()
    diseases = service.get_all_diseases()
    
    logger.info(f"Total diseases in database: {len(diseases)}")
    
    # Test a few diseases
    test_diseases = ['Acne', 'Melanoma 15.75k', 'Eczema']
    for disease in test_diseases:
        if disease in diseases:
            info = service.get_disease_info(disease)
            logger.info(f"\n{disease}:")
            logger.info(f"  Description: {info.get('description', 'N/A')[:100]}...")
            logger.info(f"  Urgency: {info.get('urgency', 'N/A')}")
            logger.info(f"  Symptoms: {', '.join(info.get('symptoms', [])[:3])}...")
        else:
            logger.warning(f"Disease not found: {disease}")

def main():
    """Main test function"""
    logger.info("Starting skin disease prediction service tests...")
    
    # Test disease information
    test_disease_info()
    
    # Test prediction service
    if test_prediction_service():
        logger.info("\n✓ All tests completed successfully!")
        return 0
    else:
        logger.error("\n✗ Some tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
