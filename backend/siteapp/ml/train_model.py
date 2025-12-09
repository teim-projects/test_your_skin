#!/usr/bin/env python3
"""
Training script for the skin disease prediction model
This script trains a CNN + DenseNet121 model on the skin disease dataset
"""

import os
import sys
import argparse
import logging
from disease_prediction_service import DiseasePredictionService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Train skin disease prediction model')
    parser.add_argument('--dataset', '-d', default='Dataset', 
                       help='Path to dataset folder (default: Dataset)')
    parser.add_argument('--output', '-o', default='enhanced_skin_disease_model.h5',
                       help='Output model path (default: enhanced_skin_disease_model.h5)')
    parser.add_argument('--epochs', '-e', type=int, default=15,
                       help='Number of training epochs (default: 15)')
    parser.add_argument('--batch-size', '-b', type=int, default=32,
                       help='Batch size for training (default: 32)')
    parser.add_argument('--img-size', '-s', type=int, default=224,
                       help='Image size for training (default: 224)')
    
    args = parser.parse_args()
    
    # Check if dataset exists
    if not os.path.exists(args.dataset):
        logger.error(f"Dataset folder not found: {args.dataset}")
        logger.error("Please ensure the Dataset folder exists with disease subfolders")
        return 1
    
    # Check if we have any disease folders
    disease_folders = [d for d in os.listdir(args.dataset) 
                      if os.path.isdir(os.path.join(args.dataset, d))]
    
    if not disease_folders:
        logger.error(f"No disease folders found in {args.dataset}")
        logger.error("Please ensure the dataset has subfolders for each disease type")
        return 1
    
    logger.info(f"Found {len(disease_folders)} disease classes: {disease_folders}")
    
    try:
        # Initialize the prediction service
        logger.info("Initializing prediction service...")
        service = DiseasePredictionService(dataset_path=args.dataset)
        
        # Update training parameters
        service.epochs = args.epochs
        service.batch_size = args.batch_size
        service.img_height = args.img_size
        service.img_width = args.img_size
        
        logger.info(f"Training parameters:")
        logger.info(f"  - Epochs: {service.epochs}")
        logger.info(f"  - Batch size: {service.batch_size}")
        logger.info(f"  - Image size: {service.img_height}x{service.img_width}")
        logger.info(f"  - Dataset: {args.dataset}")
        logger.info(f"  - Output: {args.output}")
        
        # Train the model
        logger.info("Starting model training...")
        history = service.train_model(args.output)
        
        # Log training results
        if history:
            final_accuracy = history.history.get('val_accuracy', [0])[-1]
            final_loss = history.history.get('val_loss', [float('inf')])[-1]
            logger.info(f"Training completed!")
            logger.info(f"  - Final validation accuracy: {final_accuracy:.4f}")
            logger.info(f"  - Final validation loss: {final_loss:.4f}")
            logger.info(f"  - Model saved to: {args.output}")
        
        # Test the trained model
        logger.info("Testing trained model...")
        test_predictions = service.predict(os.path.join(args.dataset, disease_folders[0], 
                                                       os.listdir(os.path.join(args.dataset, disease_folders[0]))[0]))
        
        if test_predictions:
            logger.info("Model test successful!")
            logger.info(f"Sample prediction: {test_predictions[0]['disease']} "
                       f"(confidence: {test_predictions[0]['confidence']:.3f})")
        else:
            logger.warning("Model test failed - no predictions returned")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.exception("Full error traceback:")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
