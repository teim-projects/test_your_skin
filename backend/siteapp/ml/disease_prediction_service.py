"""
Enhanced Skin Disease Prediction Service using CNN + DenseNet121
This module provides a comprehensive service for skin disease classification
with improved preprocessing, model management, and prediction capabilities.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union
from PIL import Image, ImageStat
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')
import math

# TensorFlow imports with error handling
try:
    import tensorflow as tf
    from tensorflow.keras.applications import DenseNet121
    from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras import layers, models, callbacks
    from tensorflow.keras.optimizers import Adam
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.metrics import classification_report, confusion_matrix
    TF_AVAILABLE = True
except ImportError as e:
    print(f"TensorFlow not available: {e}")
    TF_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DiseasePredictionService:
    """
    Enhanced service for skin disease prediction using CNN + DenseNet121
    """
    
    def __init__(self, dataset_path: str = "Dataset", model_path: str = None):
        """
        Initialize the disease prediction service
        
        Args:
            dataset_path: Path to the dataset folder
            model_path: Path to saved model (optional)
        """
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.model = None
        self.class_names = []
        self.class_to_index = {}
        self.index_to_class = {}
        self.use_tta = True  # enable test-time augmentation for stabler predictions
        
        # Model parameters
        self.img_height = 224
        self.img_width = 224
        self.batch_size = 32
        self.epochs = 15
        
        # Disease information database
        self.disease_info = self._load_disease_info()
        
        if not TF_AVAILABLE:
            logger.error("TensorFlow is not available. Please install TensorFlow to use this service.")
            return
            
        # Load or create model
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self._prepare_class_names()
            # After preparing class names, generate a dataset counts file for transparency
            try:
                self.write_dataset_counts(os.path.join(self.dataset_path, '..', 'dataset_counts.txt'))
            except Exception as e:
                logger.warning(f"Failed to write dataset counts: {e}")
    
    def _load_disease_info(self) -> Dict[str, Dict[str, str]]:
        """Load comprehensive disease information"""
        return {
            'Acne': {
                'description': 'A common skin condition that occurs when hair follicles become plugged with oil and dead skin cells.',
                'symptoms': ['Pimples', 'Blackheads', 'Whiteheads', 'Cysts', 'Nodules'],
                'causes': ['Excess oil production', 'Clogged hair follicles', 'Bacteria', 'Hormonal changes'],
                'treatment': 'Gentle cleansing, topical treatments, oral medications, professional treatments',
                'urgency': 'Low',
                'prevention': 'Regular cleansing, avoid picking, use non-comedogenic products'
            },
            'Basal Cell Carcinoma (BCC) 3323': {
                'description': 'The most common type of skin cancer, usually appearing as a pearly or waxy bump.',
                'symptoms': ['Pearly bump', 'Waxy appearance', 'Bleeding', 'Slow growth', 'Raised edges'],
                'causes': ['UV radiation exposure', 'Fair skin', 'Age', 'Family history'],
                'treatment': 'Surgical removal, Mohs surgery, radiation therapy, topical treatments',
                'urgency': 'High',
                'prevention': 'Sun protection, regular skin checks, avoid tanning beds'
            },
            'Benign Keratosis-like Lesions (BKL) 2624': {
                'description': 'Non-cancerous skin growths that appear as brown or black patches, often mistaken for moles.',
                'symptoms': ['Brown patches', 'Black patches', 'Rough texture', 'Flat appearance'],
                'causes': ['Sun exposure', 'Age', 'Genetic factors'],
                'treatment': 'Monitoring, cryotherapy, laser treatment, surgical removal if needed',
                'urgency': 'Low',
                'prevention': 'Sun protection, regular monitoring for changes'
            },
            'CD': {
                'description': 'Contact dermatitis is skin inflammation caused by direct contact with irritants or allergens.',
                'symptoms': ['Redness', 'Itching', 'Burning', 'Swelling', 'Blisters'],
                'causes': ['Allergens', 'Irritants', 'Chemicals', 'Plants', 'Metals'],
                'treatment': 'Avoid triggers, topical corticosteroids, antihistamines, moisturizers',
                'urgency': 'Low',
                'prevention': 'Identify and avoid triggers, use protective clothing'
            },
            'Fungal Infection': {
                'description': 'Skin infections caused by fungi, often appearing as red, itchy, circular patches.',
                'symptoms': ['Red patches', 'Itching', 'Scaling', 'Ring-shaped lesions', 'Burning'],
                'causes': ['Fungal organisms', 'Warm, moist environments', 'Weakened immune system'],
                'treatment': 'Antifungal creams, Oral antifungals, Keep area dry, Good hygiene',
                'urgency': 'Medium',
                'prevention': 'Keep skin dry, avoid sharing personal items, wear breathable clothing'
            },
            'Impetigo': {
                'description': 'A highly contagious bacterial skin infection that causes red sores and blisters.',
                'symptoms': ['Red sores', 'Blisters', 'Honey-colored crusts', 'Itching', 'Pain'],
                'causes': ['Staphylococcus aureus', 'Streptococcus pyogenes', 'Skin breaks'],
                'treatment': 'Antibiotic creams, Oral antibiotics, Keep clean and dry',
                'urgency': 'Medium',
                'prevention': 'Good hygiene, avoid scratching, don\'t share personal items'
            },
            'Melanoma 15.75k': {
                'description': 'The most serious type of skin cancer that can spread to other parts of the body.',
                'symptoms': ['Asymmetric moles', 'Irregular borders', 'Color variation', 'Large diameter', 'Evolving'],
                'causes': ['UV radiation', 'Genetic factors', 'Mole characteristics', 'Family history'],
                'treatment': 'Surgical removal, Immunotherapy, Targeted therapy, Chemotherapy',
                'urgency': 'Critical',
                'prevention': 'Sun protection, regular skin checks, avoid tanning beds'
            },
            'Psoriasis pictures Lichen Planus and related diseases - 2k': {
                'description': 'An autoimmune condition causing red, scaly patches on the skin.',
                'symptoms': ['Red patches', 'Silvery scales', 'Itching', 'Burning', 'Thickened skin'],
                'causes': ['Autoimmune disorder', 'Genetic factors', 'Triggers (stress, infections)'],
                'treatment': 'Topical treatments, Phototherapy, Systemic medications, Biologics',
                'urgency': 'Medium',
                'prevention': 'Manage triggers, Moisturize regularly, Avoid injury to skin'
            },
            'melasma And hyperpigmentation': {
                'description': 'Dark patches on the skin caused by increased melanin production.',
                'symptoms': ['Dark patches', 'Irregular shapes', 'Facial distribution', 'Sun exposure worsens'],
                'causes': ['Hormonal changes', 'Sun exposure', 'Pregnancy', 'Birth control pills'],
                'treatment': 'Topical lightening agents, Chemical peels, Laser therapy, Sun protection',
                'urgency': 'Low',
                'prevention': 'Sun protection, Hormone management, Gentle skincare'
            },
            'rosacea': {
                'description': 'A chronic skin condition causing facial redness and visible blood vessels.',
                'symptoms': ['Facial redness', 'Visible blood vessels', 'Bumps and pimples', 'Eye irritation'],
                'causes': ['Genetic factors', 'Environmental triggers', 'Immune system response'],
                'treatment': 'Topical medications, Oral antibiotics, Laser therapy, Lifestyle changes',
                'urgency': 'Low',
                'prevention': 'Identify triggers, Gentle skincare, Sun protection'
            },
            'Eczema': {
                'description': 'An inflammatory skin condition causing dry, itchy patches.',
                'symptoms': ['Dry skin', 'Itching', 'Redness', 'Scaling', 'Cracking'],
                'causes': ['Genetic factors', 'Environmental triggers', 'Immune system dysfunction'],
                'treatment': 'Moisturizers, Topical corticosteroids, Antihistamines, Avoid triggers',
                'urgency': 'Low',
                'prevention': 'Moisturize regularly, Identify triggers, Gentle skincare'
            },
            'vitiligo': {
                'description': 'A condition causing loss of skin pigment in patches.',
                'symptoms': ['White patches', 'Loss of pigment', 'Premature graying', 'Eye color changes'],
                'causes': ['Autoimmune disorder', 'Genetic factors', 'Environmental triggers'],
                'treatment': 'Topical treatments, Phototherapy, Surgical options, Cosmetic coverings',
                'urgency': 'Low',
                'prevention': 'Sun protection, Stress management, Regular monitoring'
            }
        }
    
    def _prepare_class_names(self):
        """Prepare class names from dataset folder structure"""
        if not os.path.exists(self.dataset_path):
            logger.warning(f"Dataset path {self.dataset_path} does not exist")
            return
        
        try:
            self.class_names = sorted([d for d in os.listdir(self.dataset_path) 
                                     if os.path.isdir(os.path.join(self.dataset_path, d))])
            self.class_to_index = {name: idx for idx, name in enumerate(self.class_names)}
            self.index_to_class = {idx: name for idx, name in enumerate(self.class_names)}
            logger.info(f"Found {len(self.class_names)} disease classes: {self.class_names}")
            
            # Filter disease_info to only include diseases that are actually in the dataset
            available_diseases = set(self.class_names)
            filtered_disease_info = {}
            for disease, info in self.disease_info.items():
                if disease in available_diseases:
                    filtered_disease_info[disease] = info
                else:
                    # Try to find a close match
                    for dataset_disease in available_diseases:
                        if disease.lower() in dataset_disease.lower() or dataset_disease.lower() in disease.lower():
                            filtered_disease_info[dataset_disease] = info
                            break
            
            # Update disease_info with filtered version
            self.disease_info = filtered_disease_info
            logger.info(f"Filtered disease info to {len(self.disease_info)} diseases matching dataset")
            
        except Exception as e:
            logger.error(f"Error preparing class names: {e}")
    
    def load_model(self, model_path: str):
        """Load a pre-trained model"""
        if not TF_AVAILABLE:
            logger.error("TensorFlow not available")
            return False
        
        try:
            # Try different loading methods for compatibility
            try:
                self.model = tf.keras.models.load_model(model_path, compile=False)
            except Exception as e1:
                logger.warning(f"Standard loading failed: {e1}")
                # Try with custom objects
                custom_objects = {
                    'DenseNet121': DenseNet121,
                    'GlobalAveragePooling2D': layers.GlobalAveragePooling2D,
                    'BatchNormalization': layers.BatchNormalization,
                    'Dense': layers.Dense,
                    'Dropout': layers.Dropout,
                }
                try:
                    self.model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
                except Exception as e2:
                    logger.warning(f"Custom objects loading failed: {e2}")
                    # Try with safe_mode=False for newer TensorFlow versions
                    try:
                        self.model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
                    except Exception as e3:
                        logger.warning(f"Safe mode loading failed: {e3}")
                        # Try legacy loading
                        try:
                            from tensorflow.keras.saving import legacy as keras_legacy
                            self.model = keras_legacy.load_model(model_path, compile=False)
                        except Exception as e4:
                            logger.warning(f"Legacy loading failed: {e4}")
                            # Try to create a compatible model architecture
                            try:
                                logger.info("Attempting to create compatible model architecture...")
                                self.model = self._create_compatible_model()
                                # Try to load weights only
                                weights_path = model_path.replace('.h5', '_weights.h5')
                                if os.path.exists(weights_path):
                                    self.model.load_weights(weights_path)
                                    logger.info("Loaded weights successfully")
                                else:
                                    logger.warning("No weights file found, using random weights")
                            except Exception as e5:
                                logger.error(f"All loading methods failed: {e5}")
                                raise e5
            
            # Load class names if available
            labels_path = model_path.replace('.h5', '_labels.txt').replace('.keras', '_labels.txt')
            if os.path.exists(labels_path):
                with open(labels_path, 'r') as f:
                    self.class_names = [line.strip() for line in f if line.strip()]
                self.class_to_index = {name: idx for idx, name in enumerate(self.class_names)}
                self.index_to_class = {idx: name for idx, name in enumerate(self.class_names)}
            
            logger.info(f"Model loaded successfully from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def create_model(self) -> tf.keras.Model:
        """Create a new DenseNet121-based model"""
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow not available")
        
        # Ensure we have the correct number of classes from the actual dataset
        num_classes = len(self.class_names)
        if num_classes == 0:
            raise RuntimeError("No class names found. Please ensure dataset is properly loaded.")
        
        # Base DenseNet121 model
        base_model = DenseNet121(
            input_shape=(self.img_height, self.img_width, 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False
        
        # Build the complete model
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _create_compatible_model(self) -> tf.keras.Model:
        """Create a compatible model architecture for loading"""
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow not available")
        
        # Create a simple compatible model
        model = models.Sequential([
            layers.Input(shape=(self.img_height, self.img_width, 3)),
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        return model
    
    def prepare_data(self) -> Tuple[ImageDataGenerator, ImageDataGenerator, Dict]:
        """Prepare data generators for training"""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset path {self.dataset_path} not found")
        
        # Collect image paths and labels
        image_paths = []
        labels = []
        
        for class_name in self.class_names:
            class_path = os.path.join(self.dataset_path, class_name)
            if os.path.isdir(class_path):
                for img_file in os.listdir(class_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_paths.append(os.path.join(class_path, img_file))
                        labels.append(class_name)
        
        # Create DataFrame
        df = pd.DataFrame({'image_path': image_paths, 'label': labels})
        df = df[df['image_path'].map(os.path.exists)]
        
        # Split data
        train_df, val_df = train_test_split(
            df, test_size=0.2, stratify=df['label'], random_state=42
        )
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            preprocessing_function=densenet_preprocess,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        
        val_datagen = ImageDataGenerator(preprocessing_function=densenet_preprocess)
        
        # Create generators
        train_generator = train_datagen.flow_from_dataframe(
            train_df,
            x_col='image_path',
            y_col='label',
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical'
        )
        
        val_generator = val_datagen.flow_from_dataframe(
            val_df,
            x_col='image_path',
            y_col='label',
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical'
        )
        
        # Compute class weights
        unique_labels = np.unique(train_df['label'])
        class_weights = compute_class_weight('balanced', classes=unique_labels, y=train_df['label'])
        label_to_weight = dict(zip(unique_labels, class_weights))
        class_weight_dict = {train_generator.class_indices[label]: float(weight) 
                           for label, weight in label_to_weight.items()}
        
        return train_generator, val_generator, class_weight_dict

    def _count_images_per_class(self) -> Dict[str, int]:
        """Count number of images for each class in the dataset folder."""
        counts: Dict[str, int] = {}
        if not os.path.exists(self.dataset_path):
            return counts
        for class_name in self.class_names:
            class_path = os.path.join(self.dataset_path, class_name)
            if os.path.isdir(class_path):
                num_images = 0
                try:
                    for filename in os.listdir(class_path):
                        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                            num_images += 1
                except Exception:
                    pass
                counts[class_name] = num_images
        return counts

    def write_labels_with_counts(self, labels_path: str) -> None:
        """Write labels with image counts to a text file.

        Format per line: <ClassName>\t<Count>
        """
        counts = self._count_images_per_class()
        with open(labels_path, 'w', encoding='utf-8') as f:
            for name in self.class_names:
                count = counts.get(name, 0)
                f.write(f"{name}\t{count}\n")

    def write_dataset_counts(self, counts_path: str) -> None:
        """Write per-class counts to a human-friendly file.

        If a labels file exists next to a trained model, this complements it.
        """
        # Normalize path (may include ..)
        counts_path = os.path.abspath(counts_path)
        os.makedirs(os.path.dirname(counts_path), exist_ok=True)
        counts = self._count_images_per_class()
        total = sum(counts.values())
        with open(counts_path, 'w', encoding='utf-8') as f:
            f.write("# Dataset image counts per class\n")
            for name in self.class_names:
                f.write(f"{name}: {counts.get(name, 0)}\n")
            f.write(f"\nTotal images: {total}\n")
    
    def train_model(self, save_path: str = "skin_disease_model.h5"):
        """Train the model on the dataset"""
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow not available")
        
        logger.info("Preparing data...")
        train_gen, val_gen, class_weights = self.prepare_data()
        
        # Update class names from the data generator to ensure consistency
        self.class_names = list(train_gen.class_indices.keys())
        self.class_to_index = train_gen.class_indices
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        
        logger.info(f"Updated class names from data generator: {self.class_names}")
        logger.info(f"Number of classes: {len(self.class_names)}")
        
        logger.info("Creating model...")
        self.model = self.create_model()
        
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                save_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        logger.info("Training model...")
        history = self.model.fit(
            train_gen,
            steps_per_epoch=len(train_gen),
            epochs=self.epochs,
            validation_data=val_gen,
            validation_steps=len(val_gen),
            callbacks=callbacks_list,
            class_weight=class_weights,
            verbose=1
        )
        
        # Save labels and counts
        labels_path = save_path.replace('.h5', '_labels.txt')
        with open(labels_path, 'w', encoding='utf-8') as f:
            for name in self.class_names:
                f.write(f"{name}\n")
        # Also write labels with counts and a dataset counts summary
        try:
            labels_with_counts = save_path.replace('.h5', '_labels_with_counts.txt')
            self.write_labels_with_counts(labels_with_counts)
            self.write_dataset_counts(os.path.join(os.path.dirname(os.path.abspath(save_path)), 'dataset_counts.txt'))
        except Exception as e:
            logger.warning(f"Could not write counts files: {e}")
        
        logger.info(f"Model training completed. Saved to {save_path}")
        return history
    
    def _load_pil_image(self, image: Union[Image.Image, str, bytes]) -> Image.Image:
        """Normalize different input types into a RGB PIL image"""
        if isinstance(image, Image.Image):
            img = image.copy()
        elif isinstance(image, str):
            img = Image.open(image)
        elif isinstance(image, bytes):
            img = Image.open(BytesIO(image))
        else:
            raise ValueError("Unsupported image type for prediction")
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img

    def _prepare_array(self, image: Image.Image) -> np.ndarray:
        """Resize and preprocess a single PIL image, returning CHW array"""
        resized = image.resize((self.img_height, self.img_width))
        arr = np.array(resized, dtype=np.float32)
        return arr

    def _generate_tta_batch(self, image: Image.Image) -> np.ndarray:
        """Generate a batch of augmented images for TTA"""
        variants = [
            image,
            image.transpose(Image.FLIP_LEFT_RIGHT),
            image.transpose(Image.FLIP_TOP_BOTTOM),
            image.rotate(90, expand=True),
            image.rotate(270, expand=True),
        ]
        processed = [self._prepare_array(variant) for variant in variants]
        batch = np.stack(processed, axis=0)
        if densenet_preprocess is not None:
            batch = densenet_preprocess(batch)
        return batch

    def preprocess_image(self, image: Union[Image.Image, str, bytes]) -> np.ndarray:
        """Preprocess image for prediction"""
        img = self._load_pil_image(image)
        arr = self._prepare_array(img)
        batch = np.expand_dims(arr, axis=0)
        if densenet_preprocess is not None:
            batch = densenet_preprocess(batch)
        return batch
    
    def predict(self, image: Union[Image.Image, str, bytes], top_k: int = 3) -> List[Dict]:
        """Predict disease from image"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Please load a model first.")
        
        try:
            # Preprocess image (with optional test-time augmentation)
            pil_image = self._load_pil_image(image)
            if self.use_tta:
                processed_batch = self._generate_tta_batch(pil_image)
            else:
                processed_batch = self.preprocess_image(pil_image)
            
            # Make prediction
            predictions = self.model.predict(processed_batch, verbose=0)
            
            # Get top-k predictions
            if isinstance(predictions, (list, tuple)):
                predictions = predictions[0]
            elif predictions.ndim == 2 and self.use_tta:
                predictions = predictions.mean(axis=0)
            elif predictions.ndim == 2:
                predictions = predictions[0]
            
            # Convert to probabilities
            probabilities = tf.nn.softmax(predictions, axis=-1).numpy().reshape(-1)
            
            # Get top-k indices
            top_indices = np.argsort(probabilities)[::-1][:top_k]
            
            # Format results
            results = []
            for idx in top_indices:
                if idx < len(self.class_names):
                    disease_name = self.class_names[idx]
                    confidence = float(probabilities[idx])
                    
                    # Get disease information
                    disease_info = self.disease_info.get(disease_name, {})
                    
                    results.append({
                        'disease': disease_name,
                        'confidence': confidence,
                        'description': disease_info.get('description', 'No description available'),
                        'symptoms': disease_info.get('symptoms', []),
                        'treatment': disease_info.get('treatment', 'Consult a dermatologist'),
                        'urgency': disease_info.get('urgency', 'Unknown'),
                        'prevention': disease_info.get('prevention', 'Consult a dermatologist')
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    def validate_image_quality(self, image: Union[Image.Image, str, bytes]) -> Dict[str, Union[bool, str]]:
        """Validate image quality for prediction"""
        try:
            # Load image if needed
            if isinstance(image, str):
                image = Image.open(image)
            elif isinstance(image, bytes):
                image = Image.open(BytesIO(image))
            
            # Basic quality checks
            width, height = image.size
            min_size = min(width, height)
            
            # Check minimum size
            size_ok = min_size >= 224
            
            # Check contrast (blur detection)
            gray_image = image.convert('L')
            stat = ImageStat.Stat(gray_image)
            contrast = stat.stddev[0]
            contrast_ok = contrast >= 10.0
            
            # Check brightness
            brightness = stat.mean[0]
            brightness_ok = 20 <= brightness <= 235
            
            # Simple skin detection using YCbCr thresholds
            # Reference heuristic ranges often used for skin: 77<=Cb<=127 and 133<=Cr<=173
            # We'll sample a resized thumbnail to keep it fast
            thumb = image.convert('YCbCr').resize((64, 64))
            ycbcr = np.array(thumb)
            cb = ycbcr[:, :, 1]
            cr = ycbcr[:, :, 2]
            skin_mask = (cb >= 77) & (cb <= 127) & (cr >= 133) & (cr <= 173)
            skin_ratio = float(np.mean(skin_mask)) if skin_mask.size > 0 else 0.0
            is_skin_like = skin_ratio >= 0.15  # at least 15% pixels look like skin

            # Overall quality
            quality_ok = size_ok and contrast_ok and brightness_ok and is_skin_like

            tips: List[str] = []
            if not size_ok:
                tips.append('Use images at least 224x224 pixels.')
            if not contrast_ok:
                tips.append('Ensure the image is in focus with sufficient contrast.')
            if not brightness_ok:
                tips.append('Avoid over/underexposed images; use good lighting.')
            if not is_skin_like:
                tips.append('Upload a clear, focused image of the skin area to be analyzed.')
            
            return {
                'quality_ok': quality_ok,
                'size_ok': size_ok,
                'contrast_ok': contrast_ok,
                'brightness_ok': brightness_ok,
                'is_skin_like': is_skin_like,
                'skin_ratio': skin_ratio,
                'min_size': min_size,
                'contrast': float(contrast),
                'brightness': float(brightness),
                'message': 'Image quality is good' if quality_ok else 'The uploaded image is not recognized as a clear skin image or may be of poor quality.',
                'tips': tips,
            }
            
        except Exception as e:
            logger.error(f"Image quality validation error: {e}")
            return {
                'quality_ok': False,
                'message': f'Error validating image: {str(e)}'
            }
    
    def get_disease_info(self, disease_name: str) -> Dict:
        """Get detailed information about a specific disease"""
        return self.disease_info.get(disease_name, {})
    
    def get_all_diseases(self) -> List[str]:
        """Get list of all supported diseases"""
        return list(self.disease_info.keys())
    
    def save_model(self, path: str):
        """Save the current model"""
        if self.model is None:
            raise RuntimeError("No model to save")
        
        self.model.save(path)
        
        # Save labels
        labels_path = path.replace('.h5', '_labels.txt').replace('.keras', '_labels.txt')
        with open(labels_path, 'w') as f:
            for name in self.class_names:
                f.write(f"{name}\n")
        
        logger.info(f"Model saved to {path}")


# Example usage and testing
if __name__ == "__main__":
    # Initialize the service
    service = DiseasePredictionService(dataset_path="Dataset")
    
    # Check if we have a pre-trained model
    model_paths = [
        "my_model.h5",
        "skin_disease_cnn_densenet121_balanced.h5",
        "best_model_densenet121.h5"
    ]
    
    model_loaded = False
    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f"Loading existing model: {model_path}")
            if service.load_model(model_path):
                model_loaded = True
                break
    
    if not model_loaded:
        print("No pre-trained model found. Training new model...")
        try:
            service.train_model("enhanced_skin_disease_model.h5")
        except Exception as e:
            print(f"Training failed: {e}")
    
    # Test prediction with a sample image
    print("\nTesting prediction service...")
    print(f"Supported diseases: {service.get_all_diseases()}")
    
    # Example: Test with a sample image from the dataset
    sample_images = []
    for disease in service.class_names[:3]:  # Test first 3 diseases
        disease_path = os.path.join(service.dataset_path, disease)
        if os.path.exists(disease_path):
            for img_file in os.listdir(disease_path)[:1]:  # Take first image
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    sample_images.append(os.path.join(disease_path, img_file))
                    break
    
    for img_path in sample_images:
        print(f"\nTesting with: {img_path}")
        try:
            # Validate image quality
            quality = service.validate_image_quality(img_path)
            print(f"Quality check: {quality}")
            
            # Make prediction
            predictions = service.predict(img_path, top_k=3)
            print("Predictions:")
            for i, pred in enumerate(predictions, 1):
                print(f"  {i}. {pred['disease']} (confidence: {pred['confidence']:.3f})")
                print(f"     Description: {pred['description'][:100]}...")
                print(f"     Urgency: {pred['urgency']}")
        except Exception as e:
            print(f"Error testing {img_path}: {e}")
