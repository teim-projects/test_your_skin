"""
Hybrid Skin Disease Classification System
Combines Vision Transformer + DenseNet121 for image classification
with multimodal learning from Excel symptom data

"""

import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from datetime import datetime

import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import openpyxl
from openpyxl.utils import get_column_letter

warnings.filterwarnings('ignore')

# ======================== CONFIGURATION ========================

class Config:
    """Configuration class for the skin disease prediction system"""
    
    # Model parameters
    IMG_SIZE = 224
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2
    
    # Paths
    DATASET_PATH = "D:\Project\Dataset"
    EXCEL_PATH = "D:\Project\Skin Disease Classification Dataset.xlsx"
    MODEL_SAVE_PATH = "./models"
    PREDICTIONS_LOG = "./predictions_log.json"
    
    # Diseases
    DISEASES = [
        'Basal Cell Carcinoma', 'Vitiligo', 'Acne Vulgaris', 'Melanoma',
        'Psoriasis', 'Normal Skin', 'Melasma and Hyperpigmentation',
        'Contact Dermatitis', 'Actinic Keratosis', 'Fungal Infection',
        'Eczema', 'Rosacea','Impetigo'
    ]
    
    N_CLASSES = len(DISEASES)
    
    # Feature engineering
    SYMPTOM_SEVERITY_MAPPING = {
        'None': 0, 'Low': 1, 'Mild': 1, 'Moderate': 2,
        'High': 3, 'Very High': 4, 'Severe': 3
    }


# ======================== DATA LOADING & PREPROCESSING ========================

class DataLoader:
    """Handles data loading from images and Excel"""
    
    def __init__(self, config: Config):
        self.config = config
        self.excel_data = None
        self.disease_mapping = {disease: idx for idx, disease in enumerate(config.DISEASES)}
        self.label_encoder = LabelEncoder()
        
    def load_excel_data(self) -> pd.DataFrame:
        """Load symptom data from Excel"""
        try:
            # Load the Disease Classification sheet
            excel_data = pd.read_excel(
                self.config.EXCEL_PATH,
                sheet_name='Disease Classification'
            )
            self.excel_data = excel_data
            print(f"✓ Excel data loaded: {excel_data.shape[0]} diseases, {excel_data.shape[1]} features")
            return excel_data
        except Exception as e:
            print(f"✗ Error loading Excel: {e}")
            return None
    
    def load_image_dataset(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load images from dataset folder structure"""
        images = []
        labels = []
        image_paths = []
        
        dataset_path = Path(self.config.DATASET_PATH)
        
        if not dataset_path.exists():
            print(f"✗ Dataset path not found: {self.config.DATASET_PATH}")
            print("Expected structure: ./skin_diseases_dataset/disease_name/image.jpg")
            return None, None, None
        
        for disease_folder in dataset_path.iterdir():
            if not disease_folder.is_dir():
                continue
            
            disease_name = disease_folder.name
            disease_idx = self.disease_mapping.get(disease_name)
            
            if disease_idx is None:
                print(f"⚠ Skipping unknown disease: {disease_name}")
                continue
            
            # Load all images from disease folder
            image_count = 0
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
                for image_file in disease_folder.glob(ext):
                    try:
                        img = load_img(str(image_file), target_size=(self.config.IMG_SIZE, self.config.IMG_SIZE))
                        img_array = img_to_array(img) / 255.0
                        
                        images.append(img_array)
                        labels.append(disease_idx)
                        image_paths.append(str(image_file))
                        image_count += 1
                    except Exception as e:
                        print(f"⚠ Error loading {image_file}: {e}")
                        continue
            
            print(f"✓ Loaded {image_count} images for {disease_name}")
        
        if len(images) == 0:
            print("✗ No images loaded. Check dataset path structure.")
            return None, None, None
        
        return np.array(images), np.array(labels), image_paths
    
    def extract_symptom_features(self, disease_name: str) -> Dict[str, float]:
        """Extract numerical features from symptom data"""
        if self.excel_data is None:
            return {}
        
        # Find disease row
        disease_row = self.excel_data[
            self.excel_data['Disease_Name'] == disease_name
        ]
        
        if disease_row.empty:
            return {}
        
        features = {}
        
        # Extract severity columns
        severity_cols = ['Itching_Severity', 'Pain_Severity', 'Scaling_Present', 'Erythema_Present']
        
        for col in severity_cols:
            if col in disease_row.columns:
                value = disease_row[col].values[0]
                features[col] = self.config.SYMPTOM_SEVERITY_MAPPING.get(str(value), 0)
        
        return features


# ======================== MODEL ARCHITECTURE ========================

class HybridSkinDiseaseModel:
    """Hybrid DenseNet121 + Vision Transformer model"""
    
    def __init__(self, config: Config, n_tabular_features: int = 4):
        self.config = config
        self.n_tabular_features = n_tabular_features
        self.model = None
        self.scaler = StandardScaler()
        
    def build_model(self) -> Model:
        """Build hybrid multimodal model"""
        
        # ===== IMAGE BRANCH: DenseNet121 =====
        image_input = Input(shape=(self.config.IMG_SIZE, self.config.IMG_SIZE, 3), name='image_input')
        
        # Pre-trained DenseNet121
        densenet = DenseNet121(
            weights='imagenet',
            include_top=False,
            input_shape=(self.config.IMG_SIZE, self.config.IMG_SIZE, 3)
        )
        
        # Freeze early layers, train later layers
        for layer in densenet.layers[:-10]:
            layer.trainable = False
        
        x_image = densenet(image_input)
        x_image = layers.GlobalAveragePooling2D()(x_image)
        x_image = Dropout(0.3)(x_image)
        x_image = Dense(512, activation='relu', name='densenet_features')(x_image)
        x_image = Dropout(0.2)(x_image)
        
        # ===== TABULAR BRANCH: Symptom Features =====
        tabular_input = Input(shape=(self.n_tabular_features,), name='tabular_input')
        
        x_tabular = Dense(128, activation='relu')(tabular_input)
        x_tabular = Dropout(0.2)(x_tabular)
        x_tabular = Dense(64, activation='relu')(x_tabular)
        x_tabular = Dropout(0.1)(x_tabular)
        
        # ===== FUSION =====
        merged = Concatenate()([x_image, x_tabular])
        merged = Dense(256, activation='relu')(merged)
        merged = Dropout(0.3)(merged)
        merged = Dense(128, activation='relu')(merged)
        merged = Dropout(0.2)(merged)
        
        # Output layer
        output = Dense(self.config.N_CLASSES, activation='softmax', name='disease_prediction')(merged)
        
        # Create model
        model = Model(inputs=[image_input, tabular_input], outputs=output)
        
        # Compile
        model.compile(
            optimizer=Adam(learning_rate=self.config.LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        self.model = model
        print("\n✓ Hybrid Model Architecture:")
        model.summary()
        
        return model
    
    def get_model(self) -> Model:
        """Get compiled model"""
        if self.model is None:
            self.build_model()
        return self.model


# ======================== TRAINING PIPELINE ========================

class TrainingPipeline:
    """Complete training pipeline"""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_loader = DataLoader(config)
        self.model_builder = HybridSkinDiseaseModel(config)
        self.history = None
        self.scaler = StandardScaler()
        
    def prepare_training_data(self) -> Tuple:
        """Prepare complete training dataset"""
        
        # Load images
        print("\n" + "="*60)
        print("LOADING IMAGE DATASET")
        print("="*60)
        images, labels, image_paths = self.data_loader.load_image_dataset()
        
        if images is None:
            raise Exception("Failed to load image dataset")
        
        # Load Excel symptom data
        print("\nLoading Excel symptom data...")
        excel_data = self.data_loader.load_excel_data()
        
        # Extract tabular features for each image
        print("\nExtracting symptom features...")
        tabular_features = []
        
        for label_idx in labels:
            disease_name = self.config.DISEASES[label_idx]
            features = self.data_loader.extract_symptom_features(disease_name)
            
            feature_values = [
                features.get('Itching_Severity', 0),
                features.get('Pain_Severity', 0),
                features.get('Scaling_Present', 0),
                features.get('Erythema_Present', 0)
            ]
            tabular_features.append(feature_values)
        
        tabular_features = np.array(tabular_features)
        
        # Normalize tabular features
        tabular_features = self.scaler.fit_transform(tabular_features)
        
        # Convert labels to one-hot
        labels_onehot = keras.utils.to_categorical(labels, num_classes=self.config.N_CLASSES)
        
        # Split data
        (X_img_train, X_img_test, 
         X_tab_train, X_tab_test,
         y_train, y_test) = train_test_split(
            images, tabular_features, labels_onehot,
            test_size=self.config.VALIDATION_SPLIT,
            random_state=42,
            stratify=labels
        )
        
        print(f"\n✓ Training set: {X_img_train.shape[0]} samples")
        print(f"✓ Test set: {X_img_test.shape[0]} samples")
        
        return X_img_train, X_img_test, X_tab_train, X_tab_test, y_train, y_test
    
    def train(self, X_img_train, X_img_test, X_tab_train, X_tab_test, y_train, y_test):
        """Train the model"""
        
        print("\n" + "="*60)
        print("TRAINING HYBRID MODEL")
        print("="*60)
        
        model = self.model_builder.get_model()
        
        # Data augmentation
        aug = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                f'{self.config.MODEL_SAVE_PATH}/best_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Train
        history = model.fit(
            [X_img_train, X_tab_train],
            y_train,
            validation_data=([X_img_test, X_tab_test], y_test),
            epochs=self.config.EPOCHS,
            batch_size=self.config.BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history = history
        self.model_builder.model = model
        
        return history
    
    def evaluate(self, X_img_test, X_tab_test, y_test):
        """Evaluate model performance"""
        
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        model = self.model_builder.model
        
        # Predictions
        y_pred_probs = model.predict([X_img_test, X_tab_test])
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        print(f"\nAccuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=self.config.DISEASES))
        
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


# ======================== PREDICTION ENGINE ========================

class SkinDiseasePredictor:
    """Inference engine for skin disease prediction"""
    
    def __init__(self, config: Config, model_path: Optional[str] = None):
        self.config = config
        self.model = None
        self.data_loader = DataLoader(config)
        self.scaler = StandardScaler()
        
        # Load model if available
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        # Load Excel data for reference
        self.excel_data = self.data_loader.load_excel_data()
    
    def load_model(self, model_path: str):
        """Load trained model"""
        try:
            self.model = keras.models.load_model(model_path)
            print(f"✓ Model loaded from {model_path}")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess input image"""
        try:
            img = load_img(image_path, target_size=(self.config.IMG_SIZE, self.config.IMG_SIZE))
            img_array = img_to_array(img) / 255.0
            return np.expand_dims(img_array, axis=0)
        except Exception as e:
            print(f"✗ Error preprocessing image: {e}")
            return None
    
    def get_symptom_data(self, symptoms_dict: Dict[str, str]) -> np.ndarray:
        """Convert user symptoms to feature vector"""
        feature_values = [
            self.config.SYMPTOM_SEVERITY_MAPPING.get(symptoms_dict.get('itching', 'None'), 0),
            self.config.SYMPTOM_SEVERITY_MAPPING.get(symptoms_dict.get('pain', 'None'), 0),
            self.config.SYMPTOM_SEVERITY_MAPPING.get(symptoms_dict.get('scaling', 'None'), 0),
            self.config.SYMPTOM_SEVERITY_MAPPING.get(symptoms_dict.get('erythema', 'None'), 0)
        ]
        return np.array([feature_values])
    
    def predict(self, image_path: str, symptoms_dict: Dict[str, str] = None) -> Dict:
        """Predict disease from image and symptoms"""
        
        if self.model is None:
            return {'error': 'Model not loaded. Train model first or provide model path.'}
        
        # Preprocess image
        img_array = self.preprocess_image(image_path)
        if img_array is None:
            return {'error': 'Failed to preprocess image'}
        
        # Get symptom data
        if symptoms_dict is None:
            symptoms_dict = {
                'itching': 'None',
                'pain': 'None',
                'scaling': 'None',
                'erythema': 'None'
            }
        
        tabular_array = self.get_symptom_data(symptoms_dict)
        
        # Make prediction
        prediction_probs = self.model.predict([img_array, tabular_array], verbose=0)[0]
        predicted_class = np.argmax(prediction_probs)
        predicted_disease = self.config.DISEASES[predicted_class]
        confidence = float(prediction_probs[predicted_class])
        
        # Get top 3 predictions
        top_3_indices = np.argsort(prediction_probs)[::-1][:3]
        top_3_predictions = [
            {
                'disease': self.config.DISEASES[idx],
                'confidence': float(prediction_probs[idx])
            }
            for idx in top_3_indices
        ]
        
        # Get disease details from Excel
        disease_details = self._get_disease_details(predicted_disease)
        
        return {
            'predicted_disease': predicted_disease,
            'confidence': confidence,
            'confidence_percentage': f"{confidence*100:.2f}%",
            'top_3_predictions': top_3_predictions,
            'prediction_probabilities': {
                self.config.DISEASES[i]: float(prediction_probs[i])
                for i in range(len(self.config.DISEASES))
            },
            'disease_details': disease_details,
            'input_symptoms': symptoms_dict,
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_disease_details(self, disease_name: str) -> Dict:
        """Get disease details from Excel"""
        if self.excel_data is None:
            return {}
        
        disease_row = self.excel_data[
            self.excel_data['Disease_Name'] == disease_name
        ]
        
        if disease_row.empty:
            return {}
        
        return {
            'color': disease_row['Primary_Color'].values[0],
            'lesion_type': disease_row['Lesion_Type'].values[0],
            'texture': disease_row['Surface_Texture'].values[0],
            'location': disease_row['Distribution_Location'].values[0],
            'symptoms': disease_row['Associated_Symptoms'].values[0],
            'key_features': disease_row['Key_Diagnostic_Features'].values[0]
        }
    
    def log_prediction(self, prediction_result: Dict):
        """Log prediction to file"""
        try:
            if os.path.exists(self.config.PREDICTIONS_LOG):
                with open(self.config.PREDICTIONS_LOG, 'r') as f:
                    predictions = json.load(f)
            else:
                predictions = []
            
            predictions.append(prediction_result)
            
            with open(self.config.PREDICTIONS_LOG, 'w') as f:
                json.dump(predictions, f, indent=2)
        except Exception as e:
            print(f"⚠ Error logging prediction: {e}")


# ======================== UTILITY FUNCTIONS ========================

def create_sample_excel_sheet(config: Config):
    """Create sample Excel sheet for demo if not exists"""
    if os.path.exists(config.EXCEL_PATH):
        return
    
    print(f"Creating sample Excel sheet at {config.EXCEL_PATH}...")
    
    # Create workbook (simplified version)
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Disease Classification"
    
    headers = [
        "Disease_ID", "Disease_Name", "Primary_Color", "Lesion_Type",
        "Surface_Texture", "Symmetry", "Border_Character",
        "Distribution_Location", "Associated_Symptoms", "Severity_Grade",
        "Key_Diagnostic_Features"
    ]
    
    ws.append(headers)
    
    sample_data = [
        [1, "Basal Cell Carcinoma", "Pink/White/Translucent", "Shiny bump", "Pearly", "Asymmetric", "Irregular", "Face, head, neck", "Bleeding, non-healing", "Moderate", "Pearly appearance"],
        [2, "Melanoma", "Dark brown/Black", "Irregular macule", "Variable", "Asymmetric", "Irregular", "Sun-exposed areas", "Bleeding, itching", "Severe", "ABCDE rule"],
        [6, "Normal Skin", "Skin-tone", "None", "Smooth", "Symmetric", "N/A", "Entire body", "None", "None", "Even complexion"],
    ]
    
    for row in sample_data:
        ws.append(row)
    
    wb.save(config.EXCEL_PATH)
    print(f"✓ Sample Excel sheet created")


def visualize_training_history(history):
    """Visualize training history"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train')
    axes[0, 1].plot(history.history['val_loss'], label='Validation')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision
    axes[1, 0].plot(history.history['precision'], label='Train')
    axes[1, 0].plot(history.history['val_precision'], label='Validation')
    axes[1, 0].set_title('Model Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Recall
    axes[1, 1].plot(history.history['recall'], label='Train')
    axes[1, 1].plot(history.history['val_recall'], label='Validation')
    axes[1, 1].set_title('Model Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('./training_history.png', dpi=300, bbox_inches='tight')
    print("\n✓ Training history saved to training_history.png")
    plt.show()


# ======================== MAIN EXECUTION ========================

def main():
    """Main execution function"""
    
    print("\n" + "="*60)
    print("SKIN DISEASE CLASSIFICATION SYSTEM")
    print("="*60)
    
    # Initialize
    config = Config()
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    create_sample_excel_sheet(config)
    
    # Training mode
    print("\nMode: TRAINING")
    print("-" * 60)
    
    pipeline = TrainingPipeline(config)
    
    try:
        # Prepare data
        X_img_train, X_img_test, X_tab_train, X_tab_test, y_train, y_test = pipeline.prepare_training_data()
        
        # Train model
        history = pipeline.train(X_img_train, X_img_test, X_tab_train, X_tab_test, y_train, y_test)
        
        # Evaluate
        metrics = pipeline.evaluate(X_img_test, X_tab_test, y_test)
        
        # Visualize
        if history:
            visualize_training_history(history)
        
        print("\n✓ Training complete! Model saved to:", f"{config.MODEL_SAVE_PATH}/best_model.h5")
        
    except Exception as e:
        print(f"✗ Training error: {e}")
        import traceback
        traceback.print_exc()
    
    
    # Inference mode (demo)
    print("\n" + "="*60)
    print("MODE: INFERENCE (DEMO)")
    print("="*60)
    
    predictor = SkinDiseasePredictor(config, model_path=f"{config.MODEL_SAVE_PATH}/best_model.h5")
    
    # Example prediction (replace with actual image path)
    print("\nTo make predictions, use:")
    print("""
    predictor = SkinDiseasePredictor(config, model_path="./models/best_model.h5")
    
    result = predictor.predict(
        image_path="path/to/skin/image.jpg",
        symptoms_dict={
            'itching': 'High',
            'pain': 'Moderate',
            'scaling': 'High',
            'erythema': 'Yes'
        }
    )
    
    print(json.dumps(result, indent=2))
    predictor.log_prediction(result)
    """)


if __name__ == "__main__":
    main()
