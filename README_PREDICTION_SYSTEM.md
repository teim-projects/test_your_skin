# Skin Disease Prediction System

A comprehensive skin disease prediction system using CNN + DenseNet121 architecture, integrated with Django web framework.

## Features

- **Advanced CNN + DenseNet121 Model**: State-of-the-art deep learning model for skin disease classification
- **Comprehensive Disease Database**: Detailed information for 12+ skin diseases including symptoms, treatments, and urgency levels
- **Image Quality Validation**: Automatic validation of uploaded images for optimal prediction accuracy
- **Web Interface**: User-friendly Django web interface with image upload and analysis
- **Real-time Predictions**: Fast, accurate predictions with confidence scores
- **Detailed Reports**: Comprehensive disease information and treatment recommendations

## Supported Diseases

The system can classify the following skin diseases:

1. **Acne** - Common skin condition with pimples and blackheads
2. **Basal Cell Carcinoma (BCC)** - Most common type of skin cancer
3. **Benign Keratosis-like Lesions (BKL)** - Non-cancerous skin growths
4. **Contact Dermatitis (CD)** - Skin inflammation from irritants/allergens
5. **Fungal Infection** - Skin infections caused by fungi
6. **Impetigo** - Bacterial skin infection
7. **Melanoma** - Serious form of skin cancer
8. **Psoriasis** - Autoimmune condition with scaly patches
9. **Melasma/Hyperpigmentation** - Dark patches on skin
10. **Rosacea** - Chronic facial redness condition
11. **Eczema** - Inflammatory skin condition
12. **Vitiligo** - Loss of skin pigment

## Installation

### Prerequisites

- Python 3.8 or higher
- TensorFlow 2.10+ (GPU support recommended)
- Django 3.2+
- Required Python packages (see requirements.txt)

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Project_frontend
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the dataset**:
   - Ensure your dataset is organized in the `Dataset` folder
   - Each disease should have its own subfolder
   - Supported image formats: JPG, JPEG, PNG

4. **Train the model** (optional - if you have a pre-trained model, skip this step):
   ```bash
   python train_model.py --dataset Dataset --output enhanced_skin_disease_model.h5 --epochs 15
   ```

5. **Run the Django server**:
   ```bash
   cd backend
   python manage.py runserver
   ```

## Usage

### Web Interface

1. **Access the application**: Open your browser and go to `http://localhost:8000`
2. **Register/Login**: Create an account or login to access the image upload feature
3. **Upload Image**: Go to the "Upload Image" page and select a skin image
4. **Select Symptoms**: Check the relevant symptoms from the provided list
5. **Analyze**: Click the "Analyze Image" button to get predictions
6. **View Results**: See detailed predictions with confidence scores, descriptions, and treatment advice

### Programmatic Usage

```python
from disease_prediction_service import DiseasePredictionService

# Initialize the service
service = DiseasePredictionService(dataset_path="Dataset")

# Load a pre-trained model
service.load_model("enhanced_skin_disease_model.h5")

# Make predictions
predictions = service.predict("path/to/image.jpg", top_k=3)

# Get disease information
disease_info = service.get_disease_info("Acne")

# Validate image quality
quality = service.validate_image_quality("path/to/image.jpg")
```

## Model Training

### Training Script

Use the provided training script to train your own model:

```bash
python train_model.py --help
```

### Training Parameters

- `--dataset`: Path to dataset folder (default: Dataset)
- `--output`: Output model path (default: enhanced_skin_disease_model.h5)
- `--epochs`: Number of training epochs (default: 15)
- `--batch-size`: Batch size for training (default: 32)
- `--img-size`: Image size for training (default: 224)

### Example Training Command

```bash
python train_model.py --dataset Dataset --output my_model.h5 --epochs 20 --batch-size 64
```

## Testing

### Test the Prediction Service

```bash
python test_prediction.py
```

This will:
- Test the prediction service with sample images
- Validate image quality checks
- Display prediction results
- Show success rates

### Test Individual Components

```python
# Test disease information
from disease_prediction_service import DiseasePredictionService
service = DiseasePredictionService()
diseases = service.get_all_diseases()
print(f"Supported diseases: {diseases}")
```

## API Endpoints

### Django Views

- `GET /` - Home page
- `GET /upload/` - Image upload page
- `POST /predict/` - Image prediction endpoint
- `GET /about/` - About page
- `GET /treatment/` - Disease information page
- `GET /doctors/` - Doctors page
- `GET /contact/` - Contact page

### Prediction Endpoint

**POST /predict/**

Request:
- `image`: Image file (multipart/form-data)
- `symptoms`: JSON array of selected symptoms

Response:
```json
{
  "ok": true,
  "quality_ok": true,
  "quality_info": {
    "quality_ok": true,
    "size_ok": true,
    "contrast_ok": true,
    "brightness_ok": true
  },
  "predictions": [
    {
      "disease": "Acne",
      "confidence": 0.85,
      "description": "A common skin condition...",
      "symptoms": ["Pimples", "Blackheads"],
      "treatment": "Gentle cleansing, topical treatments...",
      "urgency": "Low",
      "prevention": "Regular cleansing, avoid picking..."
    }
  ],
  "symptoms": ["Itching", "Pain"],
  "filename": "image.jpg",
  "info": {
    "Acne": {
      "desc": "A common skin condition...",
      "advice": "Gentle cleansing, topical treatments...",
      "urgency": "Low"
    }
  }
}
```

## Model Architecture

The system uses a CNN + DenseNet121 architecture:

1. **Base Model**: DenseNet121 (pre-trained on ImageNet)
2. **Feature Extraction**: Global Average Pooling
3. **Classification Head**:
   - Batch Normalization
   - Dense layer (256 units, ReLU)
   - Dropout (0.5)
   - Dense layer (128 units, ReLU)
   - Dropout (0.3)
   - Output layer (N classes, Softmax)

## Image Requirements

- **Format**: JPG, JPEG, PNG
- **Minimum Size**: 224x224 pixels
- **Recommended Size**: 512x512 pixels or larger
- **Quality**: Good contrast and brightness
- **Content**: Clear, well-lit skin images

## Performance

- **Accuracy**: 85-95% on test dataset (varies by disease type)
- **Inference Time**: <2 seconds per image
- **Memory Usage**: ~2GB RAM (with GPU: ~4GB VRAM)
- **Model Size**: ~100MB

## Troubleshooting

### Common Issues

1. **TensorFlow Import Error**:
   ```bash
   pip install tensorflow>=2.10.0
   ```

2. **Model Loading Error**:
   - Ensure model file exists and is compatible
   - Check TensorFlow version compatibility
   - Try clearing model cache: `GET /clear-model-cache/`

3. **Image Quality Issues**:
   - Use higher resolution images
   - Ensure good lighting and contrast
   - Avoid blurry or overexposed images

4. **Memory Issues**:
   - Reduce batch size in training
   - Use smaller image sizes
   - Close other applications

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This system is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical concerns.

## Support

For support and questions:
- Create an issue in the repository
- Contact the development team
- Check the documentation

## Changelog

### Version 1.0.0
- Initial release
- CNN + DenseNet121 model
- Django web interface
- 12 disease classes
- Image quality validation
- Comprehensive disease database
