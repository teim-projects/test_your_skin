# 🏥 CNN DenseNet121 Skin Disease Detection System

A comprehensive web application for detecting skin diseases using deep learning with DenseNet121 architecture, integrated with Django backend and modern frontend interface.

## 🎯 **Features**

- **🧠 Deep Learning Model**: DenseNet121 pre-trained on ImageNet with custom classification head
- **🖼️ Image Analysis**: Upload and analyze skin images for disease detection
- **📊 Top-3 Predictions**: Get confidence scores for multiple disease possibilities
- **🏥 Medical Information**: Disease descriptions and medical advice
- **⚠️ Urgency Detection**: Automatic flagging of serious conditions (Carcinoma, Melanoma)
- **📋 Symptom Integration**: Combine image analysis with symptom data
- **📄 PDF Reports**: Downloadable analysis reports
- **👤 User Management**: Login/register system with user profiles

## 🏗️ **System Architecture**

### **Frontend**
- **HTML/CSS/JavaScript**: Modern responsive interface
- **Bootstrap**: Professional UI framework
- **AJAX**: Seamless prediction without page reload

### **Backend**
- **Django 5.0.6**: Web framework
- **TensorFlow 2.20.0**: Deep learning framework
- **DenseNet121**: Pre-trained CNN model
- **PIL/Pillow**: Image processing

### **Model**
- **Architecture**: DenseNet121 + Custom Classification Head
- **Input**: 224×224×3 RGB images
- **Output**: 10 disease classes
- **Training**: Transfer learning with data augmentation

## 🦠 **Supported Diseases**

1. **Acne** - Common skin condition
2. **Basal Cell Carcinoma (BCC)** - Skin cancer (URGENT)
3. **Benign Keratosis-like Lesions (BKL)** - Non-cancerous growths
4. **Contact Dermatitis (CD)** - Skin inflammation
5. **Fungal Infection** - Fungal skin infections
6. **Impetigo** - Bacterial skin infection
7. **Melanoma** - Serious skin cancer (URGENT)
8. **Psoriasis** - Autoimmune condition
9. **Melasma & Hyperpigmentation** - Dark patches
10. **Rosacea** - Chronic facial redness

## 🚀 **Quick Start**

### **Prerequisites**
- Python 3.8+
- Virtual environment (recommended)
- Required packages (see requirements.txt)

### **Installation**

1. **Clone/Download the project**
2. **Navigate to backend directory**:
   ```bash
   cd backend
   ```

3. **Create virtual environment**:
   ```bash
   python -m venv skinapp
   skinapp\Scripts\activate  # Windows
   # or
   source skinapp/bin/activate  # Linux/Mac
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Run Django server**:
   ```bash
   python manage.py runserver
   ```

6. **Access the application**:
   - Open browser: `http://127.0.0.1:8000/`
   - Upload page: `http://127.0.0.1:8000/upload/`

## 📁 **Project Structure**

```
Project_frontend/
├── backend/                 # Django backend
│   ├── siteapp/            # Main Django app
│   │   ├── views.py        # API endpoints & model integration
│   │   └── urls.py         # URL routing
│   ├── project/            # Django project settings
│   ├── static/             # Static files (CSS, JS, images)
│   └── requirements.txt    # Python dependencies
├── cnn_densenet121.py      # Model training script
├── image upload.html       # Frontend interface
├── labels.txt              # Disease class labels
├── *.h5                    # Trained model files
└── Dataset/                # Training dataset
```

## 🔧 **Technical Details**

### **Model Training**
- **Dataset**: 10 disease classes with 15,000+ images
- **Preprocessing**: DenseNet-specific normalization
- **Augmentation**: Rotation, zoom, brightness variation
- **Training**: 10 epochs with early stopping
- **Validation**: 80/20 train/validation split

### **Model Loading**
- **Multiple Formats**: .h5, .keras, SavedModel
- **Layer Name Fixing**: Automatic handling of compatibility issues
- **Lazy Loading**: Model loaded once and cached
- **Error Handling**: Comprehensive fallback strategies

### **API Endpoints**
- `GET /` - Home page
- `GET /upload/` - Image upload interface
- `POST /predict/` - Image analysis endpoint
- `GET /login/` - User login
- `GET /register/` - User registration

## 🛠️ **Development**

### **Training New Model**
1. Update dataset in `Dataset/` folder
2. Modify `cnn_densenet121.py` parameters
3. Run training script:
   ```bash
   python cnn_densenet121.py
   ```

### **Adding New Diseases**
1. Add disease folder to `Dataset/`
2. Update `labels.txt`
3. Retrain model
4. Update disease info in `views.py`

## 📊 **Performance**

- **Model Accuracy**: High accuracy on validation set
- **Inference Time**: < 2 seconds per image
- **Memory Usage**: Optimized with lazy loading
- **Scalability**: Django backend supports multiple users

## 🔒 **Security**

- **User Authentication**: Django built-in auth system
- **CSRF Protection**: Enabled for all forms
- **Input Validation**: Image size and format checks
- **Error Handling**: Secure error messages

## 📝 **Usage Instructions**

1. **Register/Login** to the system
2. **Navigate** to Upload Image page
3. **Upload** a clear skin image (JPG, PNG)
4. **Select** relevant symptoms from checklist
5. **Click** "Analyze Image" button
6. **Review** predictions with confidence scores
7. **Read** medical advice and recommendations
8. **Download** PDF report if needed

## ⚠️ **Important Notes**

- **Medical Disclaimer**: This system is for educational/research purposes
- **Not a Replacement**: Always consult healthcare professionals
- **Image Quality**: Use clear, well-lit images for best results
- **Urgent Cases**: Serious conditions are flagged for immediate attention

## 🐛 **Troubleshooting**

### **Common Issues**
1. **Model Loading Errors**: Check TensorFlow installation
2. **Import Errors**: Verify virtual environment activation
3. **Image Upload Issues**: Check file format and size
4. **Server Errors**: Check Django logs and dependencies

### **Support**
- Check Django logs: `python manage.py runserver --verbosity=2`
- Verify model files exist in project root
- Ensure all dependencies are installed

## 📄 **License**

This project is for educational and research purposes. Please ensure compliance with medical device regulations if used in clinical settings.

## 🤝 **Contributing**

1. Fork the repository
2. Create feature branch
3. Make changes
4. Test thoroughly
5. Submit pull request

---

**🏥 Built with ❤️ for medical research and education**
