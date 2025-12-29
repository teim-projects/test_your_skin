import numpy as np
import pandas as pd
import os
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')  # ensure tf.keras legacy saver/loader
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# PARAMETERS
IMGHEIGHT = 224
IMGWIDTH = 224
BATCHSIZE = 32
EPOCHS = 10

# Dataset and Labels
images_folder = r'D:\Project_frontend\Dataset'
label_list = sorted(os.listdir(images_folder))  # Assuming each folder is a disease name

image_paths = []
labels = []
for label in label_list:
    folder = os.path.join(images_folder, label)
    for img_file in glob(os.path.join(folder, '*.jpg')):
        image_paths.append(img_file)
        labels.append(label)

df = pd.DataFrame({'image_path': image_paths, 'label': labels})

# Remove samples with missing files
df = df[df['image_path'].map(os.path.exists)]

# Stratified split
traindf, valdf = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

# Data generators with optimized augmentation to prevent overfitting
traindatagen = ImageDataGenerator(
    preprocessing_function=densenet_preprocess,
    rotation_range=10,  # Reduced from 15
    width_shift_range=0.05,  # Reduced from 0.1
    height_shift_range=0.05,  # Reduced from 0.1
    zoom_range=0.05,  # Reduced from 0.1
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],  # Added brightness variation
    fill_mode='nearest'
)
valdatagen = ImageDataGenerator(preprocessing_function=densenet_preprocess)

traingenerator = traindatagen.flow_from_dataframe(
    traindf,
    x_col='image_path',
    y_col='label',
    target_size=(IMGHEIGHT, IMGWIDTH),
    batch_size=BATCHSIZE,
    class_mode='categorical'
)

valgenerator = valdatagen.flow_from_dataframe(
    valdf,
    x_col='image_path',
    y_col='label',
    target_size=(IMGHEIGHT, IMGWIDTH),
    batch_size=BATCHSIZE,
    class_mode='categorical'
)

# Compute class weights mapped to class indices (required by Keras)
unique_labels = np.unique(traindf['label'])
class_weights = compute_class_weight('balanced', classes=unique_labels, y=traindf['label'])
label_to_weight = dict(zip(unique_labels, class_weights))
label_to_index = traingenerator.class_indices  # {label: index}
class_weight_dict = {label_to_index[label]: float(label_to_weight[label]) for label in unique_labels}

# Build DenseNet121 Model
base_model = DenseNet121(
    input_shape=(IMGHEIGHT, IMGWIDTH, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(traingenerator.class_indices), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Enhanced callbacks to prevent overfitting and noise learning
earlystop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=4,  # Increased patience
    restore_best_weights=True,
    verbose=1
)

# Reduce learning rate when validation loss plateaus
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-7,
    verbose=1
)

# Save the best model during training
model_checkpoint = callbacks.ModelCheckpoint(
    'best_model_densenet121.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# Train the model
history = model.fit(
    traingenerator,
    steps_per_epoch=len(traingenerator),
    epochs=EPOCHS,
    validation_data=valgenerator,
    validation_steps=len(valgenerator),
    callbacks=[earlystop, reduce_lr, model_checkpoint],
    class_weight=class_weight_dict,
    verbose=1
)

# Evaluate
val_loss, val_acc = model.evaluate(valgenerator)
print(f'Validation Accuracy: {val_acc*100:.2f}')

# Save labels file in class index order
index_to_label = {idx: label for label, idx in traingenerator.class_indices.items()}
labels_sorted = [index_to_label[i] for i in range(len(index_to_label))]
with open('labels.txt', 'w', encoding='utf-8') as f:
    for name in labels_sorted:
        f.write(f"{name}\n")

# Save the model in multiple formats for maximum compatibility
# Fix layer names to avoid forward slash issues
for layer in model.layers:
    if hasattr(layer, 'name') and '/' in layer.name:
        layer._name = layer.name.replace('/', '_')

# Save models with fixed names
model.save('skin_disease_cnn_densenet121_balanced.h5', include_optimizer=False)
model.save('skin_disease_cnn_densenet121_balanced.keras')  # Native Keras format doesn't support include_optimizer parameter
tf.saved_model.save(model, 'skin_disease_cnn_densenet121_balanced_savedmodel')

# Plotting
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()




