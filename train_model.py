# =============================================================================
# IMPORT LIBRARIES
# =============================================================================
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from PIL import Image
import pickle
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================
dataset_dir = 'dataset'
img_size = 128
batch_size = 32
num_epochs = 50

# =============================================================================
# LOAD AND PREPROCESS DATA
# =============================================================================
X = []
y = []
label_names = []
label_map = {}

for label in sorted(os.listdir(dataset_dir)):
    label_path = os.path.join(dataset_dir, label)
    if not os.path.isdir(label_path):
        continue

    image_files = os.listdir(label_path)
    if len(image_files) < 4:
        print(f"Skipping class {label} (not enough images)")
        continue

    for image_file in image_files:
        try:
            img_path = os.path.join(label_path, image_file)
            image = Image.open(img_path).convert('RGB')
            image = image.resize((img_size, img_size))
            image_array = np.array(image) / 255.0

            if label not in label_map:
                label_map[label] = len(label_map)
                label_names.append(label)

            X.append(image_array)
            y.append(label_map[label])
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")

X = np.array(X)
y = to_categorical(np.array(y), num_classes=len(label_names))

# =============================================================================
# SPLIT DATASET
# =============================================================================
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# =============================================================================
# COMPUTE CLASS WEIGHTS
# =============================================================================
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(np.argmax(y_train, axis=1)),
    y=np.argmax(y_train, axis=1)
)
class_weights_dict = dict(enumerate(class_weights))

# =============================================================================
# DATA AUGMENTATION
# =============================================================================
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator()

# =============================================================================
# SIMPLE CNN MODEL
# =============================================================================
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_names), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# =============================================================================
# CALLBACKS
# =============================================================================
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

# =============================================================================
# TRAIN MODEL
# =============================================================================
history = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=batch_size),
    epochs=num_epochs,
    validation_data=val_datagen.flow(X_val, y_val),
    class_weight=class_weights_dict,
    callbacks=[early_stop, lr_scheduler]
)

# =============================================================================
# SAVE MODEL AND LABELS
# =============================================================================
model.save("currency_model.h5")
print("✅ Model saved as currency_model.h5")

with open("class_names.pkl", "wb") as f:
    pickle.dump(label_names, f)
print("✅ Class names saved to class_names.pkl")

# =============================================================================
# PLOT TRAINING HISTORY
# =============================================================================
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()