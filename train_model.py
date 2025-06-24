# ======================= IMPORT LIBRARIES =======================
import os
import random
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    BatchNormalization, GlobalAveragePooling2D
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from PIL import Image
import pickle
import cv2
from collections import Counter

# ======================= CONFIGURATION =======================
# source: Pandelu, A. P. (2024, December 15). Day 48: Training Neural Networks — Hyperparameters, Batch Size, Epochs | by Adithya Prasad Pandelu | Medium. Medium. https://medium.com/@bhatadithya54764118/day-48-training-neural-networks-hyperparameters-batch-size-epochs-712c57d9e30c
DATASET_DIR = 'dataset'
IMAGE_SIZE = (128, 128)  # Keep consistent with interface
BATCH_SIZE = 16  # Reduced for better gradient updates
EPOCHS = 50  # Increased epochs
RANDOM_SEED = 42
MIN_IMAGES_PER_CLASS = 8  # Increased minimum for better training
LEARNING_RATE = 0.0005  # Reduced learning rate for better convergence
VALIDATION_SPLIT = 0.25  # 25% for validation

# Set random seeds for reproducibility
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

print("Starting improved currency classifier training...")


# ======================= ENHANCED DATA CLEANUP =======================
# source: Geeks for Geeks. (2025, April 28). Check If A File is Valid Image with Python - GeeksforGeeks. Geeks for Geeks. https://www.geeksforgeeks.org/python/check-if-a-file-is-valid-image-with-python/
def clean_dataset():
    cleaned_count = 0
    total_files = 0

    for root, dirs, files in os.walk(DATASET_DIR):
        # Skip problematic directories
        if any(skip_dir in root for skip_dir in ['_MACOSX', '__pycache__', 'test', 'train', 'validation', 'dataset']):
            continue

        for file in files:
            file_path = os.path.join(root, file)
            total_files += 1

            # Remove system files
            if file.startswith('.') or file.startswith('__') or file == 'Thumbs.db':
                try:
                    os.remove(file_path)
                    cleaned_count += 1
                    print(f"Removed system file: {file}")
                except Exception as e:
                    print(f"Could not remove {file}: {e}")
                continue

            # Validate image files
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                try:
                    with Image.open(file_path) as img:
                        img.verify()
                    # Re-open for size check (verify closes the file)
                    with Image.open(file_path) as img:
                        if img.size[0] < 32 or img.size[1] < 32:  # Too small
                            print(f"Removing too small image: {file_path}")
                            os.remove(file_path)
                            cleaned_count += 1
                except Exception as e:
                    print(f"Removing corrupted image: {file_path} - {e}")
                    try:
                        os.remove(file_path)
                        cleaned_count += 1
                    except:
                        pass

    print(f"Processed {total_files} files, cleaned {cleaned_count} invalid files")


clean_dataset()


# ======================= IMPROVED CLASS COLLECTION =======================
# source: Hule, V. (2022, January 19). Python Count Number of Files in a Directory [4 Ways] – PYnative. PYnative. https://pynative.com/python-count-number-of-files-in-a-directory/
def collect_valid_classes():
    class_stats = {}

    for folder in os.listdir(DATASET_DIR):
        folder_path = os.path.join(DATASET_DIR, folder)

        # Skip non-directories and system folders
        if not os.path.isdir(folder_path):
            continue
        if folder.startswith('.') or folder in ['_MACOSX', 'dataset', 'test', 'train', 'validation', '__pycache__']:
            continue

        # Count valid images
        image_files = []
        for file in os.listdir(folder_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                file_path = os.path.join(folder_path, file)
                # Quick size check
                try:
                    with Image.open(file_path) as img:
                        if img.size[0] >= 32 and img.size[1] >= 32:
                            image_files.append(file)
                except:
                    continue

        if len(image_files) >= MIN_IMAGES_PER_CLASS:
            class_stats[folder] = len(image_files)
            print(f"✓ Class '{folder}': {len(image_files)} images")
        else:
            print(f"✗ Class '{folder}': {len(image_files)} images (below minimum {MIN_IMAGES_PER_CLASS})")

    return class_stats


class_stats = collect_valid_classes()
valid_classes = list(class_stats.keys())

if len(valid_classes) == 0:
    raise Exception(f"No classes have enough images (minimum required: {MIN_IMAGES_PER_CLASS}).")

print(f"\nFound {len(valid_classes)} valid classes")
print(f"Total images: {sum(class_stats.values())}")
print("Top classes by image count:")
for cls, count in sorted(class_stats.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {cls}: {count} images")


# ======================= ENHANCED IMAGE PREPROCESSING =======================
# source: Koul Nimrita. (2023, December 20). Image Processing using OpenCV — Python | by Dr. Nimrita Koul | Medium. Medium. https://medium.com/@nimritakoul01/image-processing-using-opencv-python-9c9b83f4b1ca
def advanced_preprocess_image(image_path, target_size=IMAGE_SIZE):
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return None

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_shape = img.shape

        # Enhanced auto-crop with multiple strategies
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Strategy 1: Otsu thresholding for better edge detection
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Strategy 2: Morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Filter contours by area (remove tiny noise)
            min_area = (original_shape[0] * original_shape[1]) * 0.05  # At least 5% of image
            valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]

            if valid_contours:
                # Get the largest valid contour
                largest_contour = max(valid_contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)

                # Smart padding based on image size
                pad_x = max(10, int(w * 0.05))
                pad_y = max(10, int(h * 0.05))

                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(img.shape[1], x + w + pad_x)
                y2 = min(img.shape[0], y + h + pad_y)

                # Only crop if the result is reasonable
                crop_area = (x2 - x1) * (y2 - y1)
                original_area = original_shape[0] * original_shape[1]

                if (crop_area > original_area * 0.1 and  # At least 10% of original
                        (x2 - x1) > 50 and (y2 - y1) > 50):  # Minimum size
                    img = img[y1:y2, x1:x2]

        # Resize with high-quality resampling
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)

        # Enhanced normalization with contrast adjustment
        img = img.astype(np.float32) / 255.0

        # Histogram equalization in LAB color space for better contrast
        lab = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0

        return img

    except Exception as e:
        print(f"Error preprocessing {image_path}: {e}")
        return None


# ======================= BALANCED DATASET CREATION =======================
# source: Fernandez, H. L. (2025, March 4). Stratified Splitting with train_test_split Using Target and Group Variables — Part 1 | by Hugo López-Fernández | Medium. Medium. https://medium.com/@hlfzeus/stratified-splitting-with-train-test-split-using-target-and-group-variables-part-1-f3dbe5ce84fd
def create_balanced_dataset():
    BASE_DIR = 'processed_dataset'
    TRAIN_DIR = os.path.join(BASE_DIR, 'train')
    VAL_DIR = os.path.join(BASE_DIR, 'val')

    if os.path.exists(BASE_DIR):
        shutil.rmtree(BASE_DIR)

    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)

    total_processed = 0
    class_distribution = {}

    for cls in valid_classes:
        class_dir = os.path.join(DATASET_DIR, cls)
        image_files = []

        # Collect all valid images
        for file in os.listdir(class_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                file_path = os.path.join(class_dir, file)
                try:
                    with Image.open(file_path) as img:
                        if img.size[0] >= 32 and img.size[1] >= 32:
                            image_files.append(file)
                except:
                    continue

        if len(image_files) < MIN_IMAGES_PER_CLASS:
            continue

        # Shuffle for random distribution
        random.shuffle(image_files)

        # Stratified split
        n_val = max(1, int(len(image_files) * VALIDATION_SPLIT))
        n_train = len(image_files) - n_val

        train_files = image_files[:n_train]
        val_files = image_files[n_train:]

        # Create class directories
        os.makedirs(os.path.join(TRAIN_DIR, cls), exist_ok=True)
        os.makedirs(os.path.join(VAL_DIR, cls), exist_ok=True)

        # Process training files
        train_success = 0
        for i, file in enumerate(train_files):
            src_path = os.path.join(class_dir, file)
            processed_img = advanced_preprocess_image(src_path)

            if processed_img is not None:
                # Save with better quality
                dst_path = os.path.join(TRAIN_DIR, cls, f"train_{i:04d}.jpg")
                pil_img = Image.fromarray((processed_img * 255).astype(np.uint8))
                pil_img.save(dst_path, 'JPEG', quality=95, optimize=True)
                train_success += 1

        # Process validation files
        val_success = 0
        for i, file in enumerate(val_files):
            src_path = os.path.join(class_dir, file)
            processed_img = advanced_preprocess_image(src_path)

            if processed_img is not None:
                dst_path = os.path.join(VAL_DIR, cls, f"val_{i:04d}.jpg")
                pil_img = Image.fromarray((processed_img * 255).astype(np.uint8))
                pil_img.save(dst_path, 'JPEG', quality=95, optimize=True)
                val_success += 1

        class_distribution[cls] = {'train': train_success, 'val': val_success}
        total_processed += train_success + val_success

        print(f"Processed '{cls}': {train_success} train, {val_success} val")

    print(f"\nTotal processed images: {total_processed}")
    return BASE_DIR, class_distribution


# Create the enhanced balanced dataset
BASE_DIR, class_distribution = create_balanced_dataset()

# ======================= ADVANCED DATA GENERATORS =======================
# source: Lee, W. M. (2022, October 26). Image Data Augmentation for Deep Learning | Towards Data Science. Towards Data Science. https://towardsdatascience.com/image-data-augmentation-for-deep-learning-77a87fabd2bf/
# More aggressive augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=[0.8, 1.2],
    horizontal_flip=True,
    vertical_flip=False,  # Currency typically shouldn't be flipped vertically
    fill_mode='reflect',
    brightness_range=[0.7, 1.3],
    channel_shift_range=0.2,
    # Add noise and blur for robustness
    preprocessing_function=lambda x: x + np.random.normal(0, 0.01, x.shape)
)

# Simple rescaling for validation
val_datagen = ImageDataGenerator(rescale=1. / 255)

# Create generators with class balancing
train_generator = train_datagen.flow_from_directory(
    os.path.join(BASE_DIR, 'train'),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=RANDOM_SEED
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(BASE_DIR, 'val'),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    seed=RANDOM_SEED
)

# Save class names
class_names = list(train_generator.class_indices.keys())
with open('class_names.pkl', 'wb') as f:
    pickle.dump(class_names, f)

print(f"\nDataset ready:")
print(f"Classes: {len(class_names)}")
print(f"Training batches: {len(train_generator)}")
print(f"Validation batches: {len(val_generator)}")
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")


# ======================= IMPROVED MODEL ARCHITECTURE =======================
# source: Koushik. (2023, November 28). Understanding Convolutional Neural Networks (CNNs) in Depth | by Koushik | Medium. Medium. https://medium.com/@koushikkushal95/understanding-convolutional-neural-networks-cnns-in-depth-d18e299bb438
def create_advanced_model(num_classes, input_shape=(128, 128, 3)):
    model = Sequential([
        # Entry block
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Block 1
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Block 2 - More filters for complex features
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        # Block 3 - Even more filters
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        # Block 4 - High-level features
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Dropout(0.4),

        # Global Average Pooling instead of Flatten (reduces overfitting)
        GlobalAveragePooling2D(),

        # Dense layers with regularization
        Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),

        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),

        # Output layer
        Dense(num_classes, activation='softmax', name='predictions')
    ])

    return model


# Create and compile the advanced model
model = create_advanced_model(len(class_names))

# Use a more sophisticated optimizer with different learning rates
optimizer = Adam(
    learning_rate=LEARNING_RATE,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-7
)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy', 'top_k_categorical_accuracy']
)

print("\nModel Architecture:")
model.summary()

# ======================= ADVANCED CALLBACKS =======================
# source: Kashyap, P. (2024, October 30). Early Stopping in Deep Learning: A Simple Guide to Prevent Overfitting | by Piyush Kashyap | Medium. Medium. https://medium.com/@piyushkashyap045/early-stopping-in-deep-learning-a-simple-guide-to-prevent-overfitting-1073f56b493e
callbacks = [
    # Save best model
    ModelCheckpoint(
        'best_currency_classifier.h5',
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,
        mode='max',
        verbose=1
    ),

    # Early stopping with more patience
    EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1,
        mode='max'
    ),

    # Reduce learning rate
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1,
        cooldown=2
    )
]

# ======================= TRAINING WITH CLASS WEIGHTS =======================
# source: Kamaldeep. (2025, May 1). How to Improve Class Imbalance using Class Weights in ML? Analytics Vidhya. https://www.analyticsvidhya.com/blog/2020/10/improve-class-imbalance-class-weights/
# Calculate class weights for imbalanced data
class_counts = []
for cls in class_names:
    count = class_distribution.get(cls, {}).get('train', 0)
    class_counts.append(count)

# Create class weights
total_samples = sum(class_counts)
class_weights = {}
for i, count in enumerate(class_counts):
    if count > 0:
        class_weights[i] = total_samples / (len(class_names) * count)
    else:
        class_weights[i] = 1.0

print(f"\nClass weights calculated for {len(class_weights)} classes")
print("Sample class weights:", {f"class_{k}": f"{v:.2f}" for k, v in list(class_weights.items())[:5]})

# ======================= TRAIN THE MODEL =======================
# source: Ikegbo, O. S. (2025, May 8). Understanding model.fit() in TensorFlow: A Comprehensive Guide | by Ogochukwu Stanley Ikegbo | May, 2025 | Medium. Medium. https://medium.com/@stacymacbrains/heres-a-medium-style-article-that-explains-the-model-fit-8000b008c5f1
print("\nStarting advanced model training...")

# Calculate steps
steps_per_epoch = max(1, train_generator.samples // BATCH_SIZE)
validation_steps = max(1, val_generator.samples // BATCH_SIZE)

print(f"Steps per epoch: {steps_per_epoch}")
print(f"Validation steps: {validation_steps}")

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=validation_steps,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

# ======================= SAVE EVERYTHING =======================
# source: Bais, G. (2025, May 6). How to Save Trained Model in Python. Neptune Blog. https://neptune.ai/blog/saving-trained-model-in-python
# Load the best model
try:
    best_model = tf.keras.models.load_model('best_currency_classifier.h5')
    # Save as the main model file
    best_model.save("currency_classifier.h5")
    print("Best model saved as currency_classifier.h5")
except:
    # If best model doesn't exist, save current model
    model.save("currency_classifier.h5")
    print("Current model saved as currency_classifier.h5")

# Save comprehensive metadata
metadata = {
    'image_size': IMAGE_SIZE,
    'num_classes': len(class_names),
    'class_names': class_names,
    'class_distribution': class_distribution,
    'class_weights': class_weights,
    'model_architecture': 'advanced_cnn',
    'training_params': {
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'validation_split': VALIDATION_SPLIT,
        'min_images_per_class': MIN_IMAGES_PER_CLASS
    },
    'data_augmentation': {
        'rotation_range': 20,
        'width_shift_range': 0.15,
        'height_shift_range': 0.15,
        'zoom_range': [0.8, 1.2],
        'brightness_range': [0.7, 1.3]
    }
}

with open('model_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

# Save training history
with open('training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)

# ======================= FINAL EVALUATION =======================
print("\n" + "=" * 50)
print("FINAL EVALUATION")
print("=" * 50)

# Evaluate on validation set
val_loss, val_accuracy, val_top_k = model.evaluate(val_generator, verbose=0)

print(f"\nFinal Metrics:")
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f} ({val_accuracy * 100:.2f}%)")
print(f"Validation Top-3 Accuracy: {val_top_k:.4f} ({val_top_k * 100:.2f}%)")

# Training summary
final_train_acc = max(history.history['accuracy'])
final_train_loss = min(history.history['loss'])

print(f"\nTraining Summary:")
print(f"Final training accuracy: {final_train_acc:.4f} ({final_train_acc * 100:.2f}%)")
print(f"Final training loss: {final_train_loss:.4f}")
print(f"Total classes trained: {len(class_names)}")
print(f"Total training images: {train_generator.samples}")
print(f"Total validation images: {val_generator.samples}")

# Check for overfitting
acc_diff = final_train_acc - val_accuracy
if acc_diff > 0.1:
    print(f"\n⚠️ Warning: Possible overfitting detected (accuracy difference: {acc_diff:.3f})")
    print("Consider reducing model complexity or increasing regularization.")
else:
    print(f"\n✅ Good generalization (accuracy difference: {acc_diff:.3f})")

print(f"\n🎉 Training completed successfully!")
print("\nFiles created:")
print("✓ currency_classifier.h5 (main model)")
print("✓ best_currency_classifier.h5 (best checkpoint)")
print("✓ class_names.pkl (class labels)")
print("✓ model_metadata.pkl (comprehensive model info)")
print("✓ training_history.pkl (training metrics)")

print(f"\nModel is ready for use with main.py!")