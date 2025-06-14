# =============================================================================
# STEP 1: IMPORT LIBRARIES & DEFINE SETTINGS
# =============================================================================
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Set image size and batch size
IMG_SIZE = (180, 180)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE


# =============================================================================
# STEP 2: FUNCTION TO EXTRACT LABELS FROM FILENAMES
# =============================================================================
def extract_label_from_filename(filename):
    """Extract currency label from filename like 'EUR_5_2013_2.png' -> 'EUR_5'"""
    # Remove file extension
    name = os.path.splitext(filename)[0]
    # Split by underscore and take first two parts (currency and denomination)
    parts = name.split('_')
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return name


# =============================================================================
# STEP 3: FUNCTION TO LOAD DATASET FROM IMAGE FOLDER
# =============================================================================
def load_dataset_from_images(images_dir):
    """Load dataset from a folder of images where labels are in filenames"""

    # Get all image files
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    image_files = []
    labels = []

    for filename in os.listdir(images_dir):
        if filename.lower().endswith(image_extensions):
            image_files.append(os.path.join(images_dir, filename))
            label = extract_label_from_filename(filename)
            labels.append(label)

    print(f"Found {len(image_files)} images")

    # Convert labels to integers
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    print(f"Found {len(label_encoder.classes_)} classes: {list(label_encoder.classes_)}")

    # Image preprocessing function
    def process_image(image_path, label):
        image = tf.io.read_file(image_path)
        # Try to decode as different formats
        try:
            image = tf.image.decode_png(image, channels=3)
        except:
            try:
                image = tf.image.decode_jpeg(image, channels=3)
            except:
                image = tf.image.decode_image(image, channels=3)

        image = tf.image.resize(image, IMG_SIZE)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_files, encoded_labels))
    dataset = dataset.map(process_image, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)

    return dataset, label_encoder


# =============================================================================
# STEP 4: LOAD AND SPLIT DATASETS
# =============================================================================
print("📥 Loading and splitting data...")

# Check which folders exist
train_exists = os.path.exists('dataset_split/train/images')
val_exists = os.path.exists('dataset_split/val/images')
test_exists = os.path.exists('dataset_split/test/images')

if train_exists and val_exists:
    # Load from separate train/val folders
    print("Found separate train and validation folders")
    train_ds, label_encoder = load_dataset_from_images('dataset_split/train/images')
    val_ds, val_label_encoder = load_dataset_from_images('dataset_split/val/images')

elif test_exists:
    # Load from test folder and split
    print("Found test folder - will split into train/validation")
    all_ds, label_encoder = load_dataset_from_images('dataset_split/test/images')

    # Split dataset (80% train, 20% validation)
    total_batches = tf.data.experimental.cardinality(all_ds).numpy()
    train_size = int(0.8 * total_batches)

    train_ds = all_ds.take(train_size)
    val_ds = all_ds.skip(train_size)

    print(f"Split into {train_size} training batches and {total_batches - train_size} validation batches")

else:
    raise ValueError("No dataset folder found! Please check your folder structure.")

# Make sure both datasets have the same classes
num_classes = len(label_encoder.classes_)
print(f"📊 Training with {num_classes} currency classes")

# =============================================================================
# STEP 5: BUILD MODEL
# =============================================================================
print("🧠 Building model...")

model = models.Sequential([
    layers.InputLayer(input_shape=(*IMG_SIZE, 3)),

    # Data augmentation (optional, helps with overfitting)
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),

    # First convolutional block
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),

    # Second convolutional block
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),

    # Third convolutional block
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),

    # Fourth convolutional block
    layers.Conv2D(256, 3, activation='relu'),
    layers.MaxPooling2D(),

    # Flatten and dense layers
    layers.GlobalAveragePooling2D(),  # Better than Flatten for this case
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

# =============================================================================
# STEP 6: COMPILE MODEL
# =============================================================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Print model summary
print("📋 Model architecture:")
model.summary()

# =============================================================================
# STEP 7: TRAIN MODEL
# =============================================================================
print("🚀 Starting training...")

# Add callbacks for better training
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=callbacks,
    verbose=1
)

# =============================================================================
# STEP 8: SAVE MODEL AND METADATA
# =============================================================================
model.save('currency_classifier_model.keras')
print("✅ Model saved as 'currency_classifier_model.keras'")

# Save class names and label encoder for later use
import pickle

with open('class_names.pkl', 'wb') as f:
    pickle.dump(label_encoder.classes_, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("✅ Class names and label encoder saved")

# =============================================================================
# STEP 9: DISPLAY TRAINING RESULTS
# =============================================================================
print("\n📈 Training completed!")
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
print(f"Final training accuracy: {final_train_acc:.4f}")
print(f"Final validation accuracy: {final_val_acc:.4f}")

# Show all detected classes
print(f"\n🏷️  Detected currency classes:")
for i, class_name in enumerate(label_encoder.classes_):
    print(f"  {i}: {class_name}")