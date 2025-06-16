# =============================================================================
# ENHANCED TENSORFLOW CURRENCY CLASSIFIER
# =============================================================================

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers.experimental import AdamW
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.applications import EfficientNetV2B2
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

# =============================================================================
# CONFIGURATION
# =============================================================================
tf.get_logger().setLevel('ERROR') # Suppress verbose TensorFlow logs
tf.random.set_seed(42)
np.random.seed(42)

# GPU configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ {len(gpus)} GPU(s) configured.")
    except RuntimeError as e:
        print(e)

# Parameters
# Using 224x224, a standard size for ImageNet-based models like ResNet/EfficientNet
IMG_SIZE = 224
BATCH_SIZE = 32 # Increased batch size for tf.data efficiency
DATASET_DIR = 'dataset' # Single directory containing class subfolders

# Define the train/validation/test split ratios
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
# TEST_SPLIT will be the remainder

# =============================================================================
# NEW DATA PIPELINE: Loading & Splitting with tf.data
# =============================================================================
# This section replaces the old ImageDataGenerator and pre-split folders.
# It loads all data from the root 'dataset' directory and splits it on the fly.

print("📘 Loading and splitting dataset with tf.data...")
# Load the full dataset from the directory
full_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    label_mode='categorical',
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

# Get class names from the dataset object
CLASS_NAMES = full_dataset.class_names
NUM_CLASSES = len(CLASS_NAMES)
print(f"✅ Classes detected: {CLASS_NAMES}")

# Calculate split sizes
dataset_size = tf.data.experimental.cardinality(full_dataset).numpy()
train_size = int(TRAIN_SPLIT * dataset_size)
val_size = int(VAL_SPLIT * dataset_size)
test_size = dataset_size - train_size - val_size

# Split the dataset
train_dataset = full_dataset.take(train_size)
val_dataset = full_dataset.skip(train_size).take(val_size)
test_dataset = full_dataset.skip(train_size + val_size).take(test_size)

print(f"✅ Dataset split complete:")
print(f"   Training batches: {tf.data.experimental.cardinality(train_dataset).numpy()}")
print(f"   Validation batches: {tf.data.experimental.cardinality(val_dataset).numpy()}")
print(f"   Test batches: {tf.data.experimental.cardinality(test_dataset).numpy()}")


# =============================================================================
# DATA AUGMENTATION & PREPROCESSING
# =============================================================================
# Define a model for GPU-accelerated data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.1),
], name="data_augmentation")

# Function to apply augmentation and preprocessing
def prepare(ds, augment=False, shuffle=False):
    # Apply caching for performance
    ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(buffer_size=1000)

    # Apply augmentation if this is the training set
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)

    # Apply the model-specific preprocessing function (normalizes to what EfficientNetV2 expects)
    ds = ds.map(lambda x, y: (preprocess_input(x), y),
                num_parallel_calls=tf.data.AUTOTUNE)

    # Prefetch data for performance
    return ds.prefetch(buffer_size=tf.data.AUTOTUNE)

train_ds = prepare(train_dataset, augment=True, shuffle=True)
val_ds = prepare(val_dataset)
test_ds = prepare(test_dataset)

# =============================================================================
# MODEL DEFINITION
# =============================================================================
def build_currency_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    # Using EfficientNetV2-B2 - a powerful and efficient model
    base_model = EfficientNetV2B2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze base for initial training

    inputs = tf.keras.Input(shape=input_shape)
    # The data is already preprocessed, so we pass it directly
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model

model = build_currency_model()
model.summary()

# =============================================================================
# PHASE 1: TRAIN CLASSIFICATION HEAD
# =============================================================================
print("\n📘 Starting Phase 1: Training classifier head...")
model.compile(
    optimizer=AdamW(learning_rate=1e-3),
    loss=CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint("best_currency_model.keras", monitor='val_accuracy', save_best_only=True)
]

history1 = model.fit(train_ds, validation_data=val_ds, epochs=40, callbacks=callbacks)

# =============================================================================
# PHASE 2: FINE-TUNE FULL MODEL
# =============================================================================
print("\n📘 Starting Phase 2: Fine-tuning full model...")
base_model = model.layers[1] # Get the base model layer
base_model.trainable = True # Unfreeze the base model

# Fine-tune from this layer onwards. Freezing the first few stages can be beneficial.
fine_tune_at = int(len(base_model.layers) * 0.6) # Unfreeze top 40% of layers
print(f"Unfreezing from layer {fine_tune_at} onwards.")
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Use a very low learning rate with cosine decay for fine-tuning
fine_tune_epochs = 60
total_epochs = len(history1.history['loss']) + fine_tune_epochs
decay_steps = tf.data.experimental.cardinality(train_ds).numpy() * fine_tune_epochs
cosine_schedule = CosineDecay(initial_learning_rate=1e-5, decay_steps=decay_steps)

model.compile(
    optimizer=AdamW(learning_rate=cosine_schedule, weight_decay=1e-5),
    loss=CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=total_epochs,
    initial_epoch=len(history1.history['loss']),
    callbacks=callbacks # Re-using same callbacks is fine
)

# =============================================================================
# FINAL EVALUATION ON HELD-OUT TEST SET
# =============================================================================
print("\n📊 Evaluating final model on the unseen test set...")
# Load the best weights saved during the entire training process
model.load_weights("best_currency_model.keras")
test_loss, test_acc = model.evaluate(test_ds)
print(f"✅ Test Set Accuracy: {test_acc:.4f}")
print(f"✅ Test Set Loss: {test_loss:.4f}")

# =============================================================================
# VISUALIZE A BATCH OF AUGMENTED IMAGES (like the PyTorch script)
# =============================================================================
print("\n🎨 Visualizing a batch of augmented training images...")
plt.figure(figsize=(12, 8))
# Take one batch from the training dataset
for images, labels in train_ds.take(1):
    for i in range(9): # Display 9 images
        ax = plt.subplot(3, 3, i + 1)
        # We need to de-normalize for visualization.
        # EfficientNetV2 preprocess_input scales to [-1, 1] so we shift and scale back to [0, 1]
        plt.imshow((images[i].numpy() + 1) / 2)
        plt.title(CLASS_NAMES[tf.argmax(labels[i])])
        plt.axis("off")
plt.tight_layout()
plt.savefig("augmented_images_preview.png")
plt.show()

print("\n🎉 Training and evaluation complete!")