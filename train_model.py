# =============================================================================
# IMPORT LIBRARIES
# =============================================================================
import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.applications import EfficientNetB2
from sklearn.utils.class_weight import compute_class_weight

# =============================================================================
# CONFIGURATION
# =============================================================================
tf.random.set_seed(42)
np.random.seed(42)

# Enable GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ {len(gpus)} GPU(s) configured.")
    except RuntimeError as e:
        print(e)

# Paths
base_dir = 'dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'validation')

# Parameters
IMG_SIZE = 260
BATCH_SIZE = 16
NUM_CLASSES = len(os.listdir(train_dir))
CLASS_NAMES = sorted(os.listdir(train_dir))

print(f"Classes Detected: {CLASS_NAMES}")

# Save class names
with open("class_names.pkl", "wb") as f:
    pickle.dump(CLASS_NAMES, f)

# =============================================================================
# DATA AUGMENTATION
# =============================================================================
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    shear_range=0.1,
    horizontal_flip=True,
    brightness_range=(0.7, 1.3),
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# =============================================================================
# COMPUTE CLASS WEIGHTS
# =============================================================================
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))
print(f"Class Weights: {class_weights}")

# =============================================================================
# MODEL DEFINITION
# =============================================================================
def build_currency_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    base_model = EfficientNetB2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze base model for Phase 1

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model

model = build_currency_model()

# =============================================================================
# PHASE 1: TRAIN HEAD ONLY
# =============================================================================
model.compile(
    optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-4),
    loss=CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ModelCheckpoint("best_currency_model.h5", monitor='val_accuracy', save_best_only=True)
]

print("\n📘 Starting Phase 1: Training classifier head")
history1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    class_weight=class_weights,
    callbacks=callbacks
)

# =============================================================================
# PHASE 2: FINE-TUNE FULL MODEL
# =============================================================================
print("\n📘 Starting Phase 2: Fine-tuning full model")

model.trainable = True  # Unfreeze entire model

cosine_schedule = CosineDecay(
    initial_learning_rate=1e-5,
    decay_steps=(train_generator.samples // BATCH_SIZE) * 50,
    alpha=0.0
)

model.compile(
    optimizer=AdamW(learning_rate=cosine_schedule, weight_decay=1e-5),
    loss=CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

callbacks_ft = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint("best_currency_model.h5", monitor='val_accuracy', save_best_only=True)
]

history2 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=100,
    initial_epoch=len(history1.history['loss']),
    class_weight=class_weights,
    callbacks=callbacks_ft
)

# =============================================================================
# SAVE FINAL MODEL
# =============================================================================
model.save("currency_classifier_final_model.h5")
print("✅ Final model saved as currency_classifier_final_model.h5")

# =============================================================================
# FINAL EVALUATION
# =============================================================================
val_loss, val_acc = model.evaluate(val_generator)
print(f"📊 Final Validation Accuracy: {val_acc:.4f}")
