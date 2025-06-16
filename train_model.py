# =============================================================================
# ENHANCED TENSORFLOW CURRENCY CLASSIFIER
# =============================================================================
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers.experimental import AdamW
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.applications import EfficientNetV2B2
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

# =============================================================================
# CONFIGURATION
# =============================================================================
tf.get_logger().setLevel('ERROR')
tf.random.set_seed(42)
np.random.seed(42)

# GPU setup
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✅ {len(gpus)} GPU(s) configured.")

IMG_SIZE = 224
BATCH_SIZE = 32
DATASET_DIR = 'dataset'
TRAIN_SPLIT, VAL_SPLIT = 0.7, 0.15

# =============================================================================
# DATA LOADING
# =============================================================================
print("📘 Loading dataset...")
full_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    label_mode='categorical',
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

CLASS_NAMES = full_dataset.class_names
NUM_CLASSES = len(CLASS_NAMES)
print(f"✅ Classes: {CLASS_NAMES}")

total_batches = tf.data.experimental.cardinality(full_dataset).numpy()
train_size = int(TRAIN_SPLIT * total_batches)
val_size = int(VAL_SPLIT * total_batches)

train_dataset = full_dataset.take(train_size)
val_dataset = full_dataset.skip(train_size).take(val_size)
test_dataset = full_dataset.skip(train_size + val_size)

# =============================================================================
# AUGMENTATION + PREPROCESSING
# =============================================================================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.1),
], name="augmentation")

def prepare(ds, augment=False, shuffle=False):
    ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(1000)
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda x, y: (preprocess_input(x), y),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)

train_ds = prepare(train_dataset, augment=True, shuffle=True)
val_ds = prepare(val_dataset)
test_ds = prepare(test_dataset)

# =============================================================================
# MODEL BUILDING
# =============================================================================
def build_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    base_model = EfficientNetV2B2(include_top=False, weights='imagenet', input_shape=input_shape)
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs)

model = build_model()
model.summary()

# =============================================================================
# PHASE 1: TRAIN CLASSIFICATION HEAD
# =============================================================================
print("\n📘 Phase 1: Training top layers...")
model.compile(
    optimizer=AdamW(learning_rate=1e-3),
    loss=CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

callbacks_phase1 = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
]

history1 = model.fit(train_ds, validation_data=val_ds, epochs=40, callbacks=callbacks_phase1)

# =============================================================================
# PHASE 2: FINE-TUNING
# =============================================================================
print("\n📘 Phase 2: Fine-tuning full model...")
base_model = model.layers[1]
base_model.trainable = True

fine_tune_at = int(len(base_model.layers) * 0.6)
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

decay_steps = tf.data.experimental.cardinality(train_ds).numpy() * 60
cosine_schedule = CosineDecay(initial_learning_rate=1e-5, decay_steps=decay_steps)

model.compile(
    optimizer=AdamW(learning_rate=cosine_schedule),
    loss=CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=len(history1.history['loss']) + 60,
    initial_epoch=len(history1.history['loss']),
    callbacks=callbacks_phase1
)

# =============================================================================
# SAVE COMPLETE MODEL
# =============================================================================
print("\n💾 Saving full model...")
model.save("best_currency_model.keras")

# =============================================================================
# EVALUATE ON TEST SET
# =============================================================================
print("\n📊 Evaluating on test set...")
# Reload to confirm everything is correct
model = tf.keras.models.load_model("best_currency_model.keras")
test_loss, test_acc = model.evaluate(test_ds)
print(f"✅ Test Accuracy: {test_acc:.4f}")
print(f"✅ Test Loss: {test_loss:.4f}")

# =============================================================================
# VISUALIZE AUGMENTED IMAGES
# =============================================================================
print("\n🎨 Visualizing augmented samples...")
plt.figure(figsize=(12, 8))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow((images[i].numpy() + 1) / 2)  # De-normalize
        plt.title(CLASS_NAMES[np.argmax(labels[i])])
        plt.axis("off")
plt.tight_layout()
plt.savefig("augmented_images_preview.png")
plt.show()

print("\n🎉 Training and evaluation complete.")
