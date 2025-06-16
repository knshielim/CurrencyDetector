# =============================================================================
# Currency Classification Model using EfficientNetV2B2
# =============================================================================

import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import EfficientNetV2B2
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Fix OMP error in Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# =============================================================================
# STEP 1 - Set Paths and Parameters
# =============================================================================
ZIP_FILE = "dataset.zip"
EXTRACT_PATH = "dataset"
IMG_SIZE = (260, 260)
BATCH_SIZE = 32
EPOCHS_STAGE1 = 40
EPOCHS_STAGE2 = 40
MODEL_PATH = "best_currency_model.keras"

# =============================================================================
# STEP 2 - Extract Dataset
# =============================================================================
if not os.path.exists(EXTRACT_PATH):
    with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
        zip_ref.extractall()

train_dir = os.path.join(EXTRACT_PATH, "train")
val_dir = os.path.join(EXTRACT_PATH, "validation")
test_dir = os.path.join(EXTRACT_PATH, "test")

# =============================================================================
# STEP 3 - Load Datasets
# =============================================================================
train_ds = image_dataset_from_directory(train_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="categorical")
val_ds = image_dataset_from_directory(val_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="categorical")
test_ds = image_dataset_from_directory(test_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="categorical")

CLASS_NAMES = train_ds.class_names
NUM_CLASSES = len(CLASS_NAMES)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

# =============================================================================
# STEP 4 - Data Augmentation
# =============================================================================
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
    tf.keras.layers.RandomBrightness(0.1)
], name="data_augmentation")

# =============================================================================
# STEP 5 - Build Model
# =============================================================================
base_model = EfficientNetV2B2(include_top=False, weights="imagenet", input_shape=IMG_SIZE + (3,))
base_model.trainable = False

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = Model(inputs, outputs)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# =============================================================================
# STEP 6 - Callbacks
# =============================================================================
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3)
]

# =============================================================================
# STEP 7 - Compute Class Weights
# =============================================================================
all_labels = []
for _, labels in train_ds.unbatch():
    all_labels.append(np.argmax(labels.numpy()))
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(all_labels), y=all_labels)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

# =============================================================================
# STEP 8 - Train Model: Stage 1
# =============================================================================
print("📈 Training Stage 1: Feature Extraction...")
history1 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_STAGE1,
                     callbacks=callbacks, class_weight=class_weights_dict)

# =============================================================================
# STEP 9 - Fine-tune Model: Stage 2
# =============================================================================
print("🔁 Training Stage 2: Fine-tuning...")
base_model.trainable = True
for layer in base_model.layers[:100]:  # Optionally freeze early layers
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="categorical_crossentropy", metrics=["accuracy"])
history2 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_STAGE2,
                     callbacks=callbacks, class_weight=class_weights_dict)

# =============================================================================
# STEP 10 - Evaluate on Test Set
# =============================================================================
print("\n📊 Evaluating on test set...")
model.load_weights(MODEL_PATH)
test_loss, test_acc = model.evaluate(test_ds)
print(f"✅ Test Accuracy: {test_acc:.4f}")
print(f"✅ Test Loss: {test_loss:.4f}")

# =============================================================================
# STEP 11 - Visualize Augmented Samples
# =============================================================================
print("\n🎨 Visualizing augmented samples...")
plt.figure(figsize=(12, 8))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        image = tf.clip_by_value((images[i] * 127.5 + 127.5) / 255.0, 0, 1)
        plt.imshow(image)
        plt.title(CLASS_NAMES[np.argmax(labels[i])])
        plt.axis("off")
plt.tight_layout()
plt.savefig("augmented_images_preview.png")
plt.show()
