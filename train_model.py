# =============================================================================
# MAXIMUM ACCURACY CURRENCY CLASSIFIER - TRAINING SCRIPT
# =============================================================================
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.applications import EfficientNetB2
from sklearn.utils.class_weight import compute_class_weight
import pickle
from tqdm import tqdm

# =============================================================================
# CONFIGURATION AND ENVIRONMENT SETUP
# =============================================================================
tf.random.set_seed(42)
np.random.seed(42)

# Enable memory growth for GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Running on {len(gpus)} GPU(s).")
    except RuntimeError as e:
        print(e)

# Set dataset paths
base_dir = 'dataset'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# Image and batch parameters
IMG_SIZE = 260
img_height, img_width = IMG_SIZE, IMG_SIZE
batch_size = 16

# Detect number of classes
num_classes = len(os.listdir(train_dir))
print(f"Number of classes detected: {num_classes}")
print(f"Classes: {sorted(os.listdir(train_dir))}")

# =============================================================================
# DATA GENERATORS
# =============================================================================
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='reflect',
    brightness_range=[0.7, 1.3]
)

val_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    seed=42
)

validation_generator = val_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
    seed=42
)

# Save class label mapping
with open("class_names.pkl", "wb") as f:
    pickle.dump(list(train_generator.class_indices.keys()), f)
print(f"Class mapping saved: {train_generator.class_indices}")

# =============================================================================
# CLASS WEIGHT COMPUTATION
# =============================================================================
class_labels = train_generator.classes
class_weight_values = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(class_labels),
    y=class_labels
)
class_weights = dict(zip(np.unique(class_labels), class_weight_values))
print(f"Calculated class weights: {class_weights}")

# =============================================================================
# MODEL DEFINITION
# =============================================================================
def create_max_accuracy_model(num_classes, img_height, img_width):
    base_model = EfficientNetB2(
        weights='imagenet',
        include_top=False,
        input_shape=(img_height, img_width, 3),
        pooling='avg'
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(img_height, img_width, 3))
    x = base_model(inputs, training=False)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs, outputs)

    return model

model = create_max_accuracy_model(num_classes, img_height, img_width)

# =============================================================================
# COMPILE MODEL - PHASE 1
# =============================================================================
model.compile(
    optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-4),
    loss=CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

model.summary()

# =============================================================================
# CALLBACKS
# =============================================================================
checkpoint_filepath = 'best_currency_model.h5'

callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
    ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_accuracy', save_best_only=True, save_weights_only=False, verbose=1, mode='max')
]

# =============================================================================
# TRAINING - PHASE 1 (Train Classification Head)
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 1: Training the new classification head")
print("=" * 60)

history_head = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=50,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

# =============================================================================
# FINE-TUNING - PHASE 2
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 2: Fine-tuning the full model with Cosine Decay LR")
print("=" * 60)

model.layers[1].trainable = True  # Unfreeze base model

total_steps = (train_generator.samples // batch_size) * 100
cosine_decay_schedule = CosineDecay(
    initial_learning_rate=1e-5,
    decay_steps=total_steps,
    alpha=0.0
)

model.compile(
    optimizer=AdamW(learning_rate=cosine_decay_schedule, weight_decay=1e-5),
    loss=CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

callbacks_finetune = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_accuracy', save_best_only=True, verbose=1, mode='max')
]

initial_epoch = len(history_head.history['loss'])
history_finetune = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=initial_epoch + 100,
    callbacks=callbacks_finetune,
    class_weight=class_weights,
    verbose=1,
    initial_epoch=initial_epoch
)

# =============================================================================
# SAVE FINAL MODEL
# =============================================================================
print("\nLoading best model from checkpoint for final evaluation...")
best_model = tf.keras.models.load_model(checkpoint_filepath)
best_model.save('currency_classifier_final_model.h5')
print("Final best model saved as 'currency_classifier_final_model.h5'")

# =============================================================================
# STANDARD VALIDATION EVALUATION
# =============================================================================
print("\n" + "=" * 50)
print("FINAL EVALUATION ON VALIDATION SET (Standard)")
print("=" * 50)
standard_results = best_model.evaluate(validation_generator, verbose=1)
for name, value in zip(best_model.metrics_names, standard_results):
    print(f"{name.replace('_', ' ').capitalize():<20}: {value:.4f}")

# =============================================================================
# TEST-TIME AUGMENTATION (TTA) EVALUATION
# =============================================================================
print("\n" + "=" * 50)
print("PERFORMING EVALUATION WITH TEST-TIME AUGMENTATION (TTA)")
print("=" * 50)

tta_steps = 5
tta_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='reflect'
)

validation_generator.reset()
predictions_tta = []

for i in tqdm(range(len(validation_generator)), desc="TTA Progress"):
    images, labels = validation_generator[i]
    batch_preds = []
    for j in range(images.shape[0]):
        img = images[j]
        tta_images = np.array([tta_datagen.random_transform(img) for _ in range(tta_steps)])
        preds = best_model.predict(tta_images, verbose=0)
        avg_preds = np.mean(preds, axis=0)
        batch_preds.append(avg_preds)
    predictions_tta.extend(batch_preds)

predicted_classes_tta = np.argmax(np.array(predictions_tta), axis=1)
true_classes = validation_generator.classes
tta_accuracy = np.mean(predicted_classes_tta == true_classes)

print(f"\nStandard Validation Accuracy: {standard_results[1]:.4f}")
print(f"TTA Enhanced Accuracy:      {tta_accuracy:.4f}  <-- Accuracy Boost!")

# =============================================================================
# FINAL TRAINING SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 70)
print(f"✅ Final best model saved as: currency_classifier_final_model.h5")
print(f"✅ Class names saved as: class_names.pkl")
print(f"✅ Model architecture used: Transfer Learning (EfficientNetB2)")
print(f"✅ Final Standard Accuracy: {standard_results[1]:.4f}")
print(f"✅ Final TTA Accuracy: {tta_accuracy:.4f}")
print("=" * 70)
print("\n🎉 Your maximum accuracy currency classifier is ready!")
