# =============================================================================
# STEP 1 - Import Libraries
# =============================================================================
import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight

# =============================================================================
# STEP 2 - Setup Paths and Constants
# =============================================================================
base_dir = 'dataset'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

img_height, img_width = 180, 180
batch_size = 16  # Reduced batch size for better gradient updates
num_classes = len(os.listdir(train_dir))

print(f"Number of classes detected: {num_classes}")
print(f"Classes: {os.listdir(train_dir)}")

# =============================================================================
# STEP 3 - Create Image Data Generators
# =============================================================================
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=15,  # Reduced rotation for currency notes
    width_shift_range=0.1,  # Small shifts
    height_shift_range=0.1,
    shear_range=0.1,  # Small shear transformation
    zoom_range=0.15,  # Reduced zoom range
    horizontal_flip=False,  # Currency notes shouldn't be flipped
    vertical_flip=False,  # Currency notes shouldn't be flipped
    fill_mode='nearest',
    brightness_range=[0.8, 1.2]  # Brightness variation
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

# Save class names for later use
import pickle

with open("class_names.pkl", "wb") as f:
    pickle.dump(list(train_generator.class_indices.keys()), f)

validation_generator = val_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
    seed=42
)


# =============================================================================
# STEP 4 - Calculate Class Weights for Imbalanced Dataset
# =============================================================================
def calculate_class_weights(generator):
    """Calculate class weights to handle imbalanced dataset"""
    # Get all labels
    labels = []
    for i in range(len(generator)):
        _, batch_labels = generator[i]
        labels.extend(np.argmax(batch_labels, axis=1))
        if i >= len(generator) - 1:
            break

    # Calculate class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(labels),
        y=labels
    )

    return dict(enumerate(class_weights))


class_weights = calculate_class_weights(train_generator)
print(f"Class weights: {class_weights}")


# =============================================================================
# STEP 5 - Build CNN Model
# =============================================================================
def create_improved_model(num_classes, img_height, img_width):
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        # Fourth Convolutional Block
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        # Fully Connected Layers
        layers.GlobalAveragePooling2D(),  # Better than Flatten for overfitting
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model


model = create_improved_model(num_classes, img_height, img_width)

# =============================================================================
# STEP 6 - Compile the Model
# =============================================================================
model.compile(
    optimizer=Adam(learning_rate=0.001),  # Lower learning rate
    loss='categorical_crossentropy',
    metrics=['accuracy', 'top_k_categorical_accuracy']
)

# Print model summary
model.summary()

# =============================================================================
# STEP 7 - Setup Callbacks
# =============================================================================
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    ModelCheckpoint(
        'best_currency_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# =============================================================================
# STEP 8 - Train the Model
# =============================================================================
EPOCHS = 50  # Increased epochs with early stopping

print("Starting training...")
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights,  # Use class weights
    verbose=1
)

# =============================================================================
# STEP 9 - Load Best Model and Save
# =============================================================================
# Load the best model
best_model = tf.keras.models.load_model('best_currency_model.h5')

# Save as the final model
best_model.save('currency_classifier_model.h5')
print("Model saved as 'currency_classifier_model.h5'")

# =============================================================================
# STEP 10 - Evaluate Model Performance
# =============================================================================
print("\nEvaluating model on validation set...")
val_loss, val_accuracy, val_top_k = best_model.evaluate(validation_generator, verbose=1)
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Validation Top-K Accuracy: {val_top_k:.4f}")


# =============================================================================
# STEP 11 - Plot Training History
# =============================================================================
def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(15, 5))

    # Plot accuracy
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy', marker='o')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy', marker='s')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)

    # Plot loss
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, loss, label='Training Loss', marker='o')
    plt.plot(epochs_range, val_loss, label='Validation Loss', marker='s')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)

    # Plot learning rate (if available)
    plt.subplot(1, 3, 3)
    if 'lr' in history.history:
        plt.plot(epochs_range, history.history['lr'], label='Learning Rate', marker='o')
        plt.legend()
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, 'Learning Rate\nNot Recorded', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Learning Rate Schedule')

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()


plot_training_history(history)


# =============================================================================
# STEP 12 - Generate Classification Report
# =============================================================================
def generate_classification_report(model, generator):
    """Generate detailed classification report"""
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns

    # Get predictions
    generator.reset()
    predictions = model.predict(generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)

    # Get true labels
    true_classes = generator.classes
    class_labels = list(generator.class_indices.keys())

    # Classification report
    report = classification_report(true_classes, predicted_classes,
                                   target_names=class_labels, digits=4)
    print("\nClassification Report:")
    print(report)

    # Confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()


# Generate classification report
generate_classification_report(best_model, validation_generator)

print("\n" + "=" * 50)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 50)
print(f"Final model saved as: currency_classifier_model.h5")
print(f"Best model saved as: best_currency_model.h5")
print(f"Training plots saved as: training_history.png")
print(f"Confusion matrix saved as: confusion_matrix.png")