import tensorflow as tf
import numpy as np
import os

# --- 1. Configuration & Placeholders ---
# These are the parameters you will need to set based on your final dataset.
# For now, we'll use some dummy values.

# Image dimensions
IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_CHANNELS = 3
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# Number of output classes for each category
# TODO: Update these numbers when your dataset is finalized.
NUM_COUNTRIES = 5  # Example: USD, MYR, EUR, SGD, IDR
NUM_NOMINALS = 6   # Example: 1, 5, 10, 20, 50, 100
NUM_YEARS = 4      # Example: 2009, 2012, 2013, 2018

# Training parameters
BATCH_SIZE = 32
EPOCHS = 10 # Start with a small number for testing, increase for real training (e.g., 30-50)


# --- 2. Dummy Data Generation ---
# This section is a PLACEHOLDER for your real data pipeline.
# Member 1 will provide the 'dataset_split/' folder.
# Once you have that, you can replace this entire section with:
# `tf.keras.utils.image_dataset_from_directory()`

def generate_dummy_data(num_samples, num_countries, num_nominals, num_years):
    """
    Generates a dummy dataset of random images and corresponding labels.
    This function simulates the data loading process.

    Args:
        num_samples (int): The total number of dummy samples to generate.
        num_countries (int): The number of country classes.
        num_nominals (int): The number of nominal value classes.
        num_years (int): The number of year/version classes.

    Returns:
        tf.data.Dataset: A TensorFlow dataset ready for training.
    """
    print(f"Generating {num_samples} dummy samples...")
    # Generate random "images" (random pixel values)
    dummy_images = np.random.rand(num_samples, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS).astype(np.float32)

    # Generate random integer labels for each of the three outputs
    dummy_country_labels = np.random.randint(0, num_countries, size=num_samples)
    dummy_nominal_labels = np.random.randint(0, num_nominals, size=num_samples)
    dummy_year_labels = np.random.randint(0, num_years, size=num_samples)

    # The model expects a dictionary of labels, where keys match the output layer names
    dummy_labels = {
        "country_output": dummy_country_labels,
        "nominal_output": dummy_nominal_labels,
        "year_output": dummy_year_labels
    }

    # Create a tf.data.Dataset object
    dataset = tf.data.Dataset.from_tensor_slices((dummy_images, dummy_labels))
    dataset = dataset.shuffle(buffer_size=num_samples).batch(BATCH_SIZE)
    print("Dummy data generation complete.")
    return dataset

# Generate the datasets
# In a real scenario, you'd have separate folders for train, val, and test
train_dataset = generate_dummy_data(1000, NUM_COUNTRIES, NUM_NOMINALS, NUM_YEARS)
val_dataset = generate_dummy_data(200, NUM_COUNTRIES, NUM_NOMINALS, NUM_YEARS)
test_dataset = generate_dummy_data(200, NUM_COUNTRIES, NUM_NOMINALS, NUM_YEARS)


# --- 3. Model Design (Transfer Learning) ---
def build_model(num_countries, num_nominals, num_years):
    """
    Builds a multi-output currency classification model using MobileNetV2.

    Args:
        num_countries (int): The number of country classes.
        num_nominals (int): The number of nominal value classes.
        num_years (int): The number of year/version classes.

    Returns:
        tf.keras.Model: The compiled Keras model.
    """
    print("Building model...")
    # Load the base model (MobileNetV2) pre-trained on ImageNet
    # We exclude the top classification layer (`include_top=False`)
    # to add our own custom output layers.
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SHAPE,
        include_top=False,
        weights='imagenet'
    )

    # Freeze the layers of the base model so we only train our new layers
    base_model.trainable = False

    # Get the output of the base model
    base_model_output = base_model.output

    # Add a pooling layer to reduce the spatial dimensions
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model_output)
    x = tf.keras.layers.Dropout(0.3)(x) # Add dropout for regularization

    # --- Create the three separate output layers ("heads") ---
    # Each output is a Dense layer with a softmax activation function.
    # The name of each layer is important, as it will be used to match the labels.
    output_country = tf.keras.layers.Dense(num_countries, activation='softmax', name='country_output')(x)
    output_nominal = tf.keras.layers.Dense(num_nominals, activation='softmax', name='nominal_output')(x)
    output_year = tf.keras.layers.Dense(num_years, activation='softmax', name='year_output')(x)

    # Create the final model with one input and three outputs
    model = tf.keras.Model(
        inputs=base_model.input,
        outputs=[output_country, output_nominal, output_year]
    )
    print("Model built successfully.")
    return model

# Instantiate the model
model = build_model(NUM_COUNTRIES, NUM_NOMINALS, NUM_YEARS)

# Display the model's architecture
model.summary()


# --- 4. Compile the Model ---
print("Compiling model...")
# When compiling a multi-output model, you MUST specify metrics for each output.
# Using a dictionary is the clearest way. The keys must match the output layer names.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss={
        'country_output': 'sparse_categorical_crossentropy',
        'nominal_output': 'sparse_categorical_crossentropy',
        'year_output': 'sparse_categorical_crossentropy'
    },
    metrics={
        'country_output': 'accuracy',
        'nominal_output': 'accuracy',
        'year_output': 'accuracy'
    }
)
print("Model compiled.")


# --- 5. Train the Model ---
# TODO: Add callbacks for more robust training on the real dataset.
# Example:
# callbacks = [
#     tf.keras.callbacks.ModelCheckpoint("currency_classifier.h5", save_best_only=True),
#     tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
# ]
print("Starting model training...")
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset
    # callbacks=callbacks # Uncomment when using real data
)
print("Training complete.")


# --- 6. Evaluate the Model ---
print("Evaluating model on the test set...")
# The evaluation will return the total loss, followed by the loss and accuracy
# for each of the three outputs, in the order they were defined in the compile() call.
results = model.evaluate(test_dataset)

# Print the results in a readable format
print("\n--- Evaluation Results ---")
print(f"Total Loss: {results[0]:.4f}")
print(f"Country Loss: {results[1]:.4f}, Country Accuracy: {results[4]:.4f}")
print(f"Nominal Loss: {results[2]:.4f}, Nominal Accuracy: {results[5]:.4f}")
print(f"Year Loss: {results[3]:.4f}, Year Accuracy: {results[6]:.4f}")
print("--------------------------\n")


# --- 7. Save the Model ---
# This saves the entire model (architecture, weights, optimizer state)
# into a single HDF5 file.
MODEL_SAVE_PATH = "currency_classifier.h5"
print(f"Saving trained model to {MODEL_SAVE_PATH}...")
model.save(MODEL_SAVE_PATH)
print("Model saved successfully.")

# To load the model later for predictions:
# loaded_model = tf.keras.models.load_model(MODEL_SAVE_PATH)
# print("Model loaded successfully for prediction.")

