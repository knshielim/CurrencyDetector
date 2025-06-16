import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from photo_augmentation import data_augmentation

# File location
dataset_path = "dataset"
img_height = 224
img_width = 224
batch_size = 32
seed = 123

# Load raw dataset
raw_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    image_size=(img_height, img_width),
    batch_size=None,
    label_mode='int',
    shuffle=False  # Ensure repeatability
)

# Extract file paths and labels
file_paths = raw_ds.file_paths
labels = [label.numpy() for _, label in raw_ds]

# Split dataset into train, val, and test (70/15/15)
train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    file_paths, labels, test_size=0.3, stratify=labels, random_state=seed
)

val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths, temp_labels, test_size=0.5, stratify=temp_labels, random_state=seed
)

#  helper to wrap "data_augmentation"
def preprocess_image(path, label, is_training=False):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [img_height, img_width])
    img = tf.cast(img, tf.float32) / 255.0

    if is_training:
        img = data_augmentation(img, training=True)
    return img, label

# Dataset creation function using augmentation
def create_dataset(paths, labels, is_training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda x, y: preprocess_image(x, y, is_training), num_parallel_calls=tf.data.AUTOTUNE)
    if is_training:
        ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# Create the datasets
train_ds = create_dataset(train_paths, train_labels, is_training=True)
val_ds = create_dataset(val_paths, val_labels, is_training=False)
test_ds = create_dataset(test_paths, test_labels, is_training=False)

# Show dataset sizes
print(" Dataset Summary:")
print(f"Total images: {len(file_paths)}")
print(f"Training set: {len(train_paths)} images")
print(f"Validation set: {len(val_paths)} images")
print(f"Test set: {len(test_paths)} images")
