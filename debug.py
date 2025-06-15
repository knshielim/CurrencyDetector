import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle

# Ubah ke path dataset kamu
DATASET_DIR = "dataset/train"  # atau sesuai dengan struktur aslimu
IMG_SIZE = 260
BATCH_SIZE = 4

# Load class names (jika perlu untuk label mapping)
if os.path.exists("class_names.pkl"):
    with open("class_names.pkl", "rb") as f:
        CLASS_NAMES = pickle.load(f)
else:
    CLASS_NAMES = sorted(os.listdir(DATASET_DIR))

# Setup ImageDataGenerator seperti di train_model.py
datagen = ImageDataGenerator(
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

generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

# Ambil satu batch dan tampilkan
images, labels = next(generator)

for i in range(len(images)):
    plt.imshow(images[i])
    class_idx = np.argmax(labels[i])
    class_name = CLASS_NAMES[class_idx]
    plt.title(f"Augmented - Class: {class_name}")
    plt.axis("off")
    plt.show()
