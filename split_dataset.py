import os
import zipfile
import shutil
import random

# =============================================================================
# STEP 1: SETUP PATHS AND CONSTANTS
# =============================================================================
ZIP_PATH = "dataset.zip"                   # Your zip file
EXTRACT_DIR = "dataset"                    # Folder to extract to
TRAIN_DIR = os.path.join(EXTRACT_DIR, "train")
VAL_DIR = os.path.join(EXTRACT_DIR, "validation")
SPLIT_RATIO = 0.8                          # 80% train, 20% val

# =============================================================================
# STEP 2: EXTRACT ZIP FILE AND FLATTEN
# =============================================================================
if not os.path.exists(EXTRACT_DIR):
    os.makedirs(EXTRACT_DIR)

print(f"📦 Extracting '{ZIP_PATH}' to '{EXTRACT_DIR}'...")
with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(EXTRACT_DIR)

# Flatten nested dataset folder if needed
nested_dataset = os.path.join(EXTRACT_DIR, 'dataset')
if os.path.exists(nested_dataset):
    print("🔄 Flattening nested 'dataset/dataset/' structure...")
    for folder in os.listdir(nested_dataset):
        src = os.path.join(nested_dataset, folder)
        dst = os.path.join(EXTRACT_DIR, folder)
        if not os.path.exists(dst):
            shutil.move(src, dst)
    shutil.rmtree(nested_dataset)

# Remove macOS metadata folder if exists
macosx_folder = os.path.join(EXTRACT_DIR, '__MACOSX')
if os.path.exists(macosx_folder):
    shutil.rmtree(macosx_folder)
    print("🗑️ Removed __MACOSX folder.")

# =============================================================================
# STEP 3: SPLIT INTO TRAIN/VALIDATION
# =============================================================================
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

print("\n🔍 Scanning for class folders...\n")
class_folders = [f for f in os.listdir(EXTRACT_DIR)
                 if os.path.isdir(os.path.join(EXTRACT_DIR, f)) and f not in ['train', 'validation']]

for class_name in class_folders:
    class_path = os.path.join(EXTRACT_DIR, class_name)
    images = [f for f in os.listdir(class_path)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not images:
        print(f"⚠️ No images in '{class_name}', skipping...\n")
        continue

    random.shuffle(images)
    split_index = int(len(images) * SPLIT_RATIO)
    train_images = images[:split_index]
    val_images = images[split_index:]

    train_class_dir = os.path.join(TRAIN_DIR, class_name)
    val_class_dir = os.path.join(VAL_DIR, class_name)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(val_class_dir, exist_ok=True)

    for img in train_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(train_class_dir, img))
    for img in val_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(val_class_dir, img))

    print(f"✅ {len(train_images)} images → train/{class_name}")
    print(f"✅ {len(val_images)} images → validation/{class_name}\n")

print("🎉 All done! Dataset is split into 'train/' and 'validation/' folders.")
