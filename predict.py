# =============================================================================
# MAXIMUM ACCURACY CURRENCY PREDICTOR
# =============================================================================
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pickle

# =============================================================================
# LOAD IMAGE PATH
# =============================================================================
# Load selected image path saved externally
with open("selected_image_path.txt", "r") as f:
    image_path = f.read().strip()

# =============================================================================
# LOAD TRAINED MODEL AND CLASS LABELS
# =============================================================================
# Load best trained model and class names
model = load_model("currency_classifier_final_model.h5")
with open("class_names.pkl", "rb") as f:
    class_names = pickle.load(f)

# =============================================================================
# PREPROCESS INPUT IMAGE
# =============================================================================
# Load and preprocess the selected image for prediction
img = load_img(image_path, target_size=(260, 260))  # Match model input size
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0  # Normalize to [0,1]

# =============================================================================
# MAKE PREDICTION
# =============================================================================
predictions = model.predict(img_array)
top_3_indices = np.argsort(predictions[0])[::-1][:3]

# =============================================================================
# DISPLAY RESULTS
# =============================================================================
print("Top Predictions:")
for idx in top_3_indices:
    label = class_names[idx]
    confidence = predictions[0][idx] * 100
    print(f"- {label}: {confidence:.2f}%")
