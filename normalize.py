from PIL import Image
import numpy as np
import cv2

def auto_crop(image_path):
    """
    Fungsi crop otomatis menggunakan threshold dan kontur terbesar (biasanya objek utama).
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        cropped = img[y:y+h, x:x+w]
        pil_image = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        return pil_image
    else:
        return Image.open(image_path).convert("RGB")

def normalize_image(image_path, size=(260, 260), grayscale=False):
    """
    Normalize gambar dengan auto-crop untuk CNN input (260x260).
    """
    try:
        img = auto_crop(image_path)

        if grayscale:
            img = img.convert('L')
        else:
            img = img.convert('RGB')

        img_resized = img.resize(size)
        img_array = np.asarray(img_resized, dtype=np.float32) / 255.0

        if grayscale:
            img_array = img_array.reshape(1, size[1], size[0], 1)
        else:
            img_array = img_array.reshape(1, size[1], size[0], 3)

        return img_array
    except Exception as e:
        print(f"Error in normalization: {e}")
        return None
