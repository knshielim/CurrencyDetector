from PIL import Image
import numpy as np
import cv2

def auto_crop(image_path, padding_ratio=0.05):
    """
    Fungsi crop otomatis dengan padding tambahan agar tidak terlalu agresif.
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

        # Tambahkan padding
        pad_x = int(w * padding_ratio)
        pad_y = int(h * padding_ratio)

        x1 = max(x - pad_x, 0)
        y1 = max(y - pad_y, 0)
        x2 = min(x + w + pad_x, img.shape[1])
        y2 = min(y + h + pad_y, img.shape[0])

        cropped = img[y1:y2, x1:x2]
        pil_image = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        return pil_image
    else:
        return Image.open(image_path).convert("RGB")

"""
def auto_crop(image_path):
    
    # Fungsi crop otomatis menggunakan threshold dan kontur terbesar (biasanya objek utama).
    
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
"""
def normalize_image(image_path, size=(260, 260), grayscale=False):
    """
    Normalize gambar dengan auto-crop untuk CNN input (260x260).
    """
    img_raw = Image.open(image_path)
    img_raw.save("debug_original.jpg")
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
