from PIL import Image
import numpy as np

def normalize_image(image_path, size=(128, 128), grayscale=True):
    """
    Normalize gambar untuk CNN input.

    Args:
        image_path (str): Path ke gambar.
        size (tuple): Ukuran (width, height).
        grayscale (bool): True jika ingin grayscale.

    Returns:
        np.ndarray: Array ternormalisasi dengan shape (1, H, W, C)
    """
    try:
        img = Image.open(image_path)

        if grayscale:
            img = img.convert('L')  # grayscale
        else:
            img = img.convert('RGB')  # warna

        img_resized = img.resize(size)
        img_array = np.asarray(img_resized, dtype=np.float32) / 255.0

        # Reshape jadi batch format untuk CNN
        if grayscale:
            img_array = img_array.reshape(1, size[1], size[0], 1)  # (1, height, width, 1)
        else:
            img_array = img_array.reshape(1, size[1], size[0], 3)  # (1, height, width, 3)

        return img_array
    except Exception as e:
        print(f"Error in normalization: {e}")
        return None
