import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import os
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model


def normalize_image(filepath, size=(128, 128)):
    """
    Preprocesses an image file to match model input format.
    Resizes, normalizes, and reshapes the image as per training config.
    """
    try:
        img = tf.io.read_file(filepath)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, size)
        img = img / 255.0  # Normalize as done in training
        input_array = tf.expand_dims(img, axis=0)  # Add batch dimension
        return input_array
    except Exception as e:
        messagebox.showerror("Error", f"Image preprocessing failed:\n{e}")
        return None


class FileUploaderApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Currency Detector")
        self.geometry("600x550")
        self.configure(bg="#f0f2f5")

        try:
            self.model = load_model("currency_model.h5")
        except Exception as e:
            messagebox.showerror("Model Load Error", f"Could not load model:\n{e}")
            self.model = None

        try:
            with open("class_names.pkl", "rb") as f:
                self.class_names = pickle.load(f)
        except Exception as e:
            messagebox.showerror("Class Names Load Error", f"Could not load class names:\n{e}")
            self.class_names = []

        self.selected_filepath = None
        self._create_widgets()

    def _create_widgets(self):
        tk.Label(self, text="Upload Currency Image", font=("Helvetica", 16, "bold"), bg="#f0f2f5").pack(pady=20)

        tk.Button(self, text="Choose Image", command=self._browse_files).pack()

        self.predict_button = tk.Button(self, text="Predict", command=self._analyze_file)
        self.predict_button.pack(pady=10)

        self.result_label = tk.Label(self, text="", bg="#f0f2f5", font=("Helvetica", 12))
        self.result_label.pack()

        self.image_label = tk.Label(self, bg="#f0f2f5")
        self.image_label.pack(pady=10)

    def _browse_files(self):
        filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if filepath:
            self.selected_filepath = filepath
            img = Image.open(filepath)
            img.thumbnail((200, 200))
            tk_image = ImageTk.PhotoImage(img)
            self.image_label.configure(image=tk_image)
            self.image_label.image = tk_image
            self.result_label.config(text=f"Selected: {os.path.basename(filepath)}")

    def _analyze_file(self):
        if not self.selected_filepath:
            messagebox.showwarning("No File", "Please choose an image file first.")
            return

        if self.model is None or not self.class_names:
            messagebox.showerror("Prediction Error", "Model or class names not loaded.")
            return

        input_array = normalize_image(self.selected_filepath)
        if input_array is None:
            return

        try:
            predictions = self.model.predict(input_array)
            top_3_indices = np.argsort(predictions[0])[::-1][:3]

            result_text = "Top Predictions:\n"
            for idx in top_3_indices:
                label = self.class_names[idx]
                confidence = predictions[0][idx] * 100
                result_text += f"- {label}: {confidence:.2f}%\n"

            self.result_label.config(text=result_text)
        except Exception as e:
            messagebox.showerror("Prediction Error", f"An error occurred during prediction:\n{e}")


if __name__ == "__main__":
    app = FileUploaderApp()
    app.mainloop()
