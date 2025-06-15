import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import os
import pickle
from tensorflow.keras.models import load_model
from normalize import normalize_image

class FileUploaderApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Currency Detector")
        self.geometry("600x500")
        self.configure(bg="#f0f2f5")

        self.model = load_model("currency_classifier_final_model.h5")
        with open("class_names.pkl", "rb") as f:
            self.class_names = pickle.load(f)

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
            self.result_label.config(text=f"Selected: {os.path.basename(filepath)}")

    def _analyze_file(self):
        if not self.selected_filepath:
            messagebox.showwarning("No File", "Please choose an image file first.")
            return

        input_array = normalize_image(self.selected_filepath)

        if input_array is None:
            messagebox.showerror("Error", "Failed to process the image.")
            return

        # Convert normalized array (float32, 0–1) back to PIL Image for display
        display_image = (input_array[0] * 255).astype(np.uint8)
        display_image = Image.fromarray(display_image)

        # Resize for UI display (misalnya 200x200 agar pas)
        display_image = display_image.resize((200, 200))
        tk_image = ImageTk.PhotoImage(display_image)

        self.image_label.configure(image=tk_image)
        self.image_label.image = tk_image  # prevent garbage collection

        # Predict
        predictions = self.model.predict(input_array)
        top_3_indices = np.argsort(predictions[0])[::-1][:3]

        result_text = "Top Predictions:\n"
        for idx in top_3_indices:
            label = self.class_names[idx]
            confidence = predictions[0][idx] * 100
            result_text += f"- {label}: {confidence:.2f}%\n"

        self.result_label.config(text=result_text)

"""
    def _analyze_file(self):
        if not self.selected_filepath:
            messagebox.showwarning("No File", "Please choose an image file first.")
            return

        input_array = normalize_image(self.selected_filepath)
        if input_array is None:
            messagebox.showerror("Error", "Failed to process the image.")
            return

        predictions = self.model.predict(input_array)
        top_3_indices = np.argsort(predictions[0])[::-1][:3]

        result_text = "Top Predictions:\n"
        for idx in top_3_indices:
            label = self.class_names[idx]
            confidence = predictions[0][idx] * 100
            result_text += f"- {label}: {confidence:.2f}%\n"

        self.result_label.config(text=result_text)
"""
if __name__ == "__main__":
    app = FileUploaderApp()
    app.mainloop()
