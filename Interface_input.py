import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import os
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
import re # Import the regular expression module

def parse_currency_label(label_string):
    """
    Parses a currency label string (e.g., "Philippines_1000 Peso_(1985-2019) Series")
    into country, value, and year range.
    """
    country = "N/A"
    value = "N/A"
    year_range = "N/A"
    series_info = "N/A" # To capture the "(YYYY-YYYY) Series" or similar

    # Regex to capture parts: Country_Value Unit_(YearRange) SeriesInfo
    # This regex is a bit more robust for different formats you might have in label_map.json
    match = re.match(r"([a-zA-Z\s]+)_(\d+\s*[a-zA-Z]+)(?:_|\s*)\(?(.*?)\)?\s*(.*)", label_string)
    # The regex breakdown:
    # ([a-zA-Z\s]+) : Captures the Country (e.g., "Philippines", "Brunei")
    # _             : Matches the underscore separator
    # (\d+\s*[a-zA-Z]+) : Captures the Value and Unit (e.g., "1000 Peso", "50 Baht")
    # (?:_| )* : Optionally matches underscore or space
    # \(?(.*?)\)?   : Optionally captures content within parentheses (e.g., "1985-2019", "2003")
    # \s*(.*)       : Captures any remaining series info (e.g., "Series", "Polymer Series (2003)")

    if match:
        country_part = match.group(1).strip()
        value_part = match.group(2).strip()
        year_part = match.group(3).strip()
        series_part = match.group(4).strip()

        country = country_part.split('_')[0].strip()

        value = value_part

        if year_part:
            year_match = re.search(r"(\d{4}(?:-\d{4})?)", year_part)
            if year_match:
                year_range = year_match.group(1)
            else:
                year_range = year_part
        elif series_part:
             year_match_in_series = re.search(r"(\d{4}(?:-\d{4})?)", series_part)
             if year_match_in_series:
                 year_range = year_match_in_series.group(1)

    return country, value, year_range


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
        self.geometry("600x650")
        self.configure(bg="#f0f2f5")

        try:
            self.model = load_model("currency_classifier.h5")
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
        # Title
        tk.Label(self, text="Upload Currency Image", font=("Helvetica", 18, "bold"), bg="#f0f2f5", fg="#333").pack(pady=20)

        # Frame for image selection and display
        image_frame = tk.Frame(self, bg="#f0f2f5", bd=2, relief="groove")
        image_frame.pack(pady=10, padx=20, fill="x")

        tk.Button(image_frame, text="Choose Image", command=self._browse_files,font=("Helvetica", 10), bg="#4CAF50", fg="white", padx=10, pady=5).pack(side="top", pady=10)

        self.image_label = tk.Label(image_frame, bg="#f0f2f5")
        self.image_label.pack(pady=10)

        self.selected_file_label = tk.Label(image_frame, text="No image selected", bg="#f0f2f5",font=("Helvetica", 10, "italic"))
        self.selected_file_label.pack(pady=5)

        # Prediction Button
        self.predict_button = tk.Button(self, text="Predict Currency Details", command=self._analyze_file,font=("Helvetica", 12, "bold"), bg="#2196F3", fg="white", padx=15, pady=8,cursor="hand2")
        self.predict_button.pack(pady=20)

        # Result Display Area
        result_frame = tk.LabelFrame(self, text="Detection Results", font=("Helvetica", 14, "bold"), bg="#f0f2f5", fg="#333", padx=15, pady=15)
        result_frame.pack(pady=10, padx=20, fill="both", expand=True)

        self.country_label = tk.Label(result_frame, text="Country: N/A", bg="#f0f2f5", font=("Helvetica", 12))
        self.country_label.pack(anchor="w", pady=5)

        self.value_label = tk.Label(result_frame, text="Value: N/A", bg="#f0f2f5", font=("Helvetica", 12))
        self.value_label.pack(anchor="w", pady=5)

        self.year_label = tk.Label(result_frame, text="Year: N/A", bg="#f0f2f5", font=("Helvetica", 12))
        self.year_label.pack(anchor="w", pady=5)

        self.full_label_label = tk.Label(result_frame, text="Detected Label: N/A", bg="#f0f2f5", font=("Helvetica", 10, "italic"), wraplength=450, justify="left")
        self.full_label_label.pack(anchor="w", pady=5)

        self.confidence_label = tk.Label(result_frame, text="Confidence: N/A", bg="#f0f2f5",font=("Helvetica", 12, "italic"))
        self.confidence_label.pack(anchor="w", pady=5)

    def _browse_files(self):
        filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if filepath:
            self.selected_filepath = filepath
            img = Image.open(filepath)
            img.thumbnail((200, 200))
            tk_image = ImageTk.PhotoImage(img)
            self.image_label.configure(image=tk_image)
            self.image_label.image = tk_image
            self.selected_file_label.config(text=f"Selected: {os.path.basename(filepath)}")

            # Reset results when a new image is selected
            self.country_label.config(text="Country: N/A")
            self.value_label.config(text="Value: N/A")
            self.year_label.config(text="Year: N/A")
            self.full_label_label.config(text="Detected Label: N/A")
            self.confidence_label.config(text="Confidence: N/A")

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

            # top_3_indices = np.argsort(predictions[0])[::-1][:3]
            #
            # result_text = "Top Predictions:\n"
            # for idx in top_3_indices:
            #     label = self.class_names[idx]
            #     confidence = predictions[0][idx] * 100
            #     result_text += f"- {label}: {confidence:.2f}%\n"
            #
            # self.result_label.config(text=result_text)

            # Get the index of the highest confidence prediction
            top_idx = np.argmax(predictions[0])
            predicted_label = self.class_names[top_idx]
            confidence = predictions[0][top_idx] * 100

            # Parse the predicted label
            country, value, year_range = parse_currency_label(predicted_label)

            # Update the UI labels
            self.country_label.config(text=f"Country: {country}")
            self.value_label.config(text=f"Value: {value}")
            self.year_label.config(text=f"Year: {year_range}")
            self.full_label_label.config(text=f"Detected Label: {predicted_label}")
            self.confidence_label.config(text=f"Confidence: {confidence:.2f}%")
        except Exception as e:
            messagebox.showerror("Prediction Error", f"An error occurred during prediction:\n{e}")
            self.country_label.config(text="Country: Error")
            self.value_label.config(text="Value: Error")
            self.year_label.config(text="Year: Error")
            self.full_label_label.config(text="Detected Label: Error")
            self.confidence_label.config(text="Confidence: Error")


if __name__ == "__main__":
    app = FileUploaderApp()
    app.mainloop()
