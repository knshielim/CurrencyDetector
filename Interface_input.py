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
    print(f"\n--- DEBUG: Parsing Label: '{label_string}' ---")
    print(f"DEBUG: Raw Label String repr: {repr(label_string)} (Length: {len(label_string)})")
    country = "N/A"
    value = "N/A"
    year_range = "N/A"

    # UPDATED Main Regex:
    # - ([a-zA-Z\s]+) : Captures country name (Group 1)
    # - _ : Matches literal underscore
    # - ([a-zA-Z0-9\s]+) : Greedy capture for the value/denomination part. (Group 2)
    # - (?:_|\s+) : Non-capturing group for the separator: an underscore OR one or more spaces.
    # - (.*) : NEW: Captures everything else as 'remaining_info'. (Group 3)
    #   This makes the main regex more flexible, allowing the subsequent year extraction logic
    #   to find the year within this broader string.
    match = re.match(r"([a-zA-Z\s]+)_([a-zA-Z0-9\s]+)(?:_|\s+)(.*)", label_string)

    if match:
        country_part = match.group(1).strip()
        value_part = match.group(2).strip()
        remaining_info = match.group(3).strip()

        print(f"DEBUG: Main Regex captured groups:")
        print(f"  Group 1 (country_part): '{country_part}'")
        print(f"  Group 2 (value_part):   '{value_part}'")
        print(f"  Group 3 (remaining_info): '{remaining_info}'")

        country = country_part.split('_')[0].strip()
        value = value_part

        # Robustly extract year from remaining_info (this part remains the same)
        # 1. Try to find year in parentheses, allowing for "Present" or similar text
        year_match = re.search(r"\((\d{4}(?:[-\s]*[\w\s]+)?)\)", remaining_info)
        if year_match:
            year_range = year_match.group(1).strip()
            print(f"DEBUG: Year matched from parentheses in remaining_info: '{year_range}'")
        else:
            # 2. If not in parentheses, look for a standalone 4-digit or 4-digit-4-digit number
            year_match = re.search(r"\b(\d{4}(?:-\d{4})?)\b", remaining_info)
            if year_match:
                year_range = year_match.group(1)
                print(f"DEBUG: Year matched standalone in remaining_info: '{year_range}'")
            else:
                print(f"DEBUG: No year found in remaining_info.")

    else:
        print(f"DEBUG: FATAL: Main Regex failed for label: '{label_string}'")

    print(f"DEBUG: Final Parsed Result - Country: '{country}', Value: '{value}', Year: '{year_range}'")
    print("---------------------------------------")
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
        tk.Label(self, text="Upload Currency Image", font=("Helvetica", 18, "bold"), bg="#f0f2f5", fg="#333").pack(
            pady=20)

        # Frame for image selection and display
        image_frame = tk.Frame(self, bg="#f0f2f5", bd=2, relief="groove")
        image_frame.pack(pady=10, padx=20, fill="x")

        tk.Button(image_frame, text="Choose Image", command=self._browse_files, font=("Helvetica", 10), bg="#4CAF50",
                  fg="white", padx=10, pady=5).pack(side="top", pady=10)

        self.image_label = tk.Label(image_frame, bg="#f0f2f5")
        self.image_label.pack(pady=10)

        self.selected_file_label = tk.Label(image_frame, text="No image selected", bg="#f0f2f5",
                                            font=("Helvetica", 10, "italic"))
        self.selected_file_label.pack(pady=5)

        # Prediction Button
        self.predict_button = tk.Button(self, text="Predict Currency Details", command=self._analyze_file,
                                        font=("Helvetica", 12, "bold"), bg="#2196F3", fg="white", padx=15, pady=8,
                                        cursor="hand2")
        self.predict_button.pack(pady=20)

        # Result Display Area
        result_frame = tk.LabelFrame(self, text="Detection Results", font=("Helvetica", 14, "bold"), bg="#f0f2f5",
                                     fg="#333", padx=15, pady=15)
        result_frame.pack(pady=10, padx=20, fill="both", expand=True)

        # NEW: Frame for Text widget and Scrollbar
        text_scroll_frame = tk.Frame(result_frame, bg="#f0f2f5")
        text_scroll_frame.pack(fill="both", expand=True)

        self.results_text_widget = tk.Text(text_scroll_frame, wrap="word", height=8, font=("Helvetica", 11),
                                           bg="#ffffff", fg="#333", relief="solid", bd=1,
                                           state="disabled")  # Set to disabled by default
        self.results_text_widget.pack(side="left", fill="both", expand=True)

        self.results_scrollbar = tk.Scrollbar(text_scroll_frame, command=self.results_text_widget.yview)
        self.results_scrollbar.pack(side="right", fill="y")

        self.results_text_widget.config(yscrollcommand=self.results_scrollbar.set)

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

            # Reset results in the Text widget (CORRECTED PART)
            self.results_text_widget.config(state="normal")  # Enable editing
            self.results_text_widget.delete(1.0, tk.END)  # Clear content
            self.results_text_widget.insert(tk.END, "Top Predictions: N/A")
            self.results_text_widget.config(state="disabled")  # Disable editing

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
            #
            # result_text = "Top Predictions:\n"
            # for idx in top_3_indices:
            #     label = self.class_names[idx]
            #     confidence = predictions[0][idx] * 100
            #     result_text += f"- {label}: {confidence:.2f}%\n"
            #
            # self.result_label.config(text=result_text)

            results_text = "Top 3 Predictions:\n"
            for i, idx in enumerate(top_3_indices):
                predicted_label = self.class_names[idx]
                confidence = predictions[0][idx] * 100

                # Parse the predicted label
                country, value, year_range = parse_currency_label(predicted_label)

                results_text += (f"{i + 1}. Country: {country}\n"
                                 f"   Value: {value}\n"
                                 f"   Year: {year_range}\n"
                                 f"   Confidence: {confidence:.2f}%\n"
                                 f"   (Label: {predicted_label})\n")
                if i < 2:  # Add an extra newline between predictions for readability
                    results_text += "\n"

            # Update the UI Text widget with all top 3 results (CORRECTED PART)
            self.results_text_widget.config(state="normal")  # Enable editing
            self.results_text_widget.delete(1.0, tk.END)  # Clear content
            self.results_text_widget.insert(tk.END, results_text)
            self.results_text_widget.config(state="disabled")  # Disable editing

        except Exception as e:
            messagebox.showerror("Prediction Error", f"An error occurred during prediction:\n{e}")
            print(f"DEBUG: Prediction Exception: {e}") # Debug print
            # Update the UI Text widget with error message (CORRECTED PART)
            self.results_text_widget.config(state="normal")  # Enable editing
            self.results_text_widget.delete(1.0, tk.END)  # Clear content
            self.results_text_widget.insert(tk.END, f"Prediction Error: Could not analyze image.\nDetails: {e}")
            self.results_text_widget.config(state="disabled")  # Disable editing

if __name__ == "__main__":
    app = FileUploaderApp()
    app.mainloop()
