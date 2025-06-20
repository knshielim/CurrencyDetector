import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw
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
    match = re.match(r"([a-zA-Z\s]+)_([a-zA-Z0-9\s]+)(?:_|\s+)(.*)", label_string)
    if match:
        country_part = match.group(1).strip()
        value_part = match.group(2).strip()
        remaining_info = match.group(3).strip()
        country = country_part.split('_')[0].strip()
        value = value_part
        year_match = re.search(r"\((\d{4}(?:[-\s]*[\w\s]+)?)\)", remaining_info)
        if year_match:
            year_range = year_match.group(1).strip()
        else:
            year_match = re.search(r"\b(\d{4}(?:-\d{4})?)\b", remaining_info)
            if year_match:
                year_range = year_match.group(1)
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

# --- Function to create a rounded rectangle image programmatically ---
def create_rounded_rectangle_image(width, height, radius, color, file_name):
    img = Image.new("RGBA", (width, height), (255, 255, 255, 0)) # Transparent background (alpha=0)
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((0, 0, width, height), radius=radius, fill=color)
    img.save(file_name)
    print(f"Generated {file_name}")

# IMPORTANT: These lines are for generating images. Uncomment, run once, then re-comment.
# create_rounded_rectangle_image(250, 50, 15, "#4CAF50", "button_green_rounded.png") # Choose Image Button (Accent Green)
# create_rounded_rectangle_image(300, 60, 18, "#388E3C", "button_dark_green_rounded.png") # Predict Button (Darker Green)


# --- Main Application Class ---
class FileUploaderApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Currency Detector")
        self.geometry("700x750")
        self.configure(bg="#2C2C2C") # Dark gray background for the main window

        # Load model and class names
        try:
            self.model = load_model("currency_classifier.h5") # Ensure this path is correct
        except Exception as e:
            messagebox.showerror("Model Load Error", f"Could not load model. Please ensure 'currency_classifier.h5' is in the same directory.\nDetails: {e}")
            self.model = None

        try:
            with open("class_names.pkl", "rb") as f:
                self.class_names = pickle.load(f)
        except Exception as e:
            messagebox.showerror("Class Names Load Error", f"Could not load class names. Please ensure 'class_names.pkl' is in the same directory.\nDetails: {e}")
            self.class_names = []

        self.selected_filepath = None

        # Load button images with new names to reflect colors (ensure these files exist)
        try:
            self.btn_green_img = ImageTk.PhotoImage(Image.open("button_green_rounded.png"))
            self.btn_dark_green_img = ImageTk.PhotoImage(Image.open("button_dark_green_rounded.png")) # Changed filename
        except FileNotFoundError as e:
            messagebox.showerror("Image Load Error", f"Could not find button image files. Please ensure 'button_green_rounded.png' and 'button_dark_green_rounded.png' are in the same directory.\nDetails: {e}")
            self.btn_green_img = None
            self.btn_dark_green_img = None
        except Exception as e:
            messagebox.showerror("Image Load Error", f"An error occurred loading button images:\n{e}")
            self.btn_green_img = None
            self.btn_dark_green_img = None

        self._create_widgets()

    def _create_widgets(self):
        # Create a main_canvas to hold all content and allow scrolling
        main_canvas = tk.Canvas(self, bg="#2C2C2C", highlightthickness=0) # Match window bg, remove canvas border
        main_canvas.pack(side="left", fill="both", expand=True)

        # Create a scrollbar for the canvas
        scrollbar = tk.Scrollbar(self, orient="vertical", command=main_canvas.yview) # Removed troughcolor
        scrollbar.pack(side="right", fill="y")

        # Configure the canvas to use the scrollbar
        main_canvas.configure(yscrollcommand=scrollbar.set)

        # Create a frame inside the canvas to put all actual widgets
        # This frame will be the scrollable content
        self.scrollable_frame = tk.Frame(main_canvas, bg="#2C2C2C") # Match window bg

        # Add the scrollable_frame to a window on the canvas
        # IMPORTANT: Store the ID of the window item on the canvas
        self.canvas_frame_id = main_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        # Update the scrollregion of the canvas whenever the scrollable_frame's size changes
        self.scrollable_frame.bind("<Configure>", lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all")))

        # NEW: Bind to the main_canvas's <Configure> event to resize the internal frame
        # This makes the content expand horizontally when the canvas (and thus the window) resizes
        def _on_canvas_resize(event):
            canvas_width = event.width
            # Update the width of the window item to match the canvas width
            main_canvas.itemconfig(self.canvas_frame_id, width=canvas_width)

        main_canvas.bind("<Configure>", _on_canvas_resize)


        # --- Now, place all original content inside self.scrollable_frame ---

        # --- Header Frame (inside scrollable_frame) ---
        header_frame = tk.Frame(self.scrollable_frame, bg="#1C1C1C", pady=10) # Even darker gray for header
        header_frame.pack(fill="x")
        tk.Label(header_frame, text="💰 Currency Detector 💰", font=("Arial", 24, "bold"), bg="#1C1C1C", fg="white").pack(pady=5)

        # --- Main Content Frame (inside scrollable_frame) ---
        main_content_frame = tk.Frame(self.scrollable_frame, bg="#2C2C2C", padx=20, pady=20) # Match main window bg
        main_content_frame.pack(fill="both", expand=True)

        # --- Image Selection and Display Area (inside main_content_frame) ---
        image_selection_frame = tk.LabelFrame(main_content_frame, text="Image Selection",
                                              font=("Arial", 14, "bold"), bg="#3A3A3A", fg="#E0E0E0", # Darker gray content frame, light text
                                              padx=15, pady=15, bd=2, relief="groove")
        image_selection_frame.pack(pady=10, fill="x")

        # Choose Image Button (inside image_selection_frame)
        self.choose_image_button = tk.Button(image_selection_frame,
                                             text="📂 Choose Image",
                                             command=self._browse_files,
                                             font=("Arial", 12, "bold"),
                                             fg="white", # Text color
                                             cursor="hand2",
                                             relief="flat", # No native button border
                                             bd=0,
                                             bg=image_selection_frame['bg'], # Set background to match parent frame's bg
                                             activebackground=image_selection_frame['bg']) # Set active background to match parent frame's bg
        if self.btn_green_img:
            self.choose_image_button.config(image=self.btn_green_img, compound="center") # Place text on image
        else:
            self.choose_image_button.config(bg="#4CAF50") # Fallback color
        self.choose_image_button.pack(pady=10)

        # Image Display Label (inside image_selection_frame)
        self.image_label = tk.Label(image_selection_frame, bg="#424242", bd=1, relief="solid") # Slightly lighter dark gray for image area
        self.image_label.pack(pady=10, padx=10, fill="both", expand=True)
        # Placeholder image - now dark gray
        self.placeholder_img = Image.new('RGB', (200, 200), color = '#606060') # Darker grey placeholder
        self.placeholder_tk = ImageTk.PhotoImage(self.placeholder_img)
        self.image_label.configure(image=self.placeholder_tk)
        self.image_label.image = self.placeholder_tk # Keep a reference!

        # Selected File Name Label (inside image_selection_frame)
        self.selected_file_label = tk.Label(image_selection_frame, text="No image selected", bg="#3A3A3A", # Match content frame bg
                                            font=("Arial", 10, "italic"), fg="#C0C0C0") # Lighter gray for file name
        self.selected_file_label.pack(pady=5)

        # --- Predict Button (inside main_content_frame) ---
        self.predict_button = tk.Button(main_content_frame,
                                        text="🚀 Predict Currency Details",
                                        command=self._analyze_file,
                                        font=("Arial", 14, "bold"),
                                        fg="white", # Text color
                                        cursor="hand2",
                                        relief="flat", # No native button border
                                        bd=0,
                                        bg=main_content_frame['bg'], # Set background to match parent frame's bg
                                        activebackground=main_content_frame['bg']) # Set active background to match parent frame's bg
        if self.btn_dark_green_img: # Use the darker green image
            self.predict_button.config(image=self.btn_dark_green_img, compound="center")
        else:
            self.predict_button.config(bg="#388E3C") # Fallback color
        self.predict_button.pack(pady=20)

        # --- Result Display Area (inside main_content_frame) ---
        result_frame = tk.LabelFrame(main_content_frame, text="🔍 Detection Results",
                                     font=("Arial", 14, "bold"), bg="#3A3A3A", fg="#E0E0E0", # Darker gray content frame, light text
                                     padx=15, pady=15, bd=2, relief="groove")
        result_frame.pack(pady=10, fill="both", expand=True)

        # --- Use ttk.Treeview for the table ---
        style = ttk.Style()
        style.theme_use("clam") # "clam", "alt", "default", "classic"
        style.configure("Treeview",
                        background="#333333", # Row background
                        foreground="#E0E0E0", # Text color
                        fieldbackground="#333333", # Background of the entire treeview widget
                        bordercolor="#555555",
                        lightcolor="#555555",
                        darkcolor="#2C2C2C",
                        rowheight=25)
        style.map("Treeview", background=[('selected', '#4A6984')]) # Selection color

        style.configure("Treeview.Heading",
                        background="#424242", # Header background
                        foreground="white", # Header text color
                        font=("Arial", 11, "bold"),
                        relief="raised",
                        bordercolor="#555555")
        style.map("Treeview.Heading", background=[('active', '#555555')])


        self.result_tree = ttk.Treeview(result_frame, columns=("Rank", "Country", "Value", "Year", "Confidence"), show="headings")
        self.result_tree.pack(side="left", fill="both", expand=True)

        # Define column headings and properties
        self.result_tree.heading("Rank", text="#", anchor="center")
        self.result_tree.heading("Country", text="Country", anchor="w")
        self.result_tree.heading("Value", text="Value", anchor="w")
        self.result_tree.heading("Year", text="Year", anchor="w")
        self.result_tree.heading("Confidence", text="Confidence", anchor="center")

        self.result_tree.column("Rank", width=40, anchor="center")
        self.result_tree.column("Country", width=120, anchor="w")
        self.result_tree.column("Value", width=100, anchor="w")
        self.result_tree.column("Year", width=90, anchor="w")
        self.result_tree.column("Confidence", width=100, anchor="center")

        # Scrollbar for the Treeview
        self.tree_scrollbar = ttk.Scrollbar(result_frame, orient="vertical", command=self.result_tree.yview) # REMOVED troughcolor
        self.tree_scrollbar.pack(side="right", fill="y")
        self.result_tree.config(yscrollcommand=self.tree_scrollbar.set)

        # Text widget for full label details (below the table)
        self.full_details_label = tk.Label(result_frame, text="Full Label Details:",
                                           font=("Arial", 12, "bold"), bg="#3A3A3A", fg="#E0E0E0", anchor="w")
        self.full_details_label.pack(pady=(10, 0), fill="x")

        self.full_details_text = tk.Text(result_frame, wrap="word", height=8,
                                         font=("Consolas", 10), bg="#333333", fg="#E0E0E0",
                                         relief="flat", bd=0, state="disabled", padx=5, pady=5)
        self.full_details_text.pack(pady=(0, 10), fill="x", expand=False) # Not expanding vertically, only horizontally

        # Initial message for the Treeview and Text widget
        self._clear_results()
        self.result_tree.insert("", "end", values=("", "Upload an image and", "click 'Predict'", "to see", "details."))


    def _clear_results(self):
        # Clear previous table data
        for item in self.result_tree.get_children():
            self.result_tree.delete(item)
        # Clear previous full details text
        self.full_details_text.config(state="normal")
        self.full_details_text.delete(1.0, tk.END)
        self.full_details_text.config(state="disabled")


    def _browse_files(self):
        filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if filepath:
            self.selected_filepath = filepath
            img = Image.open(filepath)
            img.thumbnail((200, 200), Image.Resampling.LANCZOS)
            tk_image = ImageTk.PhotoImage(img)
            self.image_label.configure(image=tk_image)
            self.image_label.image = tk_image
            self.selected_file_label.config(text=f"Selected: {os.path.basename(filepath)}")

            self._clear_results()
            self.result_tree.insert("", "end", values=("", "", "Click 'Predict'", "", ""))
            self.full_details_text.config(state="normal")
            self.full_details_text.insert(tk.END, "Click 'Predict' to analyze the selected image.")
            self.full_details_text.config(state="disabled")


    def _analyze_file(self):
        if not self.selected_filepath:
            messagebox.showwarning("No File", "Please choose an image file first.")
            return

        if self.model is None or not self.class_names:
            messagebox.showerror("Prediction Error", "Model or class names not loaded. Please check the console for details.")
            return

        self._clear_results()
        self.result_tree.insert("", "end", values=("", "", "Analyzing image...", "", ""))
        self.full_details_text.config(state="normal")
        self.full_details_text.delete(1.0, tk.END)
        self.full_details_text.insert(tk.END, "Analyzing image... Please wait.")
        self.full_details_text.config(state="disabled")
        self.update_idletasks() # Update GUI to show "Analyzing image..." message

        input_array = normalize_image(self.selected_filepath)
        if input_array is None:
            self.full_details_text.config(state="normal")
            self.full_details_text.delete(1.0, tk.END)
            self.full_details_text.insert(tk.END, "Image preprocessing failed.")
            self.full_details_text.config(state="disabled")
            self._clear_results()
            return

        try:
            predictions = self.model.predict(input_array, verbose=0)
            top_3_indices = np.argsort(predictions[0])[::-1][:3]

            self._clear_results() # Clear "Analyzing image..." message

            raw_label_details_list = [] # List to store full raw labels

            for i, idx in enumerate(top_3_indices):
                predicted_label = self.class_names[idx] # This is the full raw label
                confidence = predictions[0][idx] * 100

                country, value, year_range = parse_currency_label(predicted_label)

                # Insert data into the Treeview
                self.result_tree.insert("", "end", values=(
                    i + 1,
                    country,
                    value,
                    year_range,
                    f"{confidence:.2f}%"
                ))

                # Add the full raw label to the list
                raw_label_details_list.append(f"- {predicted_label}")

            # Populate the full details text widget
            self.full_details_text.config(state="normal")
            self.full_details_text.insert(tk.END, "\n".join(raw_label_details_list))
            self.full_details_text.config(state="disabled")


        except Exception as e:
            messagebox.showerror("Prediction Error", f"An error occurred during prediction:\n{e}")
            print(f"DEBUG: Prediction Exception: {e}")
            self._clear_results() # Clear any pending messages
            self.result_tree.insert("", "end", values=("", "", "Prediction Failed", "", ""))
            self.full_details_text.config(state="normal")
            self.full_details_text.delete(1.0, tk.END)
            self.full_details_text.insert(tk.END, f"Prediction Error: Could not analyze image.\nDetails: {e}")
            self.full_details_text.config(state="disabled")


if __name__ == "__main__":
    app = FileUploaderApp()
    app.mainloop()