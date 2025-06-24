import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import os
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
import re
import threading
from datetime import datetime


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
        img = img / 255.0
        input_array = tf.expand_dims(img, axis=0)
        return input_array
    except Exception as e:
        return None, str(e)


def create_rounded_rectangle_image(width, height, radius, color, file_name):
    """Create rounded rectangle button images programmatically."""
    img = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((0, 0, width, height), radius=radius, fill=color)
    img.save(file_name)
    print(f"Generated {file_name}")


class FileUploaderApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Currency Detector v2.0")
        self.geometry("800x850")
        self.configure(bg="#2C2C2C")

        # Initialize variables
        self.selected_filepath = None
        self.prediction_history = []
        self.is_predicting = False
        self.status_bar = None  # Initialize status_bar to None
        self.last_results = []  # Store last prediction results

        # Create status bar first (so _update_status works)
        self._create_status_bar()

        # Create button images if they don't exist
        self._create_button_images()

        # Load model and class names
        self._load_model_and_classes()

        # Create the UI
        self._create_widgets()

    def _create_button_images(self):
        """Create button images if they don't exist."""
        button_files = ["button_green_rounded.png", "button_dark_green_rounded.png"]
        if not all(os.path.exists(f) for f in button_files):
            try:
                create_rounded_rectangle_image(250, 50, 15, "#4CAF50", "button_green_rounded.png")
                create_rounded_rectangle_image(300, 60, 18, "#388E3C", "button_dark_green_rounded.png")
            except Exception as e:
                print(f"Warning: Could not create button images: {e}")

    def _load_model_and_classes(self):
        """Load the TensorFlow model and class names."""
        try:
            self.model = load_model("currency_classifier.h5")
            self._update_status("Model loaded successfully")
        except Exception as e:
            messagebox.showerror("Model Load Error",
                                 f"Could not load model. Please ensure 'currency_classifier.h5' is in the same directory.\nDetails: {e}")
            self.model = None
            self._update_status("Model loading failed")

        try:
            with open("class_names.pkl", "rb") as f:
                self.class_names = pickle.load(f)
            self._update_status(f"Loaded {len(self.class_names)} currency classes")
        except Exception as e:
            messagebox.showerror("Class Names Load Error",
                                 f"Could not load class names. Please ensure 'class_names.pkl' is in the same directory.\nDetails: {e}")
            self.class_names = []
            self._update_status("Class names loading failed")

        # Load button images
        try:
            self.btn_green_img = ImageTk.PhotoImage(Image.open("button_green_rounded.png"))
            self.btn_dark_green_img = ImageTk.PhotoImage(Image.open("button_dark_green_rounded.png"))
        except Exception as e:
            print(f"Warning: Could not load button images: {e}")
            self.btn_green_img = None
            self.btn_dark_green_img = None

    def _create_widgets(self):
        """Create the main UI widgets."""
        # Main canvas with scrolling
        main_canvas = tk.Canvas(self, bg="#2C2C2C", highlightthickness=0)
        main_canvas.pack(side="left", fill="both", expand=True)

        scrollbar = tk.Scrollbar(self, orient="vertical", command=main_canvas.yview)
        scrollbar.pack(side="right", fill="y")
        main_canvas.configure(yscrollcommand=scrollbar.set)

        self.scrollable_frame = tk.Frame(main_canvas, bg="#2C2C2C")
        self.canvas_frame_id = main_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        self.scrollable_frame.bind("<Configure>",
                                   lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all")))

        def _on_canvas_resize(event):
            canvas_width = event.width
            main_canvas.itemconfig(self.canvas_frame_id, width=canvas_width)

        main_canvas.bind("<Configure>", _on_canvas_resize)

        # Header
        header_frame = tk.Frame(self.scrollable_frame, bg="#1C1C1C", pady=10)
        header_frame.pack(fill="x")

        title_label = tk.Label(header_frame, text="💰 Currency Detector v2.0 💰",
                               font=("Arial", 24, "bold"), bg="#1C1C1C", fg="white")
        title_label.pack(pady=5)

        subtitle_label = tk.Label(header_frame, text="AI-Powered Currency Recognition",
                                  font=("Arial", 12), bg="#1C1C1C", fg="#B0B0B0")
        subtitle_label.pack()

        # Main content
        main_content_frame = tk.Frame(self.scrollable_frame, bg="#2C2C2C", padx=20, pady=20)
        main_content_frame.pack(fill="both", expand=True)

        # Image selection frame
        self._create_image_selection_frame(main_content_frame)

        # Control buttons frame
        self._create_control_buttons_frame(main_content_frame)

        # Results frame
        self._create_results_frame(main_content_frame)

        # History frame (Now with updated styling)
        self._create_history_frame(main_content_frame)

    def _create_image_selection_frame(self, parent):
        """Create the image selection and display area."""
        image_frame = tk.LabelFrame(parent, text="Image Selection",
                                    font=("Arial", 14, "bold"), bg="#3A3A3A", fg="#E0E0E0",
                                    padx=15, pady=15, bd=2, relief="groove")
        image_frame.pack(pady=10, fill="x")

        # Choose image button
        self.choose_image_button = tk.Button(image_frame, text="📂 Choose Image",
                                             command=self._browse_files, font=("Arial", 12, "bold"),
                                             fg="white", cursor="hand2", relief="flat", bd=0,
                                             bg=image_frame['bg'], activebackground=image_frame['bg'])

        if self.btn_green_img:
            self.choose_image_button.config(image=self.btn_green_img, compound="center")
        else:
            self.choose_image_button.config(bg="#4CAF50")

        self.choose_image_button.pack(pady=10)

        # Image display
        self.image_label = tk.Label(image_frame, bg="#424242", bd=2, relief="solid")
        self.image_label.pack(pady=10, padx=10, fill="both", expand=True)

        # Placeholder image
        self.placeholder_img = Image.new('RGB', (300, 200), color='#606060')
        placeholder_draw = ImageDraw.Draw(self.placeholder_img)
        placeholder_draw.text((150, 100), "No Image Selected", fill="white", anchor="mm")
        self.placeholder_tk = ImageTk.PhotoImage(self.placeholder_img)
        self.image_label.configure(image=self.placeholder_tk)
        self.image_label.image = self.placeholder_tk

        # File info
        self.selected_file_label = tk.Label(image_frame, text="No image selected",
                                            bg="#3A3A3A", font=("Arial", 10, "italic"), fg="#C0C0C0")
        self.selected_file_label.pack(pady=5)

    def _create_control_buttons_frame(self, parent):
        """Create the control buttons and tools."""
        control_frame = tk.Frame(parent, bg="#2C2C2C")
        control_frame.pack(pady=10, fill="x")

        # Left side - Main predict button
        left_frame = tk.Frame(control_frame, bg="#2C2C2C")
        left_frame.pack(side="left")

        self.predict_button = tk.Button(left_frame, text="🚀 Predict Currency Details",
                                        command=self._analyze_file_threaded, font=("Arial", 14, "bold"),
                                        fg="white", cursor="hand2", relief="flat", bd=0,
                                        bg=parent['bg'], activebackground=parent['bg'])

        if self.btn_dark_green_img:
            self.predict_button.config(image=self.btn_dark_green_img, compound="center")
        else:
            self.predict_button.config(bg="#388E3C")

        self.predict_button.pack()

        # Right side - Tools and settings
        right_frame = tk.Frame(control_frame, bg="#2C2C2C")
        right_frame.pack(side="right", fill="x", expand=True, padx=(20, 0))

        # Confidence threshold
        conf_frame = tk.Frame(right_frame, bg="#2C2C2C")
        conf_frame.pack(anchor="e", pady=2)

        tk.Label(conf_frame, text="Confidence Threshold:", bg="#2C2C2C", fg="#E0E0E0",
                 font=("Arial", 10)).pack(side="left")

        self.confidence_var = tk.DoubleVar(value=0.1)
        self.confidence_scale = tk.Scale(conf_frame, from_=0.01, to=0.9, resolution=0.01,
                                         orient="horizontal", variable=self.confidence_var,
                                         bg="#3A3A3A", fg="#E0E0E0", highlightthickness=0,
                                         length=150, font=("Arial", 8))
        self.confidence_scale.pack(side="left", padx=5)

        # Show top N results
        topn_frame = tk.Frame(right_frame, bg="#2C2C2C")
        topn_frame.pack(anchor="e", pady=2)

        tk.Label(topn_frame, text="Show Top:", bg="#2C2C2C", fg="#E0E0E0",
                 font=("Arial", 10)).pack(side="left")

        self.topn_var = tk.IntVar(value=3)
        topn_spinbox = tk.Spinbox(topn_frame, from_=1, to=10, width=3, textvariable=self.topn_var,
                                  bg="#3A3A3A", fg="#E0E0E0", buttonbackground="#4A4A4A",
                                  font=("Arial", 9))
        topn_spinbox.pack(side="left", padx=5)

        # Save results button
        save_frame = tk.Frame(right_frame, bg="#2C2C2C")
        save_frame.pack(anchor="e", pady=2)

        self.save_button = tk.Button(save_frame, text="💾 Save Results", command=self._save_results,
                                     font=("Arial", 10), bg="#2196F3", fg="white",
                                     cursor="hand2", relief="flat", bd=0, padx=10)
        self.save_button.pack()

    def _create_results_frame(self, parent):
        """Create the results display area."""
        result_frame = tk.LabelFrame(parent, text="🔍 Detection Results",
                                     font=("Arial", 14, "bold"), bg="#3A3A3A", fg="#E0E0E0",
                                     padx=15, pady=15, bd=2, relief="groove")
        result_frame.pack(pady=10, fill="both", expand=True)

        # Configure treeview style
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", background="#333333", foreground="#E0E0E0",
                        fieldbackground="#333333", bordercolor="#555555",
                        lightcolor="#555555", darkcolor="#2C2C2C", rowheight=25)
        style.map("Treeview", background=[('selected', '#4A6984')])
        style.configure("Treeview.Heading", background="#424242", foreground="white",
                        font=("Arial", 11, "bold"), relief="raised", bordercolor="#555555")
        style.map("Treeview.Heading", background=[('active', '#555555')])

        # Results table
        self.result_tree = ttk.Treeview(result_frame,
                                        columns=("Rank", "Country", "Value", "Year", "Confidence"),
                                        show="headings", height=8)
        self.result_tree.pack(side="left", fill="both", expand=True)

        # Column configuration
        columns = [
            ("Rank", 40, "center"),
            ("Country", 120, "w"),
            ("Value", 100, "w"),
            ("Year", 90, "w"),
            ("Confidence", 100, "center")
        ]

        for col, width, anchor in columns:
            self.result_tree.heading(col, text=col, anchor=anchor)
            self.result_tree.column(col, width=width, anchor=anchor)

        # Scrollbar for treeview
        tree_scrollbar = ttk.Scrollbar(result_frame, orient="vertical", command=self.result_tree.yview)
        tree_scrollbar.pack(side="right", fill="y")
        self.result_tree.config(yscrollcommand=tree_scrollbar.set)

        # Full details section
        details_label = tk.Label(result_frame, text="Full Label Details:",
                                 font=("Arial", 12, "bold"), bg="#3A3A3A", fg="#E0E0E0", anchor="w")
        details_label.pack(pady=(10, 0), fill="x")

        self.full_details_text = tk.Text(result_frame, wrap="word", height=6,
                                         font=("Consolas", 10), bg="#333333", fg="#E0E0E0",
                                         relief="flat", bd=0, state="disabled", padx=5, pady=5)
        self.full_details_text.pack(pady=(0, 10), fill="x")

        self._clear_results()

    def _create_history_frame(self, parent):
        """Create an enhanced statistics and comparison frame with cleaner aesthetics."""
        # Use a regular Frame with a border for the "rounded corner" effect if using a custom image,
        # or simply rely on padding/colors for a cleaner appearance without explicit rounded corners
        # as native Tkinter LabelFrame does not support border-radius.

        # For a clean look, we can use a Frame and draw a rounded rectangle on a canvas behind it,
        # but for simplicity and performance, consistent padding, background, and fonts are often best.

        # We'll stick to a LabelFrame for its title capability, but enhance its appearance.
        stats_frame = tk.LabelFrame(parent, text="📊 Currency Statistics & Comparison",
                                    font=("Arial", 14, "bold"), bg="#3A3A3A", fg="#E0E0E0",
                                    padx=15, pady=15, bd=1, relief="solid") # Changed relief to solid, reduced bd
        stats_frame.pack(pady=10, fill="x", padx=5) # Added some horizontal padding for the frame

        # Create two columns for better separation
        content_frame = tk.Frame(stats_frame, bg="#3A3A3A")
        content_frame.pack(fill="both", expand=True, pady=5) # Add padding for separation

        left_stats = tk.Frame(content_frame, bg="#3A3A3A")
        left_stats.pack(side="left", fill="both", expand=True, padx=(0, 10)) # Added right padding

        right_stats = tk.Frame(content_frame, bg="#3A3A3A")
        right_stats.pack(side="right", fill="both", expand=True, padx=(10, 0)) # Added left padding

        # Left column - Currency Information
        tk.Label(left_stats, text="💰 Best Match Details", font=("Arial", 12, "bold"),
                 bg="#3A3A3A", fg="#66BB6A", anchor="w", pady=5).pack(fill="x") # Brighter green, padding

        self.currency_info_text = tk.Text(left_stats, height=8, wrap="word",
                                          font=("Segoe UI", 10), bg="#2B2B2B", fg="#F0F0F0", # Changed font, darker background, lighter text
                                          relief="flat", bd=0, state="disabled", padx=10, pady=10,
                                          insertbackground="white", selectbackground="#4A6984") # Added padding, select color
        self.currency_info_text.pack(pady=(0, 10), fill="both", expand=True)

        # Right column - Comparison with other currencies
        tk.Label(right_stats, text="🌍 Alternative Matches", font=("Arial", 12, "bold"),
                 bg="#3A3A3A", fg="#64B5F6", anchor="w", pady=5).pack(fill="x") # Brighter blue, padding

        self.comparison_text = tk.Text(right_stats, height=8, wrap="word",
                                       font=("Segoe UI", 10), bg="#2B2B2B", fg="#F0F0F0", # Changed font, darker background, lighter text
                                       relief="flat", bd=0, state="disabled", padx=10, pady=10,
                                       insertbackground="white", selectbackground="#4A6984") # Added padding, select color
        self.comparison_text.pack(pady=(0, 10), fill="both", expand=True)

        # Bottom row - Quick stats (aligned to center)
        stats_bottom = tk.Frame(stats_frame, bg="#3A3A3A", pady=5) # Added vertical padding
        stats_bottom.pack(fill="x")

        self.stats_label = tk.Label(stats_bottom, text="Ready for prediction...",
                                    font=("Segoe UI", 10, "italic"), bg="#3A3A3A", fg="#B0B0B0") # Changed font, italic
        self.stats_label.pack(anchor="center") # Centered

    def _create_status_bar(self):
        """Create the status bar at the bottom."""
        self.status_bar = tk.Label(self, text="Ready", bd=1, relief="sunken", anchor="w",
                                   bg="#1C1C1C", fg="#C0C0C0", font=("Arial", 9))
        self.status_bar.pack(side="bottom", fill="x")

    def _update_status(self, message):
        """Update the status bar message."""
        if self.status_bar is not None:  # Check if status_bar exists
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.status_bar.config(text=f"[{timestamp}] {message}")
            self.update_idletasks()
        else:
            print(f"Status: {message}")  # Fallback to console if status_bar not ready

    def _clear_results(self):
        """Clear the results display."""
        for item in self.result_tree.get_children():
            self.result_tree.delete(item)

        self.full_details_text.config(state="normal")
        self.full_details_text.delete(1.0, tk.END)
        self.full_details_text.config(state="disabled")

    def _clear_all_results(self):
        """Clear all results and reset interface."""
        self._clear_results()
        self.prediction_history.clear()

        # Clear currency info
        self.currency_info_text.config(state="normal")
        self.currency_info_text.delete(1.0, tk.END)
        self.currency_info_text.config(state="disabled")

        # Clear comparison info
        self.comparison_text.config(state="normal")
        self.comparison_text.delete(1.0, tk.END)
        self.comparison_text.config(state="disabled")

        # Reset stats
        self.stats_label.config(text="Ready for prediction...")

        self.result_tree.insert("", "end", values=("", "Upload an image and", "click 'Predict'", "to see", "details."))
        self._update_status("Interface reset")

    def _save_results(self):
        """Save prediction results to a file."""
        if not self.last_results:
            messagebox.showwarning("No Results", "No prediction results to save.")
            return

        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")]
            )

            if filename:
                with open(filename, 'w') as f:
                    f.write(f"Currency Detection Results\n")
                    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(
                        f"Image: {os.path.basename(self.selected_filepath) if self.selected_filepath else 'Unknown'}\n")
                    f.write("=" * 50 + "\n\n")

                    for rank, country, value, year, confidence in self.last_results:
                        f.write(f"#{rank}: {country} {value} ({year}) - {confidence}\n")

                    f.write("\nFull Labels:\n")
                    for i, (_, _, _, _, _) in enumerate(self.last_results):
                        if i < len(self.class_names):
                            # It's better to store the actual predicted raw labels if possible,
                            # but for now, we'll just list the relevant class names based on indices.
                            # The 'raw_labels' list created in _analyze_file is more accurate here.
                            f.write(f"- {self.class_names[i]}\n")

                messagebox.showinfo("Success", f"Results saved to {filename}")
                self._update_status(f"Results saved to {os.path.basename(filename)}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save results:\n{e}")

    def _update_currency_info(self, results, countries):
        """Update the currency information and comparison sections."""
        if not results:
            # Clear text widgets if no results
            self.currency_info_text.config(state="normal")
            self.currency_info_text.delete(1.0, tk.END)
            self.currency_info_text.insert(tk.END, "No primary currency match found.")
            self.currency_info_text.config(state="disabled")

            self.comparison_text.config(state="normal")
            self.comparison_text.delete(1.0, tk.END)
            self.comparison_text.insert(tk.END, "No alternative matches available.")
            self.comparison_text.config(state="disabled")

            self.stats_label.config(text="No valid prediction results to display stats.")
            return

        # Update currency info
        self.currency_info_text.config(state="normal")
        self.currency_info_text.delete(1.0, tk.END)

        top_result = results[0]
        info_text = f"🏆 Best Match: {top_result[1]} {top_result[2]}\n"
        info_text += f"📅 Year: {top_result[3]}\n"
        info_text += f"🎯 Confidence: {top_result[4]}\n\n"
        info_text += f"📊 Total Predictions: {len(results)}\n"
        info_text += f"🌍 Unique Countries: {len(countries)}\n" # Changed "Countries Found" to "Unique Countries"

        # Add confidence distribution
        try: # Added try-except for robustness in case confidence string parsing fails
            high_conf = len([r for r in results if float(r[4].replace('%', '')) > 70])
            med_conf = len([r for r in results if 30 <= float(r[4].replace('%', '')) <= 70])
            low_conf = len([r for r in results if float(r[4].replace('%', '')) < 30])
        except ValueError:
            high_conf, med_conf, low_conf = 0, 0, 0 # Default if parsing fails

        info_text += f"\n🔥 High Confidence (>70%): {high_conf}\n"
        info_text += f"⚡ Medium Confidence (30-70%): {med_conf}\n"
        info_text += f"❄️ Low Confidence (<30%): {low_conf}"

        self.currency_info_text.insert(tk.END, info_text)
        self.currency_info_text.config(state="disabled")

        # Update comparison info
        self.comparison_text.config(state="normal")
        self.comparison_text.delete(1.0, tk.END)

        comparison_text = "🔍 Alternative Matches:\n\n"
        if len(results) > 1:
            for i, result in enumerate(results[1:], 2):  # Skip first result
                comparison_text += f"#{i}: {result[1]} {result[2]}\n"
                comparison_text += f"   Year: {result[3]} | Conf: {result[4]}\n\n"
        else:
            comparison_text += "Only one result found above the confidence threshold.\n"
            comparison_text += "Try lowering the 'Confidence Threshold' to see more alternatives."


        self.comparison_text.insert(tk.END, comparison_text)
        self.comparison_text.config(state="disabled")

        # Update stats label
        if len(results) > 0:
            try:
                avg_confidence = sum(float(r[4].replace('%', '')) for r in results) / len(results)
            except ValueError:
                avg_confidence = 0.0 # Handle case where confidence string is not a valid float
            self.stats_label.config(
                text=f"📈 Avg Conf: {avg_confidence:.1f}% | 🎯 Best: {top_result[4]} | 🌍 Countries: {', '.join(countries[:3]) + ('...' if len(countries) > 3 else '')}")
        else:
            self.stats_label.config(text="No valid prediction results to display stats.")


    def _browse_files(self):
        """Open file dialog to select an image."""
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("PNG files", "*.png"),
            ("All files", "*.*")
        ]

        filepath = filedialog.askopenfilename(filetypes=filetypes)
        if filepath:
            self.selected_filepath = filepath
            try:
                # Display image
                img = Image.open(filepath)
                # Calculate aspect ratio and resize
                img_ratio = img.width / img.height
                display_width = 300
                display_height = int(display_width / img_ratio) if img_ratio > 1 else 200

                img.thumbnail((display_width, display_height), Image.Resampling.LANCZOS)
                tk_image = ImageTk.PhotoImage(img)
                self.image_label.configure(image=tk_image)
                self.image_label.image = tk_image

                # Update file info
                file_size = os.path.getsize(filepath) / 1024  # KB
                self.selected_file_label.config(
                    text=f"Selected: {os.path.basename(filepath)} ({file_size:.1f} KB)")

                self._clear_results()
                self.result_tree.insert("", "end", values=("", "", "Click 'Predict'", "", ""))
                self._update_status(f"Image loaded: {os.path.basename(filepath)}")

            except Exception as e:
                messagebox.showerror("Image Load Error", f"Could not load image:\n{e}")
                self._update_status("Image loading failed")

    def _analyze_file_threaded(self):
        """Run prediction in a separate thread to avoid UI freezing."""
        if self.is_predicting:
            return

        if not self.selected_filepath:
            messagebox.showwarning("No File", "Please choose an image file first.")
            return

        if self.model is None or not self.class_names:
            messagebox.showerror("Prediction Error", "Model or class names not loaded.")
            return

        self.is_predicting = True
        self.predict_button.config(state="disabled")

        # Run prediction in separate thread
        thread = threading.Thread(target=self._analyze_file, daemon=True)
        thread.start()

    def _analyze_file(self):
        """Analyze the selected file using the ML model."""
        try:
            self._update_status("Preprocessing image...")

            # Clear previous results
            self.after(0, self._clear_results)
            self.after(0, lambda: self.result_tree.insert("", "end",
                                                          values=("", "", "Analyzing image...", "", "")))

            # Preprocess image
            result = normalize_image(self.selected_filepath)
            if isinstance(result, tuple):  # Error occurred
                input_array, error = result
                if input_array is None:
                    self.after(0, lambda: self._show_error(f"Image preprocessing failed: {error}"))
                    return
            else:
                input_array = result

            self._update_status("Running prediction...")

            # Run prediction
            predictions = self.model.predict(input_array, verbose=0)

            # Use dynamic top N and confidence threshold
            top_n = self.topn_var.get()
            confidence_threshold = self.confidence_var.get()

            # Get all predictions above threshold, then take top N
            all_predictions = [(i, predictions[0][i]) for i in range(len(predictions[0]))]
            filtered_predictions = [(i, conf) for i, conf in all_predictions if conf >= confidence_threshold]
            filtered_predictions.sort(key=lambda x: x[1], reverse=True)
            top_indices = [idx for idx, _ in filtered_predictions[:top_n]]

            # Process results
            results = []
            raw_labels = []
            countries_found = []

            for i, idx in enumerate(top_indices):
                predicted_label = self.class_names[idx]
                confidence = predictions[0][idx] * 100
                country, value, year_range = parse_currency_label(predicted_label)

                results.append((i + 1, country, value, year_range, f"{confidence:.2f}%"))
                raw_labels.append(f"#{i + 1}: {predicted_label} ({confidence:.2f}%)")
                if country not in countries_found:
                    countries_found.append(country)

            # Store results for other functions
            self.last_results = results

            # Update UI in main thread
            self.after(0, lambda: self._display_results(results, raw_labels))
            self.after(0, lambda: self._update_currency_info(results, countries_found))

            # Add to history with more detail
            if results:
                best_result = results[0]
                history_entry = f"{datetime.now().strftime('%H:%M:%S')} - {best_result[1]} {best_result[2]} ({best_result[4]})"
                self.prediction_history.append(history_entry)

            self._update_status("Prediction completed successfully")

        except Exception as e:
            error_msg = f"Prediction failed: {str(e)}"
            self.after(0, lambda: self._show_error(error_msg))
            self._update_status("Prediction failed")

        finally:
            # Re-enable UI
            self.after(0, lambda: self.predict_button.config(state="normal"))
            self.is_predicting = False

    def _display_results(self, results, raw_labels):
        """Display prediction results in the UI."""
        self._clear_results()

        # Populate table
        if not results:
            self.result_tree.insert("", "end", values=("", "", "No confident predictions", "", ""))
        else:
            for result in results:
                self.result_tree.insert("", "end", values=result)

        # Populate full details
        self.full_details_text.config(state="normal")
        if raw_labels:
            self.full_details_text.insert(tk.END, "\n".join(raw_labels))
        else:
            self.full_details_text.insert(tk.END, "No detailed labels available above threshold.")
        self.full_details_text.config(state="disabled")

    def _show_error(self, error_message):
        """Display error message in the UI."""
        messagebox.showerror("Prediction Error", error_message)
        self._clear_results()
        self.result_tree.insert("", "end", values=("", "", "Prediction Failed", "", ""))

        self.full_details_text.config(state="normal")
        self.full_details_text.delete(1.0, tk.END)
        self.full_details_text.insert(tk.END, f"Error: {error_message}")
        self.full_details_text.config(state="disabled")


if __name__ == "__main__":
    app = FileUploaderApp()
    app.mainloop()