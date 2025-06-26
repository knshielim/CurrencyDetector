import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import os
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
import re


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
        messagebox.showerror("Error", f"Image preprocessing failed:\n{e}")
        return None

def create_rounded_rectangle_image(width, height, radius, color, file_name):
    img = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((0, 0, width, height), radius=radius, fill=color)
    img.save(file_name)
    print(f"Generated {file_name}")

# --- Main Application Class ---
class FileUploaderApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Currency Detector")
        self.geometry("700x800")
        self.configure(bg="#2C2C2C")

        # Load model and class names
        #Source: Geeks for Geeks. (2024, March 20). tf.keras.models.load_model in Tensorflow – GeeksforGeeks. Geeks for Geeks. https://www.geeksforgeeks.org/deep-learning/tf-keras-models-load_model-in-tensorflow/
        try:
            self.model = load_model("currency_classifier.h5")
        except Exception as e:
            messagebox.showerror("Model Load Error",f"Could not load model. Please ensure 'currency_classifier.h5' is in the same directory.\nDetails: {e}")
            self.model = None

        try:
            with open("class_names.pkl", "rb") as f:
                self.class_names = pickle.load(f)
        except Exception as e:
            messagebox.showerror("Class Names Load Error", f"Could not load class names. Please ensure 'class_names.pkl' is in the same directory.\nDetails: {e}")
            self.class_names = []

        self.selected_filepath = None

        try:
            self.btn_green_img = ImageTk.PhotoImage(Image.open("button_green_rounded.png"))
            self.btn_dark_green_img = ImageTk.PhotoImage(
                Image.open("button_dark_green_rounded.png"))
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

        #Source: Geeks for Geeks. (2021, July 14). How to create a scrollable frame in Tkinter? . Geeks for Geeks. how-to-create-a-scrollable-frame-in-tkinter/
        main_canvas = tk.Canvas(self, bg="#2C2C2C", highlightthickness=0)
        main_canvas.pack(side="left", fill="both", expand=True)

        scrollbar = tk.Scrollbar(self, orient="vertical", command=main_canvas.yview)
        scrollbar.pack(side="right", fill="y")

        main_canvas.configure(yscrollcommand=scrollbar.set)

        self.scrollable_frame = tk.Frame(main_canvas, bg="#2C2C2C")

        self.canvas_frame_id = main_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        self.scrollable_frame.bind("<Configure>", lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all")))

        def _on_canvas_resize(event):
            canvas_width = event.width
            main_canvas.itemconfig(self.canvas_frame_id, width=canvas_width)

        main_canvas.bind("<Configure>", _on_canvas_resize)

        header_frame = tk.Frame(self.scrollable_frame, bg="#1C1C1C", pady=10)
        header_frame.pack(fill="x")
        tk.Label(header_frame, text="💰 Currency Detector 💰", font=("Arial", 24, "bold"), bg="#1C1C1C", fg="white").pack(pady=5)

        main_content_frame = tk.Frame(self.scrollable_frame, bg="#2C2C2C", padx=20, pady=20)
        main_content_frame.pack(fill="both", expand=True)

        image_selection_frame = tk.LabelFrame(main_content_frame, text="Image Selection", font=("Arial", 14, "bold"), bg="#3A3A3A", fg="#E0E0E0", padx=15, pady=15, bd=2, relief="groove")
        image_selection_frame.pack(pady=10, fill="x")

        self.choose_image_button = tk.Button(image_selection_frame, text="📂 Choose Image",command=self._browse_files,font=("Arial", 12, "bold"),fg="white",cursor="hand2",relief="flat",bd=0,bg=image_selection_frame['bg'],activebackground=image_selection_frame['bg'])
        if self.btn_green_img:
            self.choose_image_button.config(image=self.btn_green_img, compound="center")
        else:
            self.choose_image_button.config(bg="#4CAF50")
        self.choose_image_button.pack(pady=10)


        self.image_label = tk.Label(image_selection_frame, bg="#424242", bd=1,relief="solid")
        self.image_label.pack(pady=10, padx=10, fill="both", expand=True)

        self.placeholder_img = Image.new('RGB', (200, 200), color='#606060')
        self.placeholder_tk = ImageTk.PhotoImage(self.placeholder_img)
        self.image_label.configure(image=self.placeholder_tk)
        self.image_label.image = self.placeholder_tk

        self.selected_file_label = tk.Label(image_selection_frame, text="No image selected", bg="#3A3A3A",font=("Arial", 10, "italic"), fg="#C0C0C0")
        self.selected_file_label.pack(pady=5)

        self.predict_button = tk.Button(main_content_frame,text="🚀 Predict Currency Details", command=self._analyze_file,font=("Arial", 14, "bold"),fg="white",cursor="hand2",relief="flat",bd=0,bg=main_content_frame['bg'],activebackground=main_content_frame['bg'])
        if self.btn_dark_green_img:
            self.predict_button.config(image=self.btn_dark_green_img, compound="center")
        else:
            self.predict_button.config(bg="#388E3C")
        self.predict_button.pack(pady=20)

        result_frame = tk.LabelFrame(main_content_frame, text="🔍 Detection Results",font=("Arial", 14, "bold"), bg="#3A3A3A", fg="#E0E0E0",padx=15, pady=15, bd=2, relief="groove")
        result_frame.pack(pady=10, fill="both", expand=True)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview",background="#333333",foreground="#E0E0E0",fieldbackground="#333333",bordercolor="#555555",lightcolor="#555555",darkcolor="#2C2C2C",rowheight=25)
        style.map("Treeview", background=[('selected', '#4A6984')])

        style.configure("Treeview.Heading",background="#424242",foreground="white",font=("Arial", 11, "bold"),relief="raised",bordercolor="#555555")
        style.map("Treeview.Heading", background=[('active', '#555555')])

        self.result_tree = ttk.Treeview(result_frame, columns=("Rank", "Country", "Value", "Year", "Confidence"),show="headings")
        self.result_tree.pack(side="left", fill="both", expand=True)

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

        self.tree_scrollbar = ttk.Scrollbar(result_frame, orient="vertical",command=self.result_tree.yview)
        self.tree_scrollbar.pack(side="right", fill="y")
        self.result_tree.config(yscrollcommand=self.tree_scrollbar.set)

        details_frame = tk.LabelFrame(result_frame, text="📋 Technical Details",font=("Arial", 12, "bold"), bg="#2A2A2A", fg="#FFD700",padx=12, pady=12, bd=1, relief="groove")
        details_frame.pack(pady=(15, 5), fill="both", expand=False)

        details_content_frame = tk.Frame(details_frame, bg="#2A2A2A")
        details_content_frame.pack(fill="both", expand=True)

        info_label = tk.Label(details_content_frame,text="🔬 Raw Model Classifications:",font=("Arial", 11, "bold"),bg="#2A2A2A",fg="#87CEEB",anchor="w")
        info_label.pack(pady=(0, 8), fill="x")

        text_container = tk.Frame(details_content_frame, bg="#1E1E1E", relief="solid", bd=1)
        text_container.pack(fill="both", expand=True)

        self.full_details_text = tk.Text(text_container,wrap="word",height=6,font=("Consolas", 10),bg="#1E1E1E",fg="#E8E8E8",relief="flat",bd=0,state="disabled",padx=12,pady=8,selectbackground="#4A6984",selectforeground="white",insertbackground="#FFD700")
        self.full_details_text.pack(side="left", fill="both", expand=True)

        details_scrollbar = tk.Scrollbar(text_container, orient="vertical", command=self.full_details_text.yview,bg="#333333", troughcolor="#2A2A2A",activebackground="#555555", bd=0)
        details_scrollbar.pack(side="right", fill="y")
        self.full_details_text.config(yscrollcommand=details_scrollbar.set)

        separator = tk.Frame(details_content_frame, height=1, bg="#555555")
        separator.pack(fill="x", pady=(8, 0))


        tip_label = tk.Label(details_content_frame,text="💡 Tip: These are the raw classification labels from the AI model",font=("Arial", 9, "italic"),bg="#2A2A2A",fg="#A0A0A0",anchor="w")
        tip_label.pack(pady=(4, 0), fill="x")

        self._clear_results()
        self.result_tree.insert("", "end", values=("", "Upload an image and", "click 'Predict'", "to see", "details."))

    def _clear_results(self):

        for item in self.result_tree.get_children():
            self.result_tree.delete(item)

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
            self.full_details_text.delete(1.0, tk.END)
            self.full_details_text.insert(tk.END, "🎯 Ready to analyze! Click 'Predict' to see detailed classification results.")
            self.full_details_text.config(state="disabled")

    def _analyze_file(self):
        if not self.selected_filepath:
            messagebox.showwarning("No File", "Please choose an image file first.")
            return

        if self.model is None or not self.class_names:
            messagebox.showerror("Prediction Error","Model or class names not loaded. Please check the console for details.")
            return

        self._clear_results()
        self.result_tree.insert("", "end", values=("", "", "Analyzing image...", "", ""))
        self.full_details_text.config(state="normal")
        self.full_details_text.delete(1.0, tk.END)
        self.full_details_text.insert(tk.END,"🔄 Processing image through neural network...\n⏳ Please wait while the AI model analyzes your currency.")
        self.full_details_text.config(state="disabled")
        self.update_idletasks()

        input_array = normalize_image(self.selected_filepath)
        if input_array is None:
            self.full_details_text.config(state="normal")
            self.full_details_text.delete(1.0, tk.END)
            self.full_details_text.insert(tk.END,"❌ Error: Image preprocessing failed.\n\nPlease ensure the image is in a supported format (JPG, JPEG, PNG) and try again.")
            self.full_details_text.config(state="disabled")
            self._clear_results()
            return

        #Source: Stack Overflow user Auraham. (2017, November 17). How to predict input image using trained model in Keras?. stack overflow. https://stackoverflow.com/questions/43469281/how-to-predict-input-image-using-trained-model-in-keras
        #Source: AstroIshu. (2024). Fake Currency Detection System. GitHub repo. https://github.com/AstroIshu/Fake-Currency-Detection-System
        try:
            predictions = self.model.predict(input_array, verbose=0)
            top_3_indices = np.argsort(predictions[0])[::-1][:3]

            self._clear_results()

            raw_label_details_list = []

            for i, idx in enumerate(top_3_indices):
                predicted_label = self.class_names[idx]
                confidence = predictions[0][idx] * 100

                country, value, year_range = parse_currency_label(predicted_label)

                self.result_tree.insert("", "end", values=(i + 1,country,value,year_range,f"{confidence:.2f}%"
                ))

                confidence_indicator = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                raw_label_details_list.append(f"{confidence_indicator} Rank {i + 1} ({confidence:.2f}% confidence):")
                raw_label_details_list.append(f"   └─ {predicted_label}")
                if i < len(top_3_indices) - 1:
                    raw_label_details_list.append("")

            self.full_details_text.config(state="normal")
            self.full_details_text.delete(1.0, tk.END)

            header_text = "✨ AI Model Classification Results\n" + "─" * 45 + "\n\n"
            self.full_details_text.insert(tk.END, header_text)

            self.full_details_text.insert(tk.END, "\n".join(raw_label_details_list))

            footer_text = f"\n\n" + "─" * 45 + f"\n📊 Analyzed {len(top_3_indices)} top predictions from {len(self.class_names)} possible currencies"
            self.full_details_text.insert(tk.END, footer_text)

            self.full_details_text.config(state="disabled")


        except Exception as e:
            messagebox.showerror("Prediction Error", f"An error occurred during prediction:\n{e}")
            print(f"DEBUG: Prediction Exception: {e}")
            self._clear_results()
            self.result_tree.insert("", "end", values=("", "", "Prediction Failed", "", ""))
            self.full_details_text.config(state="normal")
            self.full_details_text.delete(1.0, tk.END)
            self.full_details_text.insert(tk.END,f"❌ Prediction Error\n\n{'─' * 30}\n\nCould not analyze the image due to an unexpected error.\n\nTechnical Details:\n{str(e)}\n\n💡 Try with a different image or check that the model files are properly loaded.")
            self.full_details_text.config(state="disabled")


if __name__ == "__main__":
    app = FileUploaderApp()
    app.mainloop()