import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import random
from normalize import normalize_image
import matplotlib.pyplot as plt

# Warna dan Font yang digunakan (tidak berubah)
BG_COLOR = "#e9ebee"
CARD_COLOR = "#ffffff"
TEXT_COLOR = "#333333"
PRIMARY_COLOR = "#007bff"
FONT_BOLD = ("Helvetica", 12, "bold")
FONT_NORMAL = ("Helvetica", 10)
FONT_SMALL = ("Helvetica", 8)


class FileUploaderApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("File Uploader")
        self.geometry("800x600")
        self.config(bg=BG_COLOR)

        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(2, weight=1)

        try:
            self.upload_icon_img = ImageTk.PhotoImage(Image.open("icons/upload-cloud.png").resize((64, 64)))
            self.file_icon_img = ImageTk.PhotoImage(Image.open("icons/file-icon.jpg").resize((24, 24)))
            self.trash_icon_img = ImageTk.PhotoImage(Image.open("icons/trash-icon.png").resize((20, 20)))
        except Exception as e:
            print(f"Error loading images: {e}")
            self.upload_icon_img, self.file_icon_img, self.trash_icon_img = None, None, None

        self._create_widgets()

    def _create_widgets(self):
        card_frame = tk.Frame(self, bg=CARD_COLOR, padx=30, pady=30, relief="solid", borderwidth=1,
                              highlightbackground="#ddd", highlightthickness=1)
        card_frame.grid(row=1, column=1, sticky="")

        # ... (Sisa dari _create_widgets tidak berubah, jadi saya persingkat)
        title_label = tk.Label(card_frame, text="Add File", font=("Helvetica", 16, "bold"), bg=CARD_COLOR,
                               fg=TEXT_COLOR)
        title_label.pack(anchor="w", pady=(0, 20))
        upload_area_canvas = tk.Canvas(card_frame, bg=CARD_COLOR, width=400, height=280, highlightthickness=0)
        upload_area_canvas.pack(fill="x", pady=10)
        upload_area_canvas.create_rectangle(5, 5, 395, 250, outline=PRIMARY_COLOR, dash=(5, 3), width=2)
        upload_content_frame = tk.Frame(upload_area_canvas, bg=CARD_COLOR)
        upload_content_frame.place(relx=0.5, rely=0.5, anchor="center")
        if self.upload_icon_img: tk.Label(upload_content_frame, image=self.upload_icon_img, bg=CARD_COLOR).pack()
        tk.Label(upload_content_frame, text="Upload your file here", font=FONT_BOLD, bg=CARD_COLOR, fg=TEXT_COLOR).pack(
            pady=(5, 2))
        tk.Label(upload_content_frame, text="Files supported: TXT, HTML, CSS", font=FONT_SMALL, bg=CARD_COLOR,
                 fg="#888").pack()
        tk.Label(upload_content_frame, text="OR", font=FONT_NORMAL, bg=CARD_COLOR, fg="#888").pack(pady=5)
        browse_button = tk.Button(upload_content_frame, text="BROWSE", font=FONT_NORMAL, bg=CARD_COLOR,
                                  fg=PRIMARY_COLOR, command=self._browse_files, relief="solid", borderwidth=1, padx=15,
                                  pady=3)
        browse_button.pack(pady=5)
        tk.Label(upload_content_frame, text="Maximum size: 2MB", font=FONT_SMALL, bg=CARD_COLOR, fg="#888").pack()
        self.file_list_frame = tk.Frame(card_frame, bg=CARD_COLOR)
        self.file_list_frame.pack(fill="x", pady=(10, 0))
        self.analyze_button = tk.Button(card_frame, text="Analyze", font=FONT_BOLD, bg=PRIMARY_COLOR, fg=CARD_COLOR,
                                        relief="flat", command=self._analyze_files)

    # [DIUBAH] Fungsi browse kembali menggunakan askopenfilename (singular)
    def _browse_files(self):
        filepath = filedialog.askopenfilename(  # Hanya bisa pilih satu file
            title="Select a file",
            filetypes=(("Image Files", "*.png *.jpg *.jpeg"), ("Text files", "*.txt"), ("HTML files", "*.html"),
                       ("CSS files", "*.css"), ("All files", "*.*"))
        )
        if filepath:
            # [BARU] Hapus file lama sebelum menambahkan yang baru
            self._clear_file_list()

            filename = os.path.basename(filepath)
            filesize = os.path.getsize(filepath)
            self._add_file_entry(filename, filesize)

        self.selected_filepath = filepath


    # [FUNGSI BARU] untuk membersihkan daftar file
    def _clear_file_list(self):
        """Hapus semua entri file dari UI dan sembunyikan tombol Analyze."""
        # Loop melalui salinan list widget anak, karena menghancurkan widget
        # saat iterasi bisa menyebabkan masalah.
        for widget in list(self.file_list_frame.winfo_children()):
            widget.destroy()

        # Sembunyikan tombol analyze jika terlihat
        if self.analyze_button.winfo_viewable():
            self.analyze_button.pack_forget()

    def _add_file_entry(self, name, size_bytes):
        item_frame = tk.Frame(self.file_list_frame, bg="#f8f9fa", padx=10, pady=5)
        item_frame.pack(fill="x", pady=(2, 0))
        item_frame.filename = name

        if self.file_icon_img:
            tk.Label(item_frame, image=self.file_icon_img, bg="#f8f9fa").pack(side="left")

        info_frame = tk.Frame(item_frame, bg="#f8f9fa")
        info_frame.pack(side="left", padx=10, fill="x", expand=True)

        tk.Label(info_frame, text=name, font=FONT_NORMAL, bg="#f8f9fa", fg=TEXT_COLOR).pack(anchor="w")
        tk.Label(info_frame, text=self._format_size(size_bytes), font=FONT_SMALL, bg="#f8f9fa", fg="#888").pack(
            anchor="w")

        if self.trash_icon_img:
            delete_button = tk.Button(item_frame, image=self.trash_icon_img, bg="#f8f9fa", relief="flat",
                                      command=lambda frame=item_frame: self._delete_file_entry(frame))
            delete_button.pack(side="right")

        if not self.analyze_button.winfo_viewable():
            self.analyze_button.pack(pady=(20, 0), fill='x', ipady=5)

    def _delete_file_entry(self, frame_to_delete):
        frame_to_delete.destroy()
        if not self.file_list_frame.winfo_children():
            self.analyze_button.pack_forget()

    def _analyze_files(self):
        file_widgets = self.file_list_frame.winfo_children()

        if not file_widgets:
            messagebox.showwarning("No File", "There is no file to analyze.")
            return

        # Sekarang kita tahu hanya ada satu file, kita bisa ambil langsung
        filepath = self.selected_filepath
        filename = os.path.basename(filepath)

        # Normalisasi gambar (panggil dari normalize.py)
        normalized_array = normalize_image(filepath)
        if normalized_array is None:
            messagebox.showerror("Error", "Failed to normalize image.")
            return

        messagebox.showinfo("Analyzing...", f"Starting analysis on: {filename}")

        dummy_currencies = {"Indonesian Rupiah": "Indonesia", "US Dollar": "USA", "Euro": "Germany",
                            "Japanese Yen": "Japan", "British Pound": "UK"}
        currency, country = random.choice(list(dummy_currencies.items()))
        year = random.randint(1990, 2023)

        final_result_string = f"File: {filename}\n- Currency: {currency}\n- Country: {country}\n- Year: {year}\n"

        messagebox.showinfo("Analysis Complete", final_result_string)

        #dihapus
        if normalized_array.shape[-1] == 1:
            plt.imshow(normalized_array[0, :, :, 0], cmap='gray')
        else:
            plt.imshow(normalized_array[0])  # RGB

        plt.title("Hasil Normalisasi")
        plt.axis("off")
        plt.show()

    def _format_size(self, size_bytes):
        if size_bytes < 1024:
            return f"{size_bytes}B"
        elif size_bytes < 1024 ** 2:
            return f"{size_bytes / 1024:.1f}KB"
        else:
            return f"{size_bytes / (1024 ** 2):.1f}MB"


if __name__ == "__main__":
    app = FileUploaderApp()
    app.mainloop()