# final_gui.py (versi revisi dengan tampilan skor plagiarisme dan pewarnaan + threading agar progress bar berjalan)

import os
import threading  # 🔹 Tambahan: untuk menjalankan proses berat di background
import tkinter as tk
from tkinter import messagebox, filedialog
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import requests
from main import detect_plagiarism, export_to_csv


def check_internet():
    try:
        requests.get("https://www.google.com", timeout=5)
        return True
    except:
        return False


class App:
    def __init__(self, root):
        self.root = root
        root.title("Online Plagiarism Detection for Scientific Journals (TF-IDF & Cosine Similarity)")
        root.geometry("1000x700")
        self.results = []
        self.overall_score = 0
        self.threshold = tk.IntVar(value=30)   # 🔹 Threshold dinamis
        self.api_choice = tk.StringVar(value="Auto")  # 🔹 Pilihan API
        self.setup_ui()

        if not check_internet():
            messagebox.showwarning("No Internet", "Please connect to the internet to use this app.")
            root.destroy()

    def setup_ui(self):
        frm = ttk.Frame(self.root, padding=10)
        frm.pack(fill=tk.BOTH, expand=True)

        # Label judul
        ttk.Label(frm, text="Online Plagiarism Detection Application", font=("Helvetica", 18, "bold")).pack(pady=5)
        ttk.Label(frm, text="(Semantic Scholar & Crossref API Based)", font=("Helvetica", 18, "bold")).pack(pady=5)

        # 🔹 Pengaturan Threshold
        control_frame = ttk.Frame(frm)
        control_frame.pack(pady=5)
        ttk.Label(control_frame, text="Threshold (%) :").grid(row=0, column=0, padx=5)
        tk.Spinbox(control_frame, from_=0, to=100, width=5, textvariable=self.threshold).grid(row=0, column=1, padx=5)

        # 🔹 Pilihan API
        ttk.Label(control_frame, text="API :").grid(row=0, column=2, padx=5)
        ttk.Combobox(control_frame, textvariable=self.api_choice, values=["Auto", "Semantic Scholar", "CrossRef"], width=20).grid(row=0, column=3, padx=5)
        
        # Label hasil skor akhir
        self.result_label = ttk.Label(frm, text="Plagiarism Level: -", font=("Helvetica", 16, "bold"))
        self.result_label.pack(pady=10)

        # Tombol
        btn_frame = ttk.Frame(frm)
        btn_frame.pack(pady=5)

        self.btn = ttk.Button(btn_frame, text="Select PDF and Check", command=self.on_browse, bootstyle=SUCCESS)
        self.btn.grid(row=0, column=0, padx=5)

        self.export_btn = ttk.Button(btn_frame, text="Export CSV", command=self.export_csv, bootstyle=INFO)
        self.export_btn.grid(row=0, column=1, padx=5)

        self.clear_btn = ttk.Button(btn_frame, text="Clear Table", command=self.clear_table, bootstyle=SECONDARY)
        self.clear_btn.grid(row=0, column=2, padx=5)

        # Progress bar
        self.progress = ttk.Progressbar(frm, mode="indeterminate", bootstyle=INFO)
        self.progress.pack(fill=tk.X, pady=5)

        # Status proses
        self.status_label = ttk.Label(frm, text="", font=("Helvetica", 12))
        self.status_label.pack(pady=5)

        # Preview teks PDF
        self.text_preview = tk.Text(frm, height=8, wrap=tk.WORD)
        self.text_preview.pack(fill=tk.BOTH, expand=True, pady=5)

        # 🔹 Tabel hasil diperluas (tambah kolom TF-IDF & Word2Vec)
        cols = ("Target", "Reference", "TF-IDF (%)", "Word2Vec (%)", "Similarity (%)", "Status")
        self.tree = ttk.Treeview(frm, columns=cols, show="headings", height=12)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, anchor=tk.CENTER, width=160)

        self.tree.tag_configure("plagiarism", background="#b9d0f7", foreground="blue")
        self.tree.tag_configure("free", background="#ddffdd", foreground="green")
        self.tree.pack(fill=tk.BOTH, expand=True, pady=5)

    def on_browse(self):
        path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if not path: return
        self.status_label.config(text="🔄 Processing... Please wait.", foreground="orange")
        self.progress.start(10)
        self.root.update_idletasks()

        # Preview PDF
        self.text_preview.delete(1.0, tk.END)
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(path)
            preview_text = reader.pages[0].extract_text()[:800]
            self.text_preview.insert(tk.END, preview_text)
        except:
            self.text_preview.insert(tk.END, "[Failed to read PDF preview]")

        def worker():
            results, overall = detect_plagiarism(
                path,
                threshold=self.threshold.get(),
                api_choice=self.api_choice.get()
            )
            self.results, self.overall_score = results, overall
            self.root.after(0, lambda: self.show_results(results, overall))

        threading.Thread(target=worker, daemon=True).start()

    def show_results(self, results, overall):
        self.progress.stop()
        self.status_label.config(text="✅ Process complete", foreground="green")
        color = "green" if overall < 20 else "blue" if overall <= 50 else "red"
        self.result_label.config(text=f"Plagiarism Level: {overall}%", foreground=color)
        self.tree.delete(*self.tree.get_children())
        for r in results:
            tag = "plagiarism" if r["Status"] == "plagiarism" else "free"
            self.tree.insert('', 'end', values=(r["Target"], r["Reference"], r["TF-IDF (%)"], r["Word2Vec (%)"], r["Similarity (%)"], r["Status"]), tags=(tag,))

    def export_csv(self):
        if not self.results:
            messagebox.showinfo("Empty", "There is no data to save yet.")
            return
        export_to_csv(self.results, self.overall_score)
        messagebox.showinfo("Success", "Data saved to hasil_plagiarisme.csv")

    def clear_table(self):
        self.tree.delete(*self.tree.get_children())
        self.results.clear()
        self.overall_score = 0
        self.text_preview.delete(1.0, tk.END)
        self.result_label.config(text="Plagiarism Level: -", foreground="black")

if __name__ == "__main__":
    root = ttk.Window(themename="flatly")
    app = App(root)
    root.mainloop()
