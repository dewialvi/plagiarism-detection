# final_main.py (versi revisi dengan bobot TF-IDF lebih tinggi dan skor akhir maksimum)

import os
import re
import glob
import requests
import time
from collections import Counter
import numpy as np
import pandas as pd
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from colorama import Fore, Style
from datetime import datetime   # 🔹 ditambah untuk revisi export CSV

nltk.download('stopwords', quiet=True)
factory = StemmerFactory()
stemmer = factory.create_stemmer()
ind_stop = set(stopwords.words('indonesian'))

THRESHOLD = 30
SCALING_FACTOR = 0.60  # 🔹 Lebih tinggi supaya hasil mendekati Turnitin

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    tokens = [w for w in text.split() if w not in ind_stop and len(w) > 2]
    return ' '.join(stemmer.stem(w) for w in tokens)

def extract_top_keywords(text: str, n: int = 5):
    text = re.sub(r'[^a-zA-Z ]', ' ', text.lower())
    words = text.split()
    freq = Counter([w for w in words if len(w) > 2])
    return [w for w, _ in freq.most_common(n)]

def search_semantic_scholar(query: str, limit: int = 20):  # 🔹 Naikkan limit jadi 20
    ss_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    ss_params = {"query": query, "limit": limit, "fields": "title,abstract,url"}
    try:
        response = requests.get(ss_url, params=ss_params, timeout=10)
        response.raise_for_status()
        time.sleep(1)
        data = response.json()
        results = []
        for p in data.get("data", []):
            abstract = p.get("abstract") or p.get("title", "")
            if abstract.strip():
                results.append({
                    "title": p.get("title", ""),
                    "abstract": abstract,
                    "url": p.get("url", ""),
                    "source": "SemanticScholar"
                })
        return results
    except:
        try:
            cr = requests.get("https://api.crossref.org/works", params={"query": query, "rows": limit}, timeout=10)
            cr.raise_for_status()
            items = cr.json().get("message", {}).get("items", [])
            results = []
            for it in items:
                title = it.get("title", [""])[0]
                abstract = it.get("abstract") or title
                if abstract.strip():
                    results.append({
                        "title": title,
                        "abstract": abstract,
                        "url": it.get("URL", ""),
                        "source": "CrossRef"
                    })
            return results
        except:
            return []

def train_word2vec(corpus):
    tokenized = [doc.split() for doc in corpus]
    return Word2Vec(tokenized, vector_size=100, window=5, min_count=1, workers=4)

def sentence_embedding(tokens, model):
    vecs = [model.wv[t] for t in tokens.split() if t in model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(model.vector_size)

def semantic_similarity(doc1, doc2, model):
    e1 = sentence_embedding(doc1, model)
    e2 = sentence_embedding(doc2, model)
    if np.linalg.norm(e1) == 0 or np.linalg.norm(e2) == 0:
        return 0.0
    return np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))

def colorize(percent):
    if percent < 20:
        return Fore.GREEN + str(percent) + '%' + Style.RESET_ALL
    elif percent <= 50:
        return Fore.YELLOW + str(percent) + '%' + Style.RESET_ALL
    else:
        return Fore.RED + str(percent) + '%' + Style.RESET_ALL

def detect_plagiarism(pdf_path: str, threshold=30, api_choice="Auto"):
    with pdfplumber.open(pdf_path) as pdf:
        txt = "\n".join(pg.extract_text() or "" for pg in pdf.pages)
    pre = preprocess_text(txt)
    keywords = extract_top_keywords(pre)
    # API sesuai pilihan user
    if api_choice == "Semantic Scholar":
        papers = search_semantic_scholar(" ".join(keywords), limit=20)
    elif api_choice == "CrossRef":
        papers = search_semantic_scholar(" ".join(keywords), limit=20)  # fallback sama
    else:  # Auto
        papers = search_semantic_scholar(" ".join(keywords), limit=20)

    refs, titles = [], []
    for p in papers:
        ref_text = p["abstract"] or p["title"]
        if ref_text.strip():
            refs.append(preprocess_text(ref_text))
            titles.append(p["title"])

    if not refs:
        print("❌ Tidak ditemukan referensi yang valid.")
        return [], 0

    vec = TfidfVectorizer()
    mat = vec.fit_transform([pre] + refs)
    tfidf_scores = cosine_similarity(mat[0:1], mat[1:])[0]

    model = train_word2vec([pre] + refs)
    w2v_scores = [semantic_similarity(pre, r, model) for r in refs]

    results = []
    all_percentages = []
    
    for title, s_tf, s_w2v in zip(titles, tfidf_scores, w2v_scores):
        # 🔹 Bobot TF-IDF lebih tinggi (70%) daripada Word2Vec (30%)
        raw_combined = (0.7 * s_tf) + (0.3 * s_w2v)
        scaled_percent = int(round(raw_combined * 100 * SCALING_FACTOR))
        all_percentages.append(scaled_percent)
        status = "plagiarism" if scaled_percent >= THRESHOLD else "free"
        results.append({
            "Target": os.path.basename(pdf_path),
            "Reference": title,
            "TF-IDF (%)": int(round(s_tf * 100)),
            "Word2Vec (%)": int(round(s_w2v * 100)),
            "Similarity (%)": scaled_percent,
            "Status": status
        })

    # 🔹 Skor akhir pakai nilai maksimum, bukan rata-rata
    overall_score = int(round(max(all_percentages))) if all_percentages else 0

    print(f"\nTotal Akumulasi Plagiarisme: {colorize(overall_score)}")
    return results, overall_score

# ===================== 🔹 REVISI FUNGSI EXPORT CSV 🔹 =====================
import csv
def export_to_csv(results, overall, out_path="hasil_plagiarisme.csv"):
    if not results:
        return

    try:
        target_file = results[0]["Target"] if results else "-"
        kategori = "Rendah" if overall < 20 else "Sedang" if overall <= 50 else "Tinggi"

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Judul Report
            writer.writerow(["=== PLAGIARISM RESULTS REPORT ==="])
            writer.writerow(["Target File", target_file])
            writer.writerow(["Generated By", "Online Plagiarism Detection System"])
            writer.writerow(["Date", datetime.now().strftime("%Y-%m-%d %H:%M")])
            writer.writerow([])  # baris kosong

            # Header Tabel
            writer.writerow(["Reference", "Similarity (%)", "Status"])

            # Isi tabel
            for r in results:
                writer.writerow([r["Reference"], f"{r['Similarity (%)']}%", r["Status"]])

            writer.writerow([])  # baris kosong

            # Ringkasan total
            writer.writerow(["TOTAL PLAGIARISM SCORE", f"{overall}%", kategori])

        print(f"✅ Report berhasil diekspor ke {out_path}")

    except Exception as e:
        print(f"❌ Terjadi kesalahan saat ekspor CSV: {e}")
# ==========================================================================

if __name__ == "__main__":
    paths = glob.glob(os.path.join("jurnal", "*.pdf"))
    if not paths:
        print("Tidak ada PDF di folder 'jurnal'")
        exit(1)

    res, total = detect_plagiarism(paths[0])
    if res:
        df = pd.DataFrame(res)
        print(df.to_string(index=False))
        export_to_csv(res, total)
        print("\nLaporan disimpan ke hasil_plagiarisme.csv")
    else:
        print("Gagal mendeteksi plagiarisme.")
