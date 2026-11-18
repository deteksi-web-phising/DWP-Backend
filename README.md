# Phishing Website Detection & Analysis - Backend API

RESTful API backend berbasis **FastAPI** dan **Machine Learning (Scikit-Learn)** untuk deteksi dini, analisis, dan klasifikasi website phishing. Dibangun dengan pola **Clean Code (Feature-First Architecture)** untuk performa tinggi, modularitas, dan skalabilitas.

---

## 🚀 Fitur Utama

- **Inference Engine & URL Feature Extraction**: Ekstraksi otomatis 30 fitur struktural URL (SSL state, IP address, subdomain, URL length, token HTTPS, dsb.) disertai dukungan input manual.
- **Multi-Model Machine Learning Benchmark**: Evaluasi dan komparasi performa 4 algoritma:
  - *Supervised Learning*: Random Forest, Decision Tree, k-Nearest Neighbors (k-NN).
  - *Unsupervised Learning*: K-Means Clustering.
- **Exploratory Data Analysis (EDA) Service**: Penyedia data agregasi statistik, distribusi kelas target (*Phishing* vs *Legitimate*), dan korelasi antar-fitur secara real-time.
- **API Documentation**: Dokumentasi interaktif otomatis menggunakan Swagger UI dan ReDoc.

---

## 🏗️ Arsitektur Proyek (Feature-First)

Struktur kode diorganisasi berdasarkan domain fitur (*feature-first*) guna memisahkan *concern* bisnis dan mempermudah kolaborasi tim:

```text
backend/
├── app/
│   ├── main.py                     # Entry point FastAPI, CORS, & registrasi router
│   ├── core/                       # Pengaturan global & konfigurasi sistem
│   │   ├── __init__.py
│   │   └── config.py               # Pydantic BaseSettings & Environment variables
│   ├── features/                   # Modul Fitur (Feature-First)
│   │   ├── detection/              # Fitur 1: Deteksi URL & Inferensi Phishing
│   │   │   ├── router.py           # Endpoints: /extract-url, /predict
│   │   │   ├── schemas.py          # Data Transfer Objects (DTO) & Pydantic models
│   │   │   ├── service.py          # Business logic inferensi model ML
│   │   │   └── extractor.py        # Logika ekstraksi 30 atribut URL
│   │   ├── benchmark/              # Fitur 2: Komparasi Model
│   │   │   ├── router.py           # Endpoint: /benchmark
│   │   │   ├── schemas.py          # Schema metrik akurasi, F1, confusion matrix
│   │   │   └── service.py          # Kalkulasi & data performa model
│   │   └── eda/                    # Fitur 3: Analisis Data Eksploratif (EDA)
│   │       ├── router.py           # Endpoint: /eda/summary
│   │       ├── schemas.py          # Schema distribusi kelas & korelasi
│   │       └── service.py          # Agregasi statistik dataset
│   └── shared/                     # Fungsi utilitas bersama
├── data/                           # Direktori dataset lokal
│   └── Dataset Phising Website.csv
├── models/                         # Direktori model ML serialisasi (.joblib / .pkl)
├── .env.example                    # Template konfigurasi environment
├── .gitignore                      # Git ignore standar Python, venv, & cache
├── requirements.txt                # Daftar dependensi Python
├── PRD.md                          # Dokumen Product Requirement Document
└── README.md                       # Dokumentasi teknis repositori
```

---

## 📋 Prasyarat Sistem

- Python 3.10 atau versi lebih baru
- `pip` package manager
- (Opsional) Virtual Environment (`venv` atau `conda`)

---

## ⚙️ Panduan Instalasi & Menjalankan

### 1. Clone Repositori & Masuk Direktori
```bash
git clone https://github.com/deteksi-web-phising/DWP-Backend.git
cd DWP-Backend
```

### 2. Buat dan Aktifkan Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Pasang Dependensi
```bash
pip install -r requirements.txt
```

### 4. Konfigurasi Environment
Salin template `.env.example` menjadi `.env`:
```bash
# Windows (PowerShell)
Copy-Item .env.example .env

# Linux / macOS
cp .env.example .env
```

### 5. Jalankan Development Server
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Server API akan aktif pada: `http://localhost:8000`

---

## 📖 Dokumentasi API Interaktif

Setelah server berjalan, Anda dapat mengakses antarmuka dokumentasi interaktif:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 🛣️ Daftar Endpoint Utama

| Method | Endpoint | Deskripsi |
| :--- | :--- | :--- |
| `GET` | `/` | Root endpoint / Status operasional API |
| `GET` | `/health` | Healthcheck endpoint untuk monitoring/container |
| `POST` | `/api/v1/detection/extract-url` | Mengekstrak 30 fitur struktural dari string URL |
| `POST` | `/api/v1/detection/predict` | Menjalankan inferensi model ML (Phishing vs Aman) |
| `GET` | `/api/v1/benchmark` | Mengambil data metrik komparasi 4 algoritma ML |
| `GET` | `/api/v1/eda/summary` | Mengambil data agregasi statistik & korelasi dataset |

---

## 👥 Tim Pengembang

- **Muhammad Fajrul**
- **Daffi Ahmad**
- **Panii Nayaka**

*Workshop Mesin Pembelajaran — Politeknik Elektronika Negeri Surabaya (PENS)*
