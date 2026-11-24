# E-Commerce Public Dataset Analysis & Dashboard ✨

Proyek akhir kelas **Belajar Fundamental Analisis Data** (Dicoding) oleh **Muhammad Dikaisa Ibnu Amin**.

Analisis dilakukan pada pesanan berstatus *delivered* periode **Januari 2017 – Agustus 2018** untuk menjawab tiga pertanyaan bisnis:
1. Kategori produk dengan revenue tertinggi dan terendah.
2. Tren jumlah pesanan dan revenue bulanan.
3. Pengaruh keterlambatan pengiriman terhadap skor review, serta state dengan keterlambatan tertinggi.

Analisis lanjutan: **RFM Analysis**, **clustering dengan manual grouping**, dan **geospatial analysis**.

## Struktur Proyek
```
submission
├── dashboard
│   ├── dashboard.py
│   ├── main_data.csv
│   └── geo_state.csv
├── data
│   └── *.csv (dataset asli)
├── notebook.ipynb
├── README.md
├── requirements.txt
└── url.txt
```

## Setup Environment - Anaconda
```
conda create --name main-ds python=3.11
conda activate main-ds
pip install -r requirements.txt
```

## Setup Environment - Shell/Terminal
```
mkdir proyek_analisis_data
cd proyek_analisis_data
pipenv install
pipenv shell
pip install -r requirements.txt
```

## Run Streamlit App
Jalankan dari folder utama proyek (`submission`):
```
streamlit run dashboard/dashboard.py
```
Dashboard akan terbuka di browser pada `http://localhost:8501`.

## Menjalankan Notebook
Buka `notebook.ipynb` di Jupyter Notebook atau Google Colab. Pastikan folder `data/` berada di direktori yang sama dengan notebook (jika memakai Colab, upload folder `data/` terlebih dahulu), lalu jalankan semua cell (*Run All*).
