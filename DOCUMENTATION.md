# Dokumentasi Aplikasi Web Streamlit: Tubes Dasildat ML Platform

Dokumen ini berisi penjelasan detail mengenai struktur direktori, fungsi masing-masing file, logika pemrograman, serta petunjuk teknis penggunaan aplikasi prediksi Machine Learning untuk **Tugas Besar Dasar-Dasar Ilmu Data (Dasildat)**.

---

## 📁 Struktur Folder & File

Berikut adalah visualisasi struktur direktori proyek Anda yang telah disederhanakan menjadi satu halaman (*single page*):

```text
Tubes Dasildat/
│
├── app.py                      # File Utama Aplikasi (GUI & Logika Prediksi)
├── requirements.txt            # Daftar pustaka (dependencies) Python
├── DOCUMENTATION.md            # File dokumentasi ini (Panduan Proyek)
│
└── models/                     # Folder untuk menaruh model hasil ekspor dari Colab
    ├── README.md               # Panduan singkat penyimpanan model (.pkl / .joblib)
    └── modelJb_SVM-HPO.joblib  # Contoh file model SVM HPO yang Anda unggah
```

---

## 🛠️ Deskripsi Fungsi File & Folder

### 1. File Utama: `app.py`
File ini adalah jantung dari aplikasi Streamlit Anda. Semua antarmuka pengguna (GUI) dan alur logika kerja diprogram di sini. 
*   **Fungsi Utama:**
    *   Mengatur konfigurasi halaman Streamlit (Judul, Icon, dan Layout lebar).
    *   Menampilkan antarmuka pengunggahan file dataset CSV di bagian kiri halaman.
    *   Menyediakan dropdown pilihan untuk memilih 1 dari 8 variasi model (SVM, KNN, Decision Tree, Neural Network; baik tipe Baseline maupun HPO).
    *   Mencari file model di folder `models/` secara otomatis dengan kecocokan nama yang fleksibel (*case-insensitive*).
    *   Menyediakan tombol upload manual apabila file model tidak ditemukan di folder `models/`.
    *   Melakukan kalkulasi prediksi pada data baru menggunakan model `.joblib` / `.pkl` yang dimuat.
    *   Menyediakan tombol unduh (*download*) untuk mengekspor hasil prediksi ke file CSV baru.

### 2. File Dependensi: `requirements.txt`
Daftar pustaka Python pihak ketiga (*third-party libraries*) yang wajib diinstal agar aplikasi dapat berjalan tanpa error.
*   **Isi Dependensi:**
    *   `streamlit`: Framework utama untuk membuat antarmuka web.
    *   `pandas`: Digunakan untuk manipulasi data tabular (membaca CSV, membuat dataframe).
    *   `numpy`: Library komputasi numerik.
    *   `scikit-learn`: Digunakan untuk mendukung objek model Machine Learning (SVM, KNN, dll.) yang dimuat oleh `joblib`.
    *   `joblib`: Digunakan untuk memuat (*load*) dan membaca file serialisasi model.
    *   `matplotlib` & `seaborn`: Digunakan untuk visualisasi data jika diperlukan di masa depan.

### 3. Folder Penyimpanan Model: `models/`
Folder fisik yang digunakan sebagai repositori lokal untuk meletakkan file model hasil pelatihan di Google Colab.
*   **File di Dalamnya:**
    *   `README.md`: Berisi instruksi singkat mengenai jenis-jenis file model yang dapat dimasukkan beserta instruksi pemanggilannya di Python.
    *   `modelJb_SVM-HPO.joblib`: File nyata model SVM yang sudah di-tuning menggunakan Hyperparameter Optimization (HPO) di Colab.

---

## 🧠 Analisis Logika Kode `app.py`

### A. Pencarian File Model yang Fleksibel (`find_model_file`)
Untuk meminimalisir kesalahan penamaan file model, dibuat fungsi pemindai file:
```python
def find_model_file(patterns):
    # Memeriksa eksistensi folder models
    if not os.path.exists("models"):
        return None
    # Mengubah target pola pencarian ke huruf kecil dan standarisasi tanda hubung '-'
    normalized_patterns = [p.lower().replace("_", "-") for p in patterns]
    # Memindai seluruh isi folder models/
    for filename in os.listdir("models"):
        name_without_ext, ext = os.path.splitext(filename)
        if ext.lower() in ['.pkl', '.joblib']:
            normalized_name = name_without_ext.lower().replace("_", "-")
            if normalized_name in normalized_patterns:
                return filename
    return None
```
*   **Kenapa ini penting?** Jika Anda mengunggah file dengan variasi nama seperti `modelJb_SVM-HPO.joblib`, `svm_hpo.pkl`, atau `modeljb_svm_hpo.joblib`, fungsi ini tetap dapat mencocokkannya secara otomatis saat opsi **"SVM - HPO"** dipilih.

### B. Penyaringan Fitur Otomatis (`target_keywords`)
Saat dataset CSV dimuat, aplikasi secara otomatis mendeteksi kolom yang akan dijadikan input model. Pustaka ini secara cerdas mengecualikan kolom target/label dari pilihan default fitur untuk mencegah error prediksi:
```python
target_keywords = ['approved', 'target', 'label', 'y', 'class', 'is_safe', 'potability', 'status', 'outcome']
default_features = [col for col in all_columns if col.lower() not in target_keywords]
```
Ini memastikan kolom seperti `is_safe` tidak masuk ke dalam fitur input model secara tidak sengaja.

---

## 🚀 Petunjuk Menjalankan Proyek

### Langkah 1: Instalasi Library
Pastikan Anda sudah berada di terminal direktori proyek `Tubes Dasildat`, kemudian jalankan:
```bash
pip install -r requirements.txt
```

### Langkah 2: Jalankan Streamlit
Untuk memulai server lokal Streamlit, jalankan perintah berikut:
```bash
streamlit run app.py
```
Browser Anda akan otomatis terbuka dan mengarah ke alamat `http://localhost:8501`.

---

## 💡 Alur Kerja Integrasi Google Colab ke Streamlit

Agar model yang Anda latih di Google Colab dapat digunakan dengan sempurna di Streamlit, ikuti panduan berikut:

1.  **Gunakan Pipeline Scikit-Learn:**
    Di Google Colab, bungkus langkah preprocessing (seperti scaling/encoding) dan model ke dalam satu objek `Pipeline`:
    ```python
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    # Membuat pipeline
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(C=10, kernel='rbf'))
    ])
    model_pipeline.fit(X_train, y_train)
    ```
2.  **Simpan Menggunakan Joblib di Colab:**
    ```python
    import joblib
    joblib.dump(model_pipeline, 'modelJb_SVM-HPO.joblib')
    ```
3.  **Unduh & Letakkan di Folder `models/`:**
    Unduh file `.joblib` tersebut dari Colab lalu pindahkan ke folder `Tubes Dasildat/models/`. Aplikasi Streamlit akan langsung mengenali dan menggunakannya.
