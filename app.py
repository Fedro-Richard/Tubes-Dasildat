import streamlit as st
import pandas as pd
import joblib
import os
import io


# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Tubes Dasildat - Model Predictor",
    page_icon="🔮",
    layout="wide"
)

# Judul 
st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>Tugas Besar Dasar-Dasar Ilmu Data</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #4B5563; font-weight: normal;'>Aplikasi Prediksi Data Klasifikasi</h3>", unsafe_allow_html=True)
st.write("---")

# Kamus yang memetakan opsi dropdown ke kemungkinan prefiks nama file (pencocokan tidak sensitif huruf besar/kecil)
MODEL_FILES_PATTERNS = {
    "SVM": ["modeljb_svm-hpo", "svm_hpo", "modeljb_svm_hpo", "modeljb_svm-baseline", "svm_baseline", "modeljb_svm_baseline", "modeljb_svm", "svm_model", "svm"],
    "KNN": ["hasilklasifikasi_waterquality1_knn-hpo", "modeljb_knn-hpo", "knn_hpo", "modeljb_knn_hpo", "modeljb_knn-baseline", "knn_baseline", "modeljb_knn_baseline", "modeljb_knn", "knn_model", "knn"],
    "Decision Tree": ["hasilklasifikasihpo-decisiontree_waterquality1", "modeljb_dt-hpo", "modeljb_decisiontree-hpo", "decision_tree_hpo", "dt_hpo", "modeljb_dt-baseline", "modeljb_decisiontree-baseline", "decision_tree_baseline", "dt_baseline", "modeljb_dt", "modeljb_decisiontree", "decision_tree_model", "dt_model", "decisiontree", "dt"],
    "Neural Network": ["modeljb_nn-hpo", "modeljb_neuralnetwork-hpo", "neural_network_hpo", "nn_hpo", "modeljb_nn-baseline", "modeljb_neuralnetwork-baseline", "neural_network_baseline", "nn_baseline", "modeljb_nn", "modeljb_neuralnetwork", "neural_network_model", "nn_model", "neuralnetwork", "nn"]
}

def find_model_file(patterns):
    """
    Mencari di direktori 'models' atau 'model' untuk setiap file dengan ekstensi .pkl atau .joblib
    yang namanya cocok dengan daftar pola (tidak sensitif huruf besar/kecil dan memperlakukan '-' dan '_' sebagai setara).
    Jika tidak ada kecocokan persis yang ditemukan, pencarian akan beralih ke pemeriksaan apakah nama file mengandung pola/kata kunci tersebut.
    """
    for folder in ["models", "model"]:
        if not os.path.exists(folder):
            continue
            
        normalized_patterns = [p.lower().replace("_", "-") for p in patterns]
        
        try:
            files = []
            for filename in os.listdir(folder):
                name_without_ext, ext = os.path.splitext(filename)
                if ext.lower() in ['.pkl', '.joblib']:
                    files.append((filename, name_without_ext))
            
            # Tahap pertama: coba pencocokan persis
            for filename, name_without_ext in files:
                normalized_name = name_without_ext.lower().replace("_", "-")
                if normalized_name in normalized_patterns:
                    return folder, filename
                    
            # Tahap kedua: coba pencocokan substring (cadangan)
            # Urutkan pola berdasarkan panjang secara menurun untuk mencocokkan pola yang paling spesifik terlebih dahulu
            sorted_patterns = sorted(normalized_patterns, key=len, reverse=True)
            for filename, name_without_ext in files:
                normalized_name = name_without_ext.lower().replace("_", "-")
                for pat in sorted_patterns:
                    if len(pat) >= 3:
                        if pat in normalized_name:
                            return folder, filename
                    else:
                        # Untuk pola yang sangat pendek seperti 'dt' or 'nn', cocokkan bagian yang dipisah oleh '-'
                        parts = normalized_name.split('-')
                        if pat in parts:
                            return folder, filename
        except Exception as e:
            st.error(f"Error scanning {folder} directory: {e}")
            
    return None, None

# Tata letak dengan dua kolom utama
col_data, col_model = st.columns([3, 2])

# Kolom kiri: Pemuatan data
with col_data:
    st.subheader("📁 1. Input Dataset")
    uploaded_file = st.file_uploader("Unggah file CSV Anda:", type=["csv"])
    
    df = None
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("File CSV berhasil dimuat!")
        except Exception as e:
            st.error(f"Error loading CSV file: {e}")
            
    if df is not None:
        st.markdown("##### Preview Data:")
        st.dataframe(df.head(8), use_container_width=True)
        
        # Tampilkan dimensi dataset
        st.caption(f"Dimensi data: {df.shape[0]} baris x {df.shape[1]} kolom")

# Kolom kanan: Konfigurasi model
with col_model:
    st.subheader("🤖 2. Pilih Model & Konfigurasi")
    
    # Dropdown model
    selected_model_option = st.selectbox(
        "Pilih Model untuk Prediksi:",
        ["-- Pilih Model --"] + list(MODEL_FILES_PATTERNS.keys())
    )
    
    model = None
    model_loaded = False
    
    if selected_model_option != "-- Pilih Model --":
        patterns = MODEL_FILES_PATTERNS[selected_model_option]
        matched_folder, matched_filename = find_model_file(patterns)
        
        if matched_filename:
            model_path = os.path.join(matched_folder, matched_filename)
            try:
                model = joblib.load(model_path)
                model_loaded = True
                st.success(f"Berhasil memuat model: `{matched_filename}` dari folder `{matched_folder}/`")
            except Exception as e:
                st.error(f"Gagal memuat model `{matched_filename}`: {e}")
        else:
            # File tidak ditemukan di folder models/ atau model/, tawarkan unggah manual
            default_suggested_name = f"modelJb_{selected_model_option.replace(' ', '')}.joblib"
            st.warning(f"File model untuk '{selected_model_option}' tidak ditemukan di folder `models/` atau `model/`.")
            st.write(f"Diharapkan memiliki nama seperti: `{default_suggested_name}` atau `{patterns[1]}.pkl`.")
            
            uploaded_model = st.file_uploader(f"Unggah file model manual (.pkl atau .joblib):", type=["pkl", "joblib"])
            
            if uploaded_model is not None:
                try:
                    model = joblib.load(uploaded_model)
                    model_loaded = True
                    st.success(f"Berhasil memuat model secara manual!")
                except Exception as e:
                    st.error(f"Gagal memuat model manual: {e}")


# Bagian bawah untuk eksekusi (jika data dan model telah dimuat)
if df is not None and model_loaded and model is not None:
    st.write("---")
    st.subheader("🔮 3. Jalankan Prediksi")
    
    all_columns = list(df.columns)
    
    # Coba tebak fitur secara otomatis (kecualikan kolom seperti target jika ada)
    target_keywords = ['approved', 'target', 'label', 'y', 'class', 'is_safe', 'potability', 'status', 'outcome']
    
    # Periksa apakah model memiliki atribut feature_names_in_ dari pelatihan
    if hasattr(model, 'feature_names_in_'):
        model_features = list(model.feature_names_in_)
        if all(col in all_columns for col in model_features):
            selected_features = model_features
        else:
            selected_features = [col for col in all_columns if col.lower() not in target_keywords]
    else:
        selected_features = [col for col in all_columns if col.lower() not in target_keywords]
    
    run_predict = st.button("✨ Lakukan Prediksi", use_container_width=True)
        
    if not selected_features:
        st.warning("Tidak ada kolom fitur yang dideteksi untuk input model.")
    
    if run_predict and selected_features:
        try:
            # fitur model
            X_input = df[selected_features]
            
            # prediksi model
            predictions = model.predict(X_input)
            
            # prediksi dataframe
            df_result = df.copy()
            df_result['Prediction_Result'] = predictions
            
            # menampilkan hasil
            st.success("Prediksi berhasil diselesaikan!")
            st.dataframe(df_result, use_container_width=True)
            
            # Tombol unduh
            try:
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df_result.to_excel(writer, index=False, sheet_name='Hasil Prediksi')
                excel_data = output.getvalue()
                
                st.download_button(
                    label="📥 Unduh Hasil Prediksi (Excel)",
                    data=excel_data,
                    file_name=f"hasil_prediksi_{selected_model_option.replace(' - ', '_').lower()}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            except ImportError:
                st.error("Pustaka 'openpyxl' tidak ditemukan. Harap instal dengan menjalankan `pip install openpyxl` di terminal Anda.")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memprediksi: {e}")
            st.warning("💡 Pastikan kolom fitur pada dataset CSV Anda sama persis urutan, nama, dan jenisnya saat Anda melatih model di Google Colab.")
else:
    if df is None:
        st.info("Silakan masukkan dataset CSV Anda terlebih dahulu di bagian kiri.")
    elif not model_loaded:
        st.info("Silakan pilih model terlebih dahulu di bagian kanan.")

# --- SIDEBAR PANEL INFORMASI ---
with st.sidebar:
    st.markdown("### 📊 Ringkasan Prediksi")
    st.markdown("---")
    
    # 1. Informasi Dataset
    st.markdown("**Dataset**")
    if uploaded_file is not None:
        dataset_name = os.path.splitext(uploaded_file.name)[0].replace("_", " ").title()
        st.write(dataset_name)
    else:
        st.info("Belum diunggah")
        
    st.write("") # spasi
    
    # 2. Informasi Algoritma
    st.markdown("**Algoritma**")
    if selected_model_option != "-- Pilih Model --":
        st.write(selected_model_option)
    else:
        st.info("Belum dipilih")
        
    st.write("") # spasi
    
    # 3. Jumlah Data
    st.markdown("**Jumlah Data**")
    if df is not None:
        st.write(f"{df.shape[0]:,} data/instance")
    else:
        st.write("-")
        
    st.write("") # Spasi
    
    # 4. Jumlah Fitur
    st.markdown("**Jumlah Fitur**")
    if 'selected_features' in locals() and selected_features:
        st.write(f"{len(selected_features)} fitur")
    elif df is not None:
        target_keywords = ['approved', 'target', 'label', 'y', 'class', 'is_safe', 'potability', 'status', 'outcome']
        est_features = [col for col in df.columns if col.lower() not in target_keywords]
        st.write(f"{len(est_features)} fitur")
    else:
        st.write("-")
