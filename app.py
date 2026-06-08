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
st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>Tugas Besar Dasar Ilmu Data</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #4B5563; font-weight: normal;'>Aplikasi Prediksi Keamanan Air Minum Berdasarkan Kandungan Kontaminan Menggunakan Algoritma Machine Learning</h3>", unsafe_allow_html=True)
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
    st.subheader("📁 1. Input Data")
    st.markdown("""
    <div style='background-color: #F8FAFC; padding: 12px; border-radius: 8px; border: 1px solid #E2E8F0; margin-bottom: 15px;'>
        <div style='font-size: 0.85rem; color: #475569; font-weight: bold; margin-bottom: 6px;'>📋 Fitur Input :</div>
        <div style='line-height: 1.8;'>
            <span style='background-color: #E2E8F0; color: #1E293B; font-size: 0.75rem; font-family: monospace; padding: 2px 6px; border-radius: 4px; display: inline-block; margin: 2px;'>aluminium</span>
            <span style='background-color: #E2E8F0; color: #1E293B; font-size: 0.75rem; font-family: monospace; padding: 2px 6px; border-radius: 4px; display: inline-block; margin: 2px;'>ammonia</span>
            <span style='background-color: #E2E8F0; color: #1E293B; font-size: 0.75rem; font-family: monospace; padding: 2px 6px; border-radius: 4px; display: inline-block; margin: 2px;'>arsenic</span>
            <span style='background-color: #E2E8F0; color: #1E293B; font-size: 0.75rem; font-family: monospace; padding: 2px 6px; border-radius: 4px; display: inline-block; margin: 2px;'>barium</span>
            <span style='background-color: #E2E8F0; color: #1E293B; font-size: 0.75rem; font-family: monospace; padding: 2px 6px; border-radius: 4px; display: inline-block; margin: 2px;'>cadmium</span>
            <span style='background-color: #E2E8F0; color: #1E293B; font-size: 0.75rem; font-family: monospace; padding: 2px 6px; border-radius: 4px; display: inline-block; margin: 2px;'>chloramine</span>
            <span style='background-color: #E2E8F0; color: #1E293B; font-size: 0.75rem; font-family: monospace; padding: 2px 6px; border-radius: 4px; display: inline-block; margin: 2px;'>chromium</span>
            <span style='background-color: #E2E8F0; color: #1E293B; font-size: 0.75rem; font-family: monospace; padding: 2px 6px; border-radius: 4px; display: inline-block; margin: 2px;'>copper</span>
            <span style='background-color: #E2E8F0; color: #1E293B; font-size: 0.75rem; font-family: monospace; padding: 2px 6px; border-radius: 4px; display: inline-block; margin: 2px;'>flouride</span>
            <span style='background-color: #E2E8F0; color: #1E293B; font-size: 0.75rem; font-family: monospace; padding: 2px 6px; border-radius: 4px; display: inline-block; margin: 2px;'>bacteria</span>
            <span style='background-color: #E2E8F0; color: #1E293B; font-size: 0.75rem; font-family: monospace; padding: 2px 6px; border-radius: 4px; display: inline-block; margin: 2px;'>viruses</span>
            <span style='background-color: #E2E8F0; color: #1E293B; font-size: 0.75rem; font-family: monospace; padding: 2px 6px; border-radius: 4px; display: inline-block; margin: 2px;'>lead</span>
            <span style='background-color: #E2E8F0; color: #1E293B; font-size: 0.75rem; font-family: monospace; padding: 2px 6px; border-radius: 4px; display: inline-block; margin: 2px;'>nitrates</span>
            <span style='background-color: #E2E8F0; color: #1E293B; font-size: 0.75rem; font-family: monospace; padding: 2px 6px; border-radius: 4px; display: inline-block; margin: 2px;'>nitrites</span>
            <span style='background-color: #E2E8F0; color: #1E293B; font-size: 0.75rem; font-family: monospace; padding: 2px 6px; border-radius: 4px; display: inline-block; margin: 2px;'>mercury</span>
            <span style='background-color: #E2E8F0; color: #1E293B; font-size: 0.75rem; font-family: monospace; padding: 2px 6px; border-radius: 4px; display: inline-block; margin: 2px;'>perchlorate</span>
            <span style='background-color: #E2E8F0; color: #1E293B; font-size: 0.75rem; font-family: monospace; padding: 2px 6px; border-radius: 4px; display: inline-block; margin: 2px;'>radium</span>
            <span style='background-color: #E2E8F0; color: #1E293B; font-size: 0.75rem; font-family: monospace; padding: 2px 6px; border-radius: 4px; display: inline-block; margin: 2px;'>selenium</span>
            <span style='background-color: #E2E8F0; color: #1E293B; font-size: 0.75rem; font-family: monospace; padding: 2px 6px; border-radius: 4px; display: inline-block; margin: 2px;'>silver</span>
            <span style='background-color: #E2E8F0; color: #1E293B; font-size: 0.75rem; font-family: monospace; padding: 2px 6px; border-radius: 4px; display: inline-block; margin: 2px;'>uranium</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Unggah file CSV atau Excel Anda:", type=["csv", "xlsx", "xls"])
    
    df = None
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.success("File berhasil dimuat!")
        except Exception as e:
            st.error(f"Error loading file: {e}")
            
    if df is not None:
        st.markdown("##### Preview Data:")
        st.dataframe(df, use_container_width=True)
        
        # Tampilkan dimensi dataset
        st.caption(f"Dimensi data: {df.shape[0]} baris x {df.shape[1]} kolom")

# Kolom kanan: Konfigurasi model
with col_model:
    st.subheader("🤖 2. Pilih Model")
    
    # Keterangan singkatan model
    st.markdown("""
    <div style='background-color: #F8FAFC; padding: 10px 12px; border-radius: 8px; border: 1px solid #E2E8F0; margin-top: -10px; margin-bottom: 15px;'>
        <div style='font-size: 0.8rem; color: #475569; font-weight: bold; margin-bottom: 4px;'>💡 Info  Algoritma:</div>
        <ul style='font-size: 0.75rem; color: #4B5563; margin-bottom: 0; padding-left: 18px;'>
            <li><b>SVM</b>: Support Vector Machine</li>
            <li><b>KNN</b>: K-Nearest Neighbors</li>
            <li><b>Decision Tree</b>: Pohon Keputusan</li>
            <li><b>Neural Network</b>: Jaringan Saraf Tiruan (MLP)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
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
            
            # hitung probabilitas jika didukung
            probabilities = None
            if hasattr(model, "predict_proba"):
                try:
                    probabilities = model.predict_proba(X_input).tolist()
                except Exception:
                    pass
            
            # Fallback ke decision_function (misal untuk SVM dengan probability=False)
            if probabilities is None and hasattr(model, "decision_function"):
                try:
                    decision_values = model.decision_function(X_input)
                    import math
                    
                    try:
                        vals = list(decision_values)
                    except TypeError:
                        vals = [decision_values]
                        
                    converted_probs = []
                    for val in vals:
                        # rumus Sigmoid: p = 1 / (1 + exp(-x))
                        p_safe = 1 / (1 + math.exp(-val))
                        converted_probs.append([1 - p_safe, p_safe])
                    probabilities = converted_probs
                except Exception:
                    pass
            
            # prediksi dataframe
            df_result = df.copy()
            df_result['Prediction_Result'] = predictions
            
            # menampilkan hasil
            st.success("Prediksi berhasil diselesaikan!")
            
            # Indikator Keamanan Air
            total_data = len(predictions)
            safe_count = int((predictions == 1).sum())
            unsafe_count = int((predictions == 0).sum())
            
            if total_data == 1:
                if predictions[0] == 1:
                    st.success("💧 **HASIL PREDIKSI: AIR AMAN UNTUK DIMINUM ✅**")
                else:
                    st.error("⚠️ **HASIL PREDIKSI: AIR TIDAK AMAN UNTUK DIMINUM ❌**")
                
                # Tampilkan probabilitas untuk single data
                if probabilities is not None:
                    prob_unsafe = probabilities[0][0] * 100
                    prob_safe = probabilities[0][1] * 100
                    st.markdown("### Probabilitas Prediksi")
                    st.markdown(f"- **Tidak Aman Diminum**: {prob_unsafe:.2f}%")
                    st.markdown(f"- **Aman untuk Diminum**: {prob_safe:.2f}%")
                else:
                    st.markdown("### Probabilitas Prediksi")
                    st.info("ℹ️ Model ini tidak mendukung probabilitas prediksi.")
            else:
                safe_pct = (safe_count / total_data) * 100
                unsafe_pct = 100 - safe_pct
                
                col_safe, col_unsafe = st.columns(2)
                with col_safe:
                    st.success(f"✅ **Aman Diminum**: {safe_count} sampel ({safe_pct:.1f}%)")
                with col_unsafe:
                    st.error(f"❌ **Tidak Aman Diminum**: {unsafe_count} sampel ({unsafe_pct:.1f}%)")
                
                # Tampilkan rata-rata probabilitas untuk data kelompok
                if probabilities is not None:
                    avg_prob_unsafe = sum([p[0] for p in probabilities]) / total_data * 100
                    avg_prob_safe = sum([p[1] for p in probabilities]) / total_data * 100
                    st.markdown("### Rata-rata Probabilitas Prediksi")
                    st.markdown(f"- **Tidak Aman Diminum**: {avg_prob_unsafe:.2f}%")
                    st.markdown(f"- **Aman untuk Diminum**: {avg_prob_safe:.2f}%")
            
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
            st.warning("💡 Pastikan kolom fitur pada dataset Anda sama persis urutan, nama, dan jenisnya saat Anda melatih model di Google Colab.")
else:
    if df is None:
        st.info("Silakan masukkan dataset CSV atau Excel Anda terlebih dahulu di bagian kiri.")
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
        
    st.write("---")
    
    # 5. Keterangan Hasil Prediksi
    st.markdown("**Keterangan Hasil Prediksi**")
    st.markdown("""
    <div style='background-color: #F8FAFC; padding: 10px 12px; border-radius: 6px; border: 1px solid #E2E8F0; font-size: 0.85rem; line-height: 1.5;'>
        <div style='color: #DC2626; font-weight: bold; margin-bottom: 6px;'>❌ 0 : Tidak Aman Diminum</div>
        <div style='color: #16A34A; font-weight: bold;'>✅ 1 : Aman untuk Diminum</div>
    </div>
    """, unsafe_allow_html=True)
