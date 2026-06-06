import streamlit as st
import pandas as pd
import joblib
import os

# Configure the Streamlit page as a single-page app
st.set_page_config(
    page_title="Tubes Dasildat - Model Predictor",
    page_icon="🔮",
    layout="wide"
)

# App Title & Styling
st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>Tugas Besar Dasar-Dasar Ilmu Data</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #4B5563; font-weight: normal;'>Aplikasi Prediksi Data Klasifikasi (Baseline vs HPO)</h3>", unsafe_allow_html=True)
st.write("---")

# Dictionary mapping dropdown options to possible filename prefixes (case-insensitive matching)
MODEL_FILES_PATTERNS = {
    "SVM - Baseline": ["modeljb_svm-baseline", "svm_baseline", "modeljb_svm_baseline"],
    "SVM - HPO": ["modeljb_svm-hpo", "svm_hpo", "modeljb_svm_hpo"],
    "KNN - Baseline": ["modeljb_knn-baseline", "knn_baseline", "modeljb_knn_baseline"],
    "KNN - HPO": ["modeljb_knn-hpo", "knn_hpo", "modeljb_knn_hpo"],
    "Decision Tree - Baseline": ["modeljb_dt-baseline", "modeljb_decisiontree-baseline", "decision_tree_baseline", "dt_baseline"],
    "Decision Tree - HPO": ["modeljb_dt-hpo", "modeljb_decisiontree-hpo", "decision_tree_hpo", "dt_hpo"],
    "Neural Network - Baseline": ["modeljb_nn-baseline", "modeljb_neuralnetwork-baseline", "neural_network_baseline", "nn_baseline"],
    "Neural Network - HPO": ["modeljb_nn-hpo", "modeljb_neuralnetwork-hpo", "neural_network_hpo", "nn_hpo"]
}

def find_model_file(patterns):
    """
    Looks in the 'models' directory for any file with .pkl or .joblib extensions
    whose name matches the pattern list (case-insensitive and treating '-' and '_' as equivalent).
    """
    if not os.path.exists("models"):
        return None
        
    normalized_patterns = [p.lower().replace("_", "-") for p in patterns]
    
    try:
        for filename in os.listdir("models"):
            name_without_ext, ext = os.path.splitext(filename)
            if ext.lower() in ['.pkl', '.joblib']:
                normalized_name = name_without_ext.lower().replace("_", "-")
                if normalized_name in normalized_patterns:
                    return filename
    except Exception as e:
        st.error(f"Error scanning models directory: {e}")
        
    return None

# Layout with two main columns
col_data, col_model = st.columns([3, 2])

# Left column: Data loading
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
        
        # Display dataset dimensions
        st.caption(f"Dimensi data: {df.shape[0]} baris x {df.shape[1]} kolom")

# Right column: Model configuration
with col_model:
    st.subheader("🤖 2. Pilih Model & Konfigurasi")
    
    # Model dropdown
    selected_model_option = st.selectbox(
        "Pilih Model untuk Prediksi:",
        ["-- Pilih Model --"] + list(MODEL_FILES_PATTERNS.keys())
    )
    
    model = None
    model_loaded = False
    
    if selected_model_option != "-- Pilih Model --":
        patterns = MODEL_FILES_PATTERNS[selected_model_option]
        matched_filename = find_model_file(patterns)
        
        if matched_filename:
            model_path = os.path.join("models", matched_filename)
            try:
                model = joblib.load(model_path)
                model_loaded = True
                st.success(f"Berhasil memuat model: `{matched_filename}` dari folder `models/`")
            except Exception as e:
                st.error(f"Gagal memuat model `{matched_filename}`: {e}")
        else:
            # File not found in models/, offer manual upload
            default_suggested_name = f"modelJb_{selected_model_option.replace(' - ', '-').replace(' ', '')}.joblib"
            st.warning(f"File model untuk '{selected_model_option}' tidak ditemukan di folder `models/`.")
            st.write(f"Diharapkan memiliki nama seperti: `{default_suggested_name}` atau `{patterns[1]}.pkl`.")
            
            uploaded_model = st.file_uploader(f"Unggah file model manual (.pkl atau .joblib):", type=["pkl", "joblib"])
            
            if uploaded_model is not None:
                try:
                    model = joblib.load(uploaded_model)
                    model_loaded = True
                    st.success(f"Berhasil memuat model secara manual!")
                except Exception as e:
                    st.error(f"Gagal memuat model manual: {e}")


# Lower section for execution (if data and model are loaded)
if df is not None and model_loaded and model is not None:
    st.write("---")
    st.subheader("🔮 3. Jalankan Prediksi")
    
    all_columns = list(df.columns)
    
    # Try to automatically guess features (exclude target-like columns if present)
    target_keywords = ['approved', 'target', 'label', 'y', 'class', 'is_safe', 'potability', 'status', 'outcome']
    default_features = [col for col in all_columns if col.lower() not in target_keywords]
    
    col_feat1, col_feat2 = st.columns([3, 1])
    with col_feat1:
        selected_features = st.multiselect(
            "Pilih kolom fitur yang digunakan sebagai input model:",
            all_columns,
            default=default_features
        )
    
    with col_feat2:
        st.write("")
        st.write("")
        run_predict = st.button("✨ Lakukan Prediksi", use_container_width=True)
        
    if not selected_features:
        st.warning("Pilih setidaknya satu kolom fitur.")
    
    if run_predict and selected_features:
        try:
            # Get model input features
            X_input = df[selected_features]
            
            # Predict
            predictions = model.predict(X_input)
            
            # Append predictions to dataframe
            df_result = df.copy()
            df_result['Prediction_Result'] = predictions
            
            # Display results
            st.success("Prediksi berhasil diselesaikan!")
            st.dataframe(df_result, use_container_width=True)
            
            # Download button
            csv_data = df_result.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Unduh Hasil Prediksi (CSV)",
                data=csv_data,
                file_name=f"hasil_prediksi_{selected_model_option.replace(' - ', '_').lower()}.csv",
                mime="text/csv",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memprediksi: {e}")
            st.warning("💡 Pastikan kolom fitur yang dipilih sama persis urutan, nama, dan jenisnya saat Anda melatih model di Google Colab.")
else:
    if df is None:
        st.info("Silakan masukkan dataset CSV Anda terlebih dahulu di bagian kiri.")
    elif not model_loaded:
        st.info("Silakan pilih model terlebih dahulu di bagian kanan.")
