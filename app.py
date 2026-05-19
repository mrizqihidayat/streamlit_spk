import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Mall Customer Segmentation", page_icon="🛍️")

# 2. Fungsi Load Model
@st.cache_resource
def load_model():
    try:
        model = joblib.load("logistic_regression_pipeline.pkl")
        return model
    except:
        return None

model = load_model()

# 3. Header
st.title("🤖 Mall Customer Segmentation")
st.write("Aplikasi ini memprediksi segmen pelanggan berdasarkan profil demografi dan perilaku belanja.")

if model is None:
    st.error("File 'logistic_regression_pipeline.pkl' tidak ditemukan. Pastikan file model ada di folder yang sama.")
else:
    # 4. Input User (Sidebar atau Utama)
    st.sidebar.header("Input Data Pelanggan")
    
    gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    age = st.sidebar.number_input("Usia (Tahun)", min_value=1, max_value=100, value=30)
    income = st.sidebar.number_input("Pendapatan Tahunan (k$)", min_value=1, max_value=200, value=50)
    score = st.sidebar.number_input("Skor Pengeluaran (1-100)", min_value=1, max_value=100, value=50)

    # 5. Hitung Fitur Tambahan (Feature Engineering dari Colab)
    # Di Colab: Propensity to Spend = Spending Score / Annual Income
    propensity = round(score / income, 2) if income > 0 else 0

    # 6. Preprocessing Input
    gender_encoded = 0 if gender == "Female" else 1
    
    # Urutan fitur harus sama dengan saat training di Colab: 
    # [Genre, Age, Annual Income, Spending Score, Propensity to Spend]
    features = np.array([[gender_encoded, age, income, score, propensity]])

    # 7. Tombol Prediksi
    if st.sidebar.button("Analisis Segmen"):
        prediction = model.predict(features)[0]
        
        # Mapping Deskripsi Cluster berdasarkan hasil analisa di Colab Anda
        cluster_info = {
            0: "Pria Tua, Pendapatan Sedang, Pengeluaran Rendah (Hemat).",
            1: "Wanita Dewasa, Pendapatan Tinggi, Pengeluaran Sangat Rendah (Sangat Hemat).",
            2: "Anak Muda (Dominan Wanita), Pendapatan Rendah, Pengeluaran Tinggi (Impulsive Buyer).",
            3: "Wanita Muda/Dewasa, Pendapatan Tinggi, Pengeluaran Tinggi (VIP Customer).",
            4: "Wanita Muda, Pendapatan Sedang, Pengeluaran Sedang (Rata-rata).",
            5: "Pria Muda, Pendapatan Tinggi, Pengeluaran Tinggi (Lifestyle Spender).",
            6: "Wanita Tua, Pendapatan Sedang, Pengeluaran Rendah (Konservatif).",
            7: "Pria Dewasa, Pendapatan Tinggi, Pengeluaran Sangat Rendah (Sangat Hemat)."
        }

        # Tampilkan Hasil
        st.subheader("Hasil Analisis")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Segmen Pelanggan", f"Cluster {prediction}")
        with col2:
            st.metric("Propensity to Spend", propensity)

        st.info(f"**Karakteristik Segmen:** {cluster_info.get(prediction, 'Segmen tidak dikenal')}")
        
        # Detail Input
        with st.expander("Lihat Detail Data Input"):
            st.write(pd.DataFrame(features, columns=['Genre', 'Age', 'Annual Income', 'Spending Score', 'Propensity']))
    else:
        st.write("Silakan masukkan data pelanggan di bilah samping dan klik tombol **Analisis Segmen**.")

# Footer
st.markdown("---")
st.caption("Model didasarkan pada algoritma K-Means & Logistic Regression (Scikit-Learn)")