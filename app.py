import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Mall Customer Segmentation", page_icon="🛍️")

@st.cache_resource
def load_model():
    try:
        model = joblib.load("logistic_regression_pipeline.pkl")
        return model
    except:
        return None

model = load_model()

st.title("Mall Customer Segmentation")
st.write("Aplikasi ini memprediksi segmen pelanggan berdasarkan profil demografi dan perilaku belanja.")

if model is None:
    st.error("File 'logistic_regression_pipeline.pkl' tidak ditemukan. Pastikan file model ada di folder yang sama.")
else:
    st.sidebar.header("Input Data Pelanggan")
    
    gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    age = st.sidebar.number_input("Usia (Tahun)", min_value=1, max_value=100, value=30)
    income = st.sidebar.number_input("Pendapatan Tahunan (k$)", min_value=1, max_value=200, value=50)
    score = st.sidebar.number_input("Skor Pengeluaran (1-100)", min_value=1, max_value=100, value=50)
    propensity = round(score / income, 2) if income > 0 else 0

    gender_map = {
        'Female': 0,
        'Male': 1
    }

    gender_encoded = gender_map[gender]
    
    features = np.array([[gender_encoded, age, income, score, propensity]])

    if st.sidebar.button("Analisis Segmen"):
        prediction = model.predict(features)[0]
        
        cluster_info = {
            0: "Pria Senior dengan Pola Belanja Moderat",
            1: "Pelanggan Muda dengan Pengeluaran Seimbang",
            2: "Anak Muda Konsumtif Berpenghasilan Rendah",
            3: "Pelanggan Premium/VIP",
            4: "Berpenghasilan Tinggi dengan Minat Belanja Rendah",
            5: "Pria Muda Konsumtif",
            6: "Wanita Senior Konservatif"
        }

        st.subheader("Hasil Analisis")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Segmen Pelanggan", f"Cluster {prediction}")
        with col2:
            st.metric("Propensity to Spend", propensity)

        st.info(f"**Karakteristik Segmen:** {cluster_info.get(prediction, 'Segmen tidak dikenal')}")
        
        with st.expander("Lihat Detail Data Input"):
            st.write(pd.DataFrame(features, columns=['Genre', 'Age', 'Annual Income', 'Spending Score', 'Propensity']))
    else:
        st.write("Silakan masukkan data pelanggan di bilah samping dan klik tombol **Analisis Segmen**.")

st.markdown("---")
st.caption("Model didasarkan pada algoritma K-Means (k=7) & Logistic Regression (Scikit-Learn)")