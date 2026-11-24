
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Pengaturan halaman
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Dashboard Prediksi Customer Churn")
st.write("Prediksi pelanggan menggunakan KNN dan Random Forest.")

# Memuat model
@st.cache_resource
def load_models():
    path = Path(__file__).parent / "model_customer_churn.joblib"
    return joblib.load(path)

bundle = load_models()
feature_columns = bundle["feature_columns"]

st.success("Model berhasil dimuat!")

# Memilih algoritma
algoritma = st.selectbox(
    "Pilih Algoritma",
    ["KNN", "Random Forest"]
)

st.subheader("Masukkan Data Pelanggan")

# Informasi pelanggan
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Jenis Kelamin", ["Female", "Male"])
    senior = st.selectbox("Lansia", ["No", "Yes"])
    partner = st.selectbox("Memiliki Pasangan", ["No", "Yes"])
    dependents = st.selectbox("Memiliki Tanggungan", ["No", "Yes"])

    tenure = st.number_input(
        "Lama Berlangganan (Bulan)",
        min_value=0, max_value=100, value=12
    )

    phone = st.selectbox("Layanan Telepon", ["No", "Yes"])

    if phone == "Yes":
        multiple = st.selectbox("Multiple Lines", ["No", "Yes"])
    else:
        multiple = "No phone service"

with col2:
    internet = st.selectbox(
        "Jenis Layanan Internet",
        ["DSL", "Fiber optic", "No"]
    )

    monthly = st.number_input(
        "Biaya Bulanan",
        min_value=0.0, value=70.0
    )

    total = st.number_input(
        "Total Biaya",
        min_value=0.0, value=840.0
    )

    contract = st.selectbox(
        "Jenis Kontrak",
        ["Month-to-month", "One year", "Two year"]
    )

    payment = st.selectbox(
        "Metode Pembayaran",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    paperless = st.selectbox(
        "Tagihan Digital",
        ["No", "Yes"]
    )

# Layanan tambahan
st.subheader("Layanan Tambahan")

layanan = {}

nama_layanan = {
    "OnlineSecurity": "Keamanan Online",
    "OnlineBackup": "Backup Online",
    "DeviceProtection": "Perlindungan Perangkat",
    "TechSupport": "Dukungan Teknis",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming Film"
}

c1, c2 = st.columns(2)

if internet == "No":
    for nama in nama_layanan:
        layanan[nama] = "No internet service"
    st.info("Layanan tambahan tidak tersedia tanpa layanan internet.")
else:
    for i, (nama, label) in enumerate(nama_layanan.items()):
        with (c1 if i % 2 == 0 else c2):
            layanan[nama] = st.selectbox(label, ["No", "Yes"])

# Tombol Prediksi
if st.button("🔍 Prediksi Churn", type="primary"):

    # Membuat 40 fitur awal bernilai 0
    data = {kolom: 0 for kolom in feature_columns}

    # Encoding fitur biner sesuai preprocessing
    data["gender"] = 1 if gender == "Male" else 0
    data["SeniorCitizen"] = 1 if senior == "Yes" else 0
    data["Partner"] = 1 if partner == "Yes" else 0
    data["Dependents"] = 1 if dependents == "Yes" else 0
    data["PhoneService"] = 1 if phone == "Yes" else 0
    data["PaperlessBilling"] = 1 if paperless == "Yes" else 0

    # Fitur numerik
    data["tenure"] = tenure
    data["MonthlyCharges"] = monthly
    data["TotalCharges"] = total

    # Encoding fitur multi-kategori
    kategori = {
        "MultipleLines": multiple,
        "InternetService": internet,
        "Contract": contract,
        "PaymentMethod": payment,
        **layanan
    }

    for nama, nilai in kategori.items():
        kolom = f"{nama}_{nilai}"
        if kolom not in data:
            st.error(f"Fitur tidak ditemukan: {kolom}")
            st.stop()
        data[kolom] = 1

    # Membuat dataframe dengan urutan fitur asli
    input_df = pd.DataFrame([data], columns=feature_columns)

    # Memilih model
    if algoritma == "KNN":
        model = bundle["knn_model"]
    else:
        model = bundle["rf_model"]

    # Melakukan prediksi
    hasil = int(model.predict(input_df)[0])

    st.subheader("Hasil Prediksi")

    if hasil == 1:
        st.warning("⚠️ Pelanggan diprediksi CHURN")
        st.write("Pelanggan berpotensi berhenti berlangganan.")
    else:
        st.success("✅ Pelanggan diprediksi TIDAK CHURN")
        st.write("Pelanggan diprediksi tetap berlangganan.")

    st.info(f"Algoritma yang digunakan: {algoritma}")

    with st.expander("Lihat data hasil preprocessing"):
        st.dataframe(input_df)

st.caption("Hasil prediksi merupakan estimasi model, bukan kepastian.")
