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

st.markdown("""
<style>
.card {border-radius: 14px; padding: 1.1rem 1.4rem; border: 1px solid rgba(128,128,128,.25);
       border-left-width: 6px;}
.card .lvl {font-size: .8rem; letter-spacing: .08em; text-transform: uppercase; opacity: .75;}
.card .big {font-size: 2.8rem; font-weight: 700; line-height: 1.15;}
.card .lbl {font-size: 1.05rem; font-weight: 600;}
.track {position: relative; height: 12px; border-radius: 999px; background: rgba(128,128,128,.25);
        margin: .9rem 0 .35rem;}
.fill {height: 100%; border-radius: 999px;}
.thr {position: absolute; top: -5px; width: 2px; height: 22px; background: currentColor; opacity: .55;}
.small {font-size: .8rem; opacity: .65;}
</style>
""", unsafe_allow_html=True)

st.title("📊 Dashboard Prediksi Customer Churn")
st.write("Prediksi pelanggan menggunakan Random Forest.")

# Memuat model
@st.cache_resource
def load_models():
    path = Path(__file__).parent / "model_customer_churn.joblib"
    return joblib.load(path)

bundle = load_models()
feature_columns = bundle["feature_columns"]
model = bundle["rf_model"]

st.success("Model berhasil dimuat!")

# Batas keputusan Random Forest: churn jika probabilitas >= 50%
BATAS = 0.5
BATAS_WASPADA = 0.35

# Nilai awal form dan contoh pelanggan
DEFAULT = dict(
    gender="Female", senior="No", partner="No", dependents="No", tenure=12,
    phone="No", multiple="No", internet="DSL", monthly=70.0, total=840.0,
    contract="Month-to-month", payment="Electronic check", paperless="No",
    OnlineSecurity="No", OnlineBackup="No", DeviceProtection="No",
    TechSupport="No", StreamingTV="No", StreamingMovies="No"
)

CONTOH_CHURN = dict(
    tenure=2, phone="Yes", multiple="No", internet="Fiber optic",
    monthly=85.0, total=170.0, contract="Month-to-month",
    payment="Electronic check", paperless="Yes"
)

CONTOH_TIDAK_CHURN = dict(
    gender="Male", partner="Yes", dependents="Yes", tenure=60, phone="Yes",
    multiple="Yes", internet="DSL", monthly=65.0, total=3900.0,
    contract="Two year", payment="Bank transfer (automatic)", paperless="No",
    OnlineSecurity="Yes", OnlineBackup="Yes", DeviceProtection="Yes",
    TechSupport="Yes"
)

for k, nilai in DEFAULT.items():
    st.session_state.setdefault(k, nilai)


def isi_form(contoh):
    st.session_state.update({**DEFAULT, **contoh})


st.subheader("Masukkan Data Pelanggan")

st.write("Isi form secara manual, atau mulai dari contoh:")
b1, b2, b3, _ = st.columns([1.3, 1.5, 0.7, 2])
b1.button("Contoh pelanggan churn", on_click=isi_form, args=(CONTOH_CHURN,))
b2.button("Contoh pelanggan tidak churn", on_click=isi_form, args=(CONTOH_TIDAK_CHURN,))
b3.button("Reset", on_click=isi_form, args=({},))

# Informasi pelanggan
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Jenis Kelamin", ["Female", "Male"], key="gender")
    senior = st.selectbox("Lansia", ["No", "Yes"], key="senior")
    partner = st.selectbox("Memiliki Pasangan", ["No", "Yes"], key="partner")
    dependents = st.selectbox("Memiliki Tanggungan", ["No", "Yes"], key="dependents")

    tenure = st.number_input(
        "Lama Berlangganan (Bulan)",
        min_value=0, max_value=100, key="tenure"
    )

    phone = st.selectbox("Layanan Telepon", ["No", "Yes"], key="phone")

    if phone == "Yes":
        multiple = st.selectbox("Multiple Lines", ["No", "Yes"], key="multiple")
    else:
        multiple = "No phone service"

with col2:
    internet = st.selectbox(
        "Jenis Layanan Internet",
        ["DSL", "Fiber optic", "No"],
        key="internet"
    )

    monthly = st.number_input(
        "Biaya Bulanan",
        min_value=0.0, key="monthly"
    )

    total = st.number_input(
        "Total Biaya",
        min_value=0.0, key="total"
    )

    contract = st.selectbox(
        "Jenis Kontrak",
        ["Month-to-month", "One year", "Two year"],
        key="contract"
    )

    payment = st.selectbox(
        "Metode Pembayaran",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ],
        key="payment"
    )

    paperless = st.selectbox(
        "Tagihan Digital",
        ["No", "Yes"],
        key="paperless"
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
            layanan[nama] = st.selectbox(label, ["No", "Yes"], key=nama)

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

    # Melakukan prediksi
    hasil = int(model.predict(input_df)[0])
    proba = float(model.predict_proba(input_df)[:, 1][0])

    # Menentukan tampilan hasil
    if hasil == 1:
        judul, rgb = "Berisiko churn", (229, 72, 77)
        saran = ("Pelanggan berpotensi berhenti berlangganan. Pertimbangkan penawaran retensi, "
                 "misalnya kontrak jangka panjang, diskon, atau layanan tambahan.")
    elif proba >= BATAS_WASPADA:
        judul, rgb = "Waspada", (245, 165, 36)
        saran = ("Pelanggan diprediksi tidak churn, tetapi probabilitasnya mendekati batas. "
                 "Sebaiknya dipantau.")
    else:
        judul, rgb = "Aman", (48, 164, 108)
        saran = "Pelanggan diprediksi tetap berlangganan. Pertahankan kualitas layanan."

    warna = "rgb({},{},{})".format(*rgb)
    latar = "rgba({},{},{},0.10)".format(*rgb)

    st.subheader("Hasil Prediksi")

    st.markdown(
        f'<div class="card" style="border-left-color:{warna}; background:{latar};">'
        f'<div class="lvl">Probabilitas churn</div>'
        f'<div class="big">{proba:.0%}</div>'
        f'<div class="lbl" style="color:{warna};">{judul}</div>'
        f'<div class="track"><div class="fill" style="width:{min(proba, 1) * 100:.1f}%; '
        f'background:{warna};"></div><div class="thr" style="left:{BATAS * 100:.0f}%;"></div></div>'
        f'<div class="small">Garis tegak = batas keputusan {BATAS:.0%}. '
        f'Di kanan garis berarti diprediksi churn.</div>'
        f'</div>', unsafe_allow_html=True)

    st.markdown(f"**Saran:** {saran}")
    st.caption("Probabilitas churn = seberapa yakin model bahwa pelanggan ini akan berhenti berlangganan.")

    with st.expander("Lihat data hasil preprocessing"):
        st.dataframe(input_df)

st.caption("Hasil prediksi merupakan estimasi model, bukan kepastian.")
