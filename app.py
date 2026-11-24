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
.block-container {max-width: 1100px;}
.hero p {opacity: .7; margin-top: -.4rem;}
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
.tag {display: inline-block; padding: .15rem .6rem; margin: .15rem .25rem .15rem 0; border-radius: 999px;
      font-size: .82rem; border: 1px solid rgba(128,128,128,.35);}
.imp-row {display: flex; align-items: center; gap: .6rem; margin: .25rem 0; font-size: .85rem;}
.imp-name {width: 230px; flex-shrink: 0;}
.imp-track {flex: 1; height: 10px; border-radius: 999px; background: rgba(128,128,128,.2);}
.imp-fill {height: 100%; border-radius: 999px; background: rgb(59,110,246);}
.imp-val {width: 52px; text-align: right; opacity: .7;}
</style>
""", unsafe_allow_html=True)

# Memuat model
@st.cache_resource
def load_models():
    path = Path(__file__).parent / "model_customer_churn.joblib"
    return joblib.load(path)


bundle = load_models()
model = bundle["rf_model"]
feature_columns = bundle["feature_columns"]

# Batas keputusan Random Forest (predict() memakai probabilitas churn >= 50%)
BATAS = 0.5
BATAS_WASPADA = 0.35

st.title("📊 Dashboard Prediksi Customer Churn")
st.markdown('<div class="hero"><p>Prediksi pelanggan berisiko berhenti berlangganan '
            'menggunakan algoritma Random Forest.</p></div>', unsafe_allow_html=True)


def buat_fitur(v):
    """Ubah input pengguna menjadi 40 fitur numerik sesuai preprocessing saat pelatihan."""
    data = {kolom: 0 for kolom in feature_columns}

    # Encoding fitur biner
    data["gender"] = 1 if v["gender"] == "Male" else 0
    data["SeniorCitizen"] = 1 if v["senior"] == "Yes" else 0
    data["Partner"] = 1 if v["partner"] == "Yes" else 0
    data["Dependents"] = 1 if v["dependents"] == "Yes" else 0
    data["PhoneService"] = 1 if v["phone"] == "Yes" else 0
    data["PaperlessBilling"] = 1 if v["paperless"] == "Yes" else 0

    # Fitur numerik
    data["tenure"] = v["tenure"]
    data["MonthlyCharges"] = v["monthly"]
    data["TotalCharges"] = v["total"]

    # Encoding fitur multi-kategori (one-hot)
    kategori = {
        "MultipleLines": v["multiple"],
        "InternetService": v["internet"],
        "Contract": v["contract"],
        "PaymentMethod": v["payment"],
        **v["layanan"]
    }
    for nama, nilai in kategori.items():
        kolom = f"{nama}_{nilai}"
        if kolom not in data:
            raise ValueError(f"Fitur tidak ditemukan: {kolom}")
        data[kolom] = 1

    return pd.DataFrame([data], columns=feature_columns)


def proba_churn(v):
    return float(model.predict_proba(buat_fitur(v))[:, 1][0])


tab_prediksi, tab_baca = st.tabs(["Prediksi", "Cara membaca hasil"])

# ======================================================== TAB PREDIKSI
with tab_prediksi:
    st.subheader("Masukkan Data Pelanggan")

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

        phone = st.selectbox("Layanan Telepon", ["No", "Yes"], index=1)

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

        total_otomatis = st.checkbox(
            "Hitung Total Biaya otomatis (lama berlangganan × biaya bulanan)",
            value=True
        )
        if total_otomatis:
            total = float(tenure * monthly)
            st.caption(f"Total Biaya: {total:,.2f}")
        else:
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

        paperless = st.selectbox("Tagihan Digital", ["No", "Yes"])

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

        v = dict(gender=gender, senior=senior, partner=partner, dependents=dependents,
                 tenure=tenure, phone=phone, multiple=multiple, internet=internet,
                 monthly=monthly, total=total, contract=contract, payment=payment,
                 paperless=paperless, layanan=layanan)

        try:
            input_df = buat_fitur(v)
        except ValueError as e:
            st.error(str(e))
            st.stop()

        hasil = int(model.predict(input_df)[0])
        proba = float(model.predict_proba(input_df)[:, 1][0])

        if hasil == 1:
            judul, rgb = "Berisiko churn", (229, 72, 77)
            saran = ("Pelanggan berpotensi berhenti berlangganan. Prioritaskan untuk program retensi: "
                     "tawarkan kontrak 1-2 tahun, diskon, atau paket layanan tambahan "
                     "(Dukungan Teknis / Keamanan Online).")
        elif proba >= BATAS_WASPADA:
            judul, rgb = "Waspada", (245, 165, 36)
            saran = ("Diprediksi tidak churn, tetapi probabilitasnya cukup dekat dengan batas. "
                     "Pantau pelanggan ini dan pertimbangkan penawaran ringan.")
        else:
            judul, rgb = "Aman", (48, 164, 108)
            saran = "Pelanggan diprediksi tetap berlangganan. Pertahankan kualitas layanan."

        warna = "rgb({},{},{})".format(*rgb)
        latar = "rgba({},{},{},0.10)".format(*rgb)

        st.divider()
        st.subheader("Hasil Prediksi")

        kiri, kanan = st.columns([1.15, 1])

        with kiri:
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

        with kanan:
            ada_internet = internet != "No"
            risiko, pelindung = [], []
            if contract == "Month-to-month":
                risiko.append("Kontrak bulanan")
            else:
                pelindung.append(f"Kontrak {contract}")
            if tenure <= 12:
                risiko.append(f"Pelanggan baru ({tenure} bulan)")
            elif tenure > 24:
                pelindung.append(f"Pelanggan lama ({tenure} bulan)")
            if internet == "Fiber optic":
                risiko.append("Internet Fiber optic")
            if payment == "Electronic check":
                risiko.append("Bayar Electronic check")
            if "automatic" in payment:
                pelindung.append("Pembayaran otomatis")
            if ada_internet and layanan["TechSupport"] == "No":
                risiko.append("Tanpa Dukungan Teknis")
            if ada_internet and layanan["TechSupport"] == "Yes":
                pelindung.append("Memakai Dukungan Teknis")
            if ada_internet and layanan["OnlineSecurity"] == "No":
                risiko.append("Tanpa Keamanan Online")
            if ada_internet and layanan["OnlineSecurity"] == "Yes":
                pelindung.append("Memakai Keamanan Online")

            def tags(daftar):
                return "".join(f'<span class="tag">{x}</span>' for x in daftar) or '<span class="small">-</span>'

            st.markdown("**Tanda yang cenderung menaikkan risiko**")
            st.markdown(tags(risiko), unsafe_allow_html=True)
            st.markdown("**Tanda yang cenderung menurunkan risiko**")
            st.markdown(tags(pelindung), unsafe_allow_html=True)
            st.caption("Daftar ini berdasarkan pola umum pada data (hasil EDA), "
                       "bukan hitungan langsung model.")

        # Simulasi retensi: model menghitung ulang dengan satu perubahan
        skenario = []
        if contract == "Month-to-month":
            skenario.append(("Kontrak 1 tahun", {"contract": "One year"}))
        if contract != "Two year":
            skenario.append(("Kontrak 2 tahun", {"contract": "Two year"}))
        if ada_internet and layanan["TechSupport"] == "No":
            skenario.append(("Tambah Dukungan Teknis",
                             {"layanan": {**layanan, "TechSupport": "Yes"}}))
        if ada_internet and layanan["OnlineSecurity"] == "No":
            skenario.append(("Tambah Keamanan Online",
                             {"layanan": {**layanan, "OnlineSecurity": "Yes"}}))
        if "automatic" not in payment:
            skenario.append(("Bayar otomatis", {"payment": "Credit card (automatic)"}))

        if skenario:
            st.markdown("#### Simulasi: apa yang terjadi jika ditawarkan...")
            kolom_sim = st.columns(len(skenario))
            for kol, (nama, ubah) in zip(kolom_sim, skenario):
                p2 = proba_churn({**v, **ubah})
                kol.metric(nama, f"{p2:.0%}", f"{(p2 - proba) * 100:+.1f} poin",
                           delta_color="inverse")
            st.caption("Setiap kotak = probabilitas churn baru jika hanya satu hal itu yang diubah, "
                       "dihitung ulang oleh model. Hijau menurun = menurunkan risiko. "
                       "Ini gambaran pola dari data, bukan jaminan hasil.")

        with st.expander("Fitur apa yang paling dipertimbangkan model? (berlaku untuk semua pelanggan)"):
            penting = (pd.Series(model.feature_importances_, index=feature_columns)
                       .sort_values(ascending=False).head(10))
            terbesar = float(penting.max())
            baris = "".join(
                f'<div class="imp-row"><div class="imp-name">{nama}</div>'
                f'<div class="imp-track"><div class="imp-fill" style="width:{nilai / terbesar * 100:.0f}%;"></div></div>'
                f'<div class="imp-val">{nilai:.1%}</div></div>'
                for nama, nilai in penting.items())
            st.markdown(baris, unsafe_allow_html=True)
            st.caption("Feature importance Random Forest: semakin panjang bar, semakin sering fitur itu "
                       "dipakai model untuk memisahkan pelanggan churn dan tidak churn.")

        with st.expander("Lihat data hasil preprocessing"):
            st.dataframe(input_df)
