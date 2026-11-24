import joblib
import pandas as pd
import streamlit as st
from fitur import add_features  # wajib di-import agar model bisa dimuat (joblib.load)

st.set_page_config(page_title='Prediksi Churn Telco', page_icon='📞', layout='wide')

# ---------------------------------------------------------------- tampilan
st.markdown("""
<style>
.block-container {max-width: 1100px; padding-top: 2.2rem;}
.hero h1 {margin-bottom: .1rem;}
.hero p {opacity: .7; margin-top: 0;}
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
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- model
@st.cache_resource
def muat_model():
    paket = joblib.load('churn_model.joblib')
    return paket['model'], paket['threshold']


model, threshold = muat_model()
nama_model = type(model.named_steps['model']).__name__

# ---------------------------------------------------------------- data input
DEFAULT = dict(gender='Female', senior='Tidak', partner='No', dependents='No', tenure=12, monthly=70.0,
               contract='Month-to-month', paperless='Yes', payment='Electronic check', phone='Yes',
               multiple='No', internet='Fiber optic', security='No', backup='No', device='No',
               tech='No', tv='No', movies='No')
PRESET_RISIKO = dict(tenure=2, monthly=85.0, contract='Month-to-month', payment='Electronic check',
                     internet='Fiber optic', paperless='Yes')
PRESET_SETIA = dict(gender='Male', partner='Yes', dependents='Yes', tenure=60, monthly=65.0,
                    contract='Two year', paperless='No', payment='Bank transfer (automatic)',
                    internet='DSL', security='Yes', backup='Yes', device='Yes', tech='Yes',
                    multiple='Yes')

for k, val in DEFAULT.items():
    st.session_state.setdefault(k, val)


def isi(preset):
    st.session_state.update({**DEFAULT, **preset})


def baris_model(v):
    """Ubah input dashboard menjadi satu baris data mentah untuk pipeline."""
    tanpa_internet = v['internet'] == 'No'
    def layanan(x):
        return 'No internet service' if tanpa_internet else x
    return pd.DataFrame([{
        'gender': v['gender'], 'SeniorCitizen': int(v['senior'] == 'Ya'), 'Partner': v['partner'],
        'Dependents': v['dependents'], 'tenure': v['tenure'], 'PhoneService': v['phone'],
        'MultipleLines': 'No phone service' if v['phone'] == 'No' else v['multiple'],
        'InternetService': v['internet'], 'OnlineSecurity': layanan(v['security']),
        'OnlineBackup': layanan(v['backup']), 'DeviceProtection': layanan(v['device']),
        'TechSupport': layanan(v['tech']), 'StreamingTV': layanan(v['tv']),
        'StreamingMovies': layanan(v['movies']), 'Contract': v['contract'],
        'PaperlessBilling': v['paperless'], 'PaymentMethod': v['payment'],
        'MonthlyCharges': v['monthly'], 'TotalCharges': v['tenure'] * v['monthly'],  # didekati
    }])


def hitung(v):
    return float(model.predict_proba(baris_model(v))[:, 1][0])


def level(p):
    if p >= threshold:
        return ('Berisiko churn', (229, 72, 77),
                'Prioritaskan untuk program retensi: tawarkan kontrak 1-2 tahun, diskon, atau paket '
                'layanan tambahan (TechSupport / OnlineSecurity) sebelum pelanggan berhenti.')
    if p >= 0.7 * threshold:
        return ('Waspada', (245, 165, 36),
                'Belum melewati batas, tetapi cukup dekat. Pantau pelanggan ini dan pertimbangkan '
                'penawaran ringan.')
    return ('Aman', (48, 164, 108),
            'Kemungkinan besar bertahan. Pertahankan kualitas layanan, tidak perlu tindakan khusus.')


# ---------------------------------------------------------------- header
st.markdown('<div class="hero"><h1>📞 Prediksi Customer Churn</h1>'
            '<p>Perkirakan seberapa besar kemungkinan pelanggan Telco berhenti berlangganan, '
            'lalu lihat apa yang bisa dilakukan.</p></div>', unsafe_allow_html=True)

tab_prediksi, tab_baca = st.tabs(['Prediksi', 'Cara membaca hasil'])

# ================================================================ TAB PREDIKSI
with tab_prediksi:
    st.write('Isi data pelanggan secara manual, atau mulai dari contoh:')
    b1, b2, b3, _ = st.columns([1.2, 1.2, 0.8, 2])
    b1.button('Contoh pelanggan berisiko', on_click=isi, args=(PRESET_RISIKO,))
    b2.button('Contoh pelanggan setia', on_click=isi, args=(PRESET_SETIA,))
    b3.button('Reset', on_click=isi, args=({},))

    yn = ['Yes', 'No']
    with st.form('form_pelanggan'):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('**Profil**')
            st.selectbox('Gender', ['Female', 'Male'], key='gender')
            st.radio('Lansia (SeniorCitizen)', ['Tidak', 'Ya'], horizontal=True, key='senior')
            st.selectbox('Partner', yn, key='partner')
            st.selectbox('Dependents (tanggungan)', yn, key='dependents')
        with c2:
            st.markdown('**Langganan & pembayaran**')
            st.slider('Lama berlangganan (bulan)', 0, 72, key='tenure')
            st.slider('Tagihan bulanan (MonthlyCharges)', 18.0, 120.0, step=0.5, key='monthly')
            st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'], key='contract')
            st.selectbox('PaymentMethod', ['Electronic check', 'Mailed check',
                                           'Bank transfer (automatic)', 'Credit card (automatic)'],
                         key='payment')
            st.selectbox('PaperlessBilling', yn, key='paperless')
        with c3:
            st.markdown('**Layanan**')
            st.selectbox('PhoneService', yn, key='phone')
            st.selectbox('MultipleLines', yn, key='multiple')
            st.selectbox('InternetService', ['DSL', 'Fiber optic', 'No'], key='internet')
            a, b = st.columns(2)
            a.selectbox('OnlineSecurity', yn, key='security')
            b.selectbox('OnlineBackup', yn, key='backup')
            a.selectbox('DeviceProtection', yn, key='device')
            b.selectbox('TechSupport', yn, key='tech')
            a.selectbox('StreamingTV', yn, key='tv')
            b.selectbox('StreamingMovies', yn, key='movies')
        st.caption('Layanan tambahan diabaikan otomatis jika pelanggan tidak berlangganan internet '
                   '(InternetService = No).')
        kirim = st.form_submit_button('Prediksi', type='primary')

    if not kirim:
        st.info('Isi data pelanggan di atas, lalu klik **Prediksi**.')
    else:
        v = {k: st.session_state[k] for k in DEFAULT}
        proba = hitung(v)
        judul, rgb, saran = level(proba)
        warna = 'rgb({},{},{})'.format(*rgb)
        latar = 'rgba({},{},{},0.10)'.format(*rgb)

        st.divider()
        kiri, kanan = st.columns([1.15, 1])
        with kiri:
            st.markdown(
                f'<div class="card" style="border-left-color:{warna}; background:{latar};">'
                f'<div class="lvl">Hasil prediksi</div>'
                f'<div class="big">{proba:.0%}</div>'
                f'<div class="lbl" style="color:{warna};">{judul}</div>'
                f'<div class="track"><div class="fill" style="width:{min(proba, 1) * 100:.1f}%; '
                f'background:{warna};"></div><div class="thr" style="left:{threshold * 100:.1f}%;"></div></div>'
                f'<div class="small">Garis tegak = threshold {threshold:.0%}. '
                f'Di kanan garis berarti berisiko churn.</div>'
                f'</div>', unsafe_allow_html=True)
            st.markdown(f'**Saran:** {saran}')

        with kanan:
            risiko, pelindung = [], []
            ada_internet = v['internet'] != 'No'
            if v['contract'] == 'Month-to-month':
                risiko.append('Kontrak bulanan')
            else:
                pelindung.append(f'Kontrak {v["contract"]}')
            if v['tenure'] <= 12:
                risiko.append(f'Pelanggan baru ({v["tenure"]} bulan)')
            elif v['tenure'] > 24:
                pelindung.append(f'Pelanggan lama ({v["tenure"]} bulan)')
            if v['internet'] == 'Fiber optic':
                risiko.append('Internet Fiber optic')
            if v['payment'] == 'Electronic check':
                risiko.append('Bayar Electronic check')
            if 'automatic' in v['payment']:
                pelindung.append('Pembayaran otomatis')
            if ada_internet and v['tech'] == 'No':
                risiko.append('Tanpa TechSupport')
            if ada_internet and v['tech'] == 'Yes':
                pelindung.append('Memakai TechSupport')
            if ada_internet and v['security'] == 'No':
                risiko.append('Tanpa OnlineSecurity')
            if ada_internet and v['security'] == 'Yes':
                pelindung.append('Memakai OnlineSecurity')

            def tags(daftar):
                return ''.join(f'<span class="tag">{x}</span>' for x in daftar) or '<span class="small">-</span>'
            st.markdown('**Tanda yang cenderung menaikkan risiko**')
            st.markdown(tags(risiko), unsafe_allow_html=True)
            st.markdown('**Tanda yang cenderung menurunkan risiko**')
            st.markdown(tags(pelindung), unsafe_allow_html=True)
            st.caption('Daftar ini berdasarkan pola umum pada data (hasil EDA), bukan hitungan langsung model.')

        # ---- simulasi retensi (dihitung ulang oleh model)
        skenario = []
        if v['contract'] == 'Month-to-month':
            skenario.append(('Kontrak 1 tahun', {'contract': 'One year'}))
        if v['contract'] != 'Two year':
            skenario.append(('Kontrak 2 tahun', {'contract': 'Two year'}))
        if ada_internet and v['tech'] == 'No':
            skenario.append(('Tambah TechSupport', {'tech': 'Yes'}))
        if ada_internet and v['security'] == 'No':
            skenario.append(('Tambah OnlineSecurity', {'security': 'Yes'}))
        if 'automatic' not in v['payment']:
            skenario.append(('Bayar otomatis', {'payment': 'Credit card (automatic)'}))

        if skenario:
            st.markdown('#### Simulasi: apa yang terjadi jika ditawarkan...')
            kolom = st.columns(len(skenario))
            for kol, (nama, ubah) in zip(kolom, skenario):
                p2 = hitung({**v, **ubah})
                kol.metric(nama, f'{p2:.0%}', f'{(p2 - proba) * 100:+.1f} poin', delta_color='inverse')
            st.caption('Setiap kotak = probabilitas churn baru jika **hanya satu hal itu** yang diubah, '
                       'dihitung ulang oleh model. Panah hijau menurun = menurunkan risiko. '
                       'Ini gambaran pola dari data, bukan jaminan hasil.')
        else:
            st.caption('Pelanggan ini sudah berada pada kondisi terbaik untuk semua skenario retensi yang tersedia.')

# ================================================================ TAB CARA MEMBACA
with tab_baca:
    st.markdown(f"""
### 1. Apa arti angka persentase?
Angka besar (misalnya **72%**) adalah **probabilitas churn**: seberapa yakin model bahwa pelanggan
ini akan berhenti. Model belajar dari 7.043 pelanggan Telco, lalu menilai seberapa mirip pelanggan
yang kamu isi dengan pelanggan-pelanggan yang dulu berhenti. Angka ini **bukan kepastian**:
72% artinya "mirip dengan kelompok yang sebagian besar berhenti", bukan "pasti berhenti".

### 2. Apa itu threshold (garis batas)?
Model mengubah probabilitas menjadi keputusan lewat satu batas. Pada model ini batasnya
**{threshold:.0%}**:
- probabilitas **di atas atau sama dengan {threshold:.0%}** → **Berisiko churn**
- antara **{0.7 * threshold:.0%} dan {threshold:.0%}** → **Waspada**
- **di bawah {0.7 * threshold:.0%}** → **Aman**

Batas ini dipilih dari data latih supaya keseimbangan antara pelanggan churn yang berhasil terdeteksi
(*recall*) dan salah tebak (*precision*) paling baik. Bar pada hasil menandai posisi batas ini dengan garis tegak.

### 3. Kenapa lebih baik salah menandai pelanggan setia daripada melewatkan yang akan churn?
Melewatkan pelanggan yang akan pergi berarti kehilangan pendapatan. Salah menawari diskon pada pelanggan
setia biayanya jauh lebih kecil. Karena itu model dibuat lebih peka terhadap churn, sehingga kadang
pelanggan yang sebenarnya setia ikut ditandai berisiko.

### 4. Bagaimana membaca bagian "Simulasi"?
Bagian ini menjawab: *"kalau pelanggan ini ditawari kontrak 1 tahun, berapa risikonya?"*
Model menghitung ulang dengan **hanya satu hal yang diubah**. Angka kecil berwarna hijau
(misalnya **-18 poin**) berarti risiko turun 18 poin persentase. Skenario dengan penurunan paling besar
adalah langkah retensi yang paling menjanjikan. Jika angkanya **naik**, model menilai langkah itu
tidak membantu untuk profil pelanggan ini.

### 5. Hal yang perlu diingat
- Model hanya membaca pola dari data historis; ia tidak mengetahui alasan pribadi pelanggan.
- `TotalCharges` diperkirakan dari lama berlangganan × tagihan bulanan.
- Hasil dipakai sebagai **alat bantu prioritas** bagi tim retensi, bukan keputusan akhir.
- Model dipakai: **{nama_model}**, dataset: Telco Customer Churn (Kaggle).
""")
