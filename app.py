import joblib
import pandas as pd
import streamlit as st
from fitur import add_features  # wajib di-import agar model bisa dimuat (joblib.load)

st.set_page_config(page_title='Prediksi Churn Telco', page_icon='📞')


@st.cache_resource
def muat_model():
    paket = joblib.load('churn_model.joblib')
    return paket['model'], paket['threshold']


model, threshold = muat_model()

st.title('📞 Prediksi Customer Churn — Telco')
st.write('Isi data pelanggan, lalu lihat apakah pelanggan berisiko berhenti berlangganan.')
st.sidebar.header('Tentang model')
st.sidebar.write(f'Model: **{type(model.named_steps["model"]).__name__}**')
st.sidebar.write(f'Threshold keputusan: **{threshold:.2f}**')
st.sidebar.caption('Dataset: Telco Customer Churn (Kaggle).')

yn = ['Yes', 'No']
tiga = ['Yes', 'No', 'No internet service']

c1, c2 = st.columns(2)
with c1:
    gender = st.selectbox('Gender', ['Female', 'Male'])
    senior = st.radio('Lansia (SeniorCitizen)', ['Tidak', 'Ya'], horizontal=True)
    partner = st.selectbox('Partner', yn, index=1)
    dependents = st.selectbox('Dependents', yn, index=1)
    tenure = st.slider('Lama berlangganan (bulan)', 0, 72, 12)
    monthly = st.slider('Tagihan bulanan (MonthlyCharges)', 18.0, 120.0, 70.0, 0.05)
    contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    paperless = st.selectbox('PaperlessBilling', yn)
    payment = st.selectbox('PaymentMethod', ['Electronic check', 'Mailed check',
                                             'Bank transfer (automatic)', 'Credit card (automatic)'])
with c2:
    phone = st.selectbox('PhoneService', yn)
    multiple = st.selectbox('MultipleLines', ['Yes', 'No', 'No phone service'], index=1)
    internet = st.selectbox('InternetService', ['DSL', 'Fiber optic', 'No'], index=1)
    security = st.selectbox('OnlineSecurity', tiga, index=1)
    backup = st.selectbox('OnlineBackup', tiga, index=1)
    device = st.selectbox('DeviceProtection', tiga, index=1)
    tech = st.selectbox('TechSupport', tiga, index=1)
    tv = st.selectbox('StreamingTV', tiga, index=1)
    movies = st.selectbox('StreamingMovies', tiga, index=1)

if st.button('Prediksi', type='primary'):
    df_in = pd.DataFrame([{
        'gender': gender, 'SeniorCitizen': int(senior == 'Ya'), 'Partner': partner,
        'Dependents': dependents, 'tenure': tenure, 'PhoneService': phone,
        'MultipleLines': multiple, 'InternetService': internet, 'OnlineSecurity': security,
        'OnlineBackup': backup, 'DeviceProtection': device, 'TechSupport': tech,
        'StreamingTV': tv, 'StreamingMovies': movies, 'Contract': contract,
        'PaperlessBilling': paperless, 'PaymentMethod': payment,
        'MonthlyCharges': monthly, 'TotalCharges': tenure * monthly,  # TotalCharges didekati
    }])
    proba = float(model.predict_proba(df_in)[:, 1][0])

    st.metric('Probabilitas churn', f'{proba:.1%}')
    st.progress(min(proba, 1.0))
    if proba >= threshold:
        st.error('BERISIKO CHURN')
        st.write('Saran: tawarkan kontrak 1 atau 2 tahun, diskon, atau paket tambahan '
                 '(TechSupport / OnlineSecurity) sebelum pelanggan berhenti.')
    else:
        st.success('Kemungkinan bertahan')
        st.write('Saran: pertahankan layanan, tidak perlu tindakan khusus.')
