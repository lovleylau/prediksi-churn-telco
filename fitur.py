import pandas as pd

LAYANAN = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
           'TechSupport', 'StreamingTV', 'StreamingMovies']

def add_features(d):
    d = d.copy()
    d['tenure_group'] = pd.cut(d['tenure'].clip(0, 72), bins=[-1, 12, 24, 48, 72],
                               labels=['0-12', '13-24', '25-48', '49-72']).astype(str)
    d['n_layanan'] = (d[LAYANAN] == 'Yes').sum(axis=1)               # jumlah layanan tambahan
    d['avg_charge'] = d['TotalCharges'] / d['tenure'].clip(lower=1)  # rata-rata tagihan per bulan
    d['autopay'] = d['PaymentMethod'].str.contains('automatic').astype(int)
    return d

num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'n_layanan', 'avg_charge']
bin_cols = ['SeniorCitizen', 'autopay']
cat_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
            'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group']
