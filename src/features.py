from pandas import DataFrame
import numpy as np
import pandas as pd

SERVICES=[
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
]

CAT_COLS = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod', 'tenure_group',
]

NUM_COLS = [
    'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
    'avg_charge', 'charge_diff', 'count_services',
]
TENURE_BINS = [-1, 6, 12, 24, 48, np.inf]
TENURE_LABELS = ['0-6', '6-12', '12-24', '24-48', '48+']
def add_features(df)-> DataFrame:
    df=df.copy()
    df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce').fillna(0.0)
    df['tenure']=df['tenure'].astype(int)
    df['SeniorCitizen']=df['SeniorCitizen'].astype(int)
    df['MonthlyCharges']=df['MonthlyCharges'].astype(float)

    df['avg_charge']=np.where(df['tenure']>0, df['TotalCharges']/df['tenure'].replace(0,1),df['MonthlyCharges'])
    df['charge_diff']=df['MonthlyCharges']-df['avg_charge']
    df['count_services']=(df[SERVICES]=="Yes").sum(axis=1)
    df['tenure_group'] = pd.cut(df['tenure'], bins=TENURE_BINS, labels=TENURE_LABELS).astype(str)
    df[CAT_COLS]=df[CAT_COLS].astype(str)

    return df[CAT_COLS+NUM_COLS]