import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# корень проекта нужен в sys.path: pipeline ссылается на src.features.add_features
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


@st.cache_resource
def load_artifact():
    return joblib.load(ROOT / 'models' / 'churn_pipeline.pkl')


artifact = load_artifact()
pipeline, threshold = artifact['pipeline'], artifact['threshold']

st.title("Прогнозирование оттока клиентов телеком-компании")
st.markdown("Введите параметры клиента, чтобы узнать вероятность его ухода.")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Пол", ["Male", "Female"])
    senior_citizen = st.selectbox("Пенсионер", ["No", "Yes"])
    partner = st.selectbox("Есть партнёр", ["No", "Yes"])
    dependents = st.selectbox("Есть иждивенцы", ["No", "Yes"])
    tenure = st.slider("Срок обслуживания (мес.)", 0, 72, 12)
    phone_service = st.selectbox("Телефонная связь", ["Yes", "No"])
    # без телефонной связи несколько линий невозможны
    if phone_service == "Yes":
        multiple_lines = st.selectbox("Несколько линий", ["No", "Yes"])
    else:
        multiple_lines = "No phone service"

with col2:
    internet_service = st.selectbox("Тип интернета", ["DSL", "Fiber optic", "No"])
    # без интернета интернет-услуги невозможны
    if internet_service != "No":
        online_security = st.selectbox("Онлайн-безопасность", ["No", "Yes"])
        online_backup = st.selectbox("Онлайн-бэкап", ["No", "Yes"])
        device_protection = st.selectbox("Защита устройства", ["No", "Yes"])
        tech_support = st.selectbox("Техподдержка", ["No", "Yes"])
        streaming_tv = st.selectbox("Стриминг ТВ", ["No", "Yes"])
        streaming_movies = st.selectbox("Стриминг фильмов", ["No", "Yes"])
    else:
        online_security = online_backup = device_protection = "No internet service"
        tech_support = streaming_tv = streaming_movies = "No internet service"

contract = st.selectbox("Тип контракта", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Безбумажные счета", ["No", "Yes"])
payment_method = st.selectbox("Способ оплаты",
                              ["Electronic check", "Mailed check",
                               "Bank transfer (automatic)", "Credit card (automatic)"])
monthly_charges = st.number_input("Ежемесячные платежи ($)", min_value=0.0, max_value=200.0,
                                  value=70.0, step=5.0)
total_charges = st.number_input("Общая сумма платежей ($)", min_value=0.0, max_value=10000.0,
                                value=float(tenure * monthly_charges), step=50.0,
                                help="По умолчанию — срок обслуживания × ежемесячный платёж")


if st.button("Рассчитать вероятность оттока"):
    # сырые данные в формате Dataset.csv — вся обработка внутри pipeline
    input_df = pd.DataFrame([{
        'gender': gender,
        'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
    }])

    proba = pipeline.predict_proba(input_df)[0, 1]

    st.subheader(f"Вероятность оттока: {proba:.2%}")

    if proba >= threshold:
        st.error("⚠️ Клиент в зоне риска. Рекомендуется предложить скидку на годовой контракт "
                 "или бесплатные доп. услуги.")
    else:
        st.success("✅ Низкий риск оттока.")

    st.caption(f"Клиент считается рискованным, если вероятность ≥ {threshold:.1%}. "
               "Порог подобран по максимуму F1 на кросс-валидации.")
