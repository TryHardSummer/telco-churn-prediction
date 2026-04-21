import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Загрузка артефактов
@st.cache_resource
def load_artifacts():
    model = joblib.load('models/catboost_model.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    return model, feature_names

model, feature_names = load_artifacts()

# Заголовок
st.title("Прогнозирование оттока клиентов телеком-компании")
st.markdown("Введите параметры клиента, чтобы узнать вероятность его ухода.")

# Разделение на колонки для компактности
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Пол", ["Male", "Female"])
    senior_citizen = st.selectbox("Пенсионер", ["No", "Yes"])
    partner = st.selectbox("Есть партнёр", ["No", "Yes"])
    dependents = st.selectbox("Есть иждивенцы", ["No", "Yes"])
    tenure = st.slider("Срок обслуживания (мес.)", 0, 72, 12)
    phone_service = st.selectbox("Телефонная связь", ["No", "Yes"])
    multiple_lines = st.selectbox("Несколько симок", ["No", "Yes", "No phone service"])

with col2:
    internet_service = st.selectbox("Тип интернета", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Онлайн-безопасность", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Онлайн-бэкап", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Защита устройства", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Техподдержка", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Стриминг ТВ", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Стриминг фильмов", ["No", "Yes", "No internet service"])

contract = st.selectbox("Тип контракта", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Безбумажные счета", ["No", "Yes"])
payment_method = st.selectbox("Способ оплаты", 
                              ["Electronic check", "Mailed check", 
                               "Bank transfer (automatic)", "Credit card (automatic)"])
monthly_charges = st.number_input("Ежемесячные платежи ($)", min_value=0.0, max_value=200.0, value=70.0, step=5.0)
total_charges = st.number_input("Общая сумма платежей ($)", min_value=0.0, max_value=10000.0, value=1000.0, step=50.0)

# Кнопка
if st.button("Рассчитать вероятность оттока"):
    # Формируем DataFrame из введённых данных
    input_dict = {
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
        'TotalCharges': total_charges
    }
    input_df = pd.DataFrame([input_dict])

    # Предобработка
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        input_df[col] = input_df[col].map({'Yes': 1, 'No': 0})

    service_cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in service_cols:
        input_df[col] = input_df[col].map({'Yes': 1, 'No': 0, 
                                           'No phone service': 0, 'No internet service': 0})

    input_df['gender'] = input_df['gender'].map({'Male': 1, 'Female': 0})

    # One-Hot Encoding
    cat_cols = ['InternetService', 'Contract', 'PaymentMethod']
    input_df = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)

    # Приводим к тому же набору признаков, что был при обучении
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_names]

    # Прогноз (без масштабирования)
    proba = model.predict_proba(input_df)[0, 1]

    # Вывод результата
    st.subheader(f"Вероятность оттока: {proba:.2%}")

    if proba >= 0.7:
        st.error("⚠️ Высокий риск оттока. Рекомендуется предложить скидку на годовой контракт или бесплатные доп. услуги.")
    elif proba >= 0.4:
        st.warning("⚠️ Средний риск оттока. Можно отправить персонализированное предложение.")
    else:
        st.success("✅ Низкий риск оттока. Клиент лоялен.")