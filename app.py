import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load("energy_prediction_model.pkl")
scaler = joblib.load("scaler.pkl")

# Updated feature list with pressure
FEATURES = ['GHI', 'temp', 'pressure', 'humidity', 'wind_speed', 'wind_power_index']

st.set_page_config(page_title="Output Predictor", layout="centered")
st.title("🔍 Weather-Based Output Predictor")

input_mode = st.radio("Choose input method:", ['Manual Entry', 'Upload CSV'])

if input_mode == 'Manual Entry':
    st.subheader("📥 Enter Weather Parameters")

    inputs = {}
    col1, col2, col3 = st.columns(3)
    with col1:
        inputs['GHI'] = st.number_input("GHI", min_value=0.0, max_value=1200.0, value=300.0)
        inputs['temp'] = st.number_input("Temperature (°C)", -10.0, 50.0, value=25.0)
    with col2:
        inputs['pressure'] = st.number_input("Pressure (hPa)", 800.0, 1100.0, value=1013.0)
        inputs['humidity'] = st.number_input("Humidity (%)", 0.0, 100.0, value=60.0)
    with col3:
        inputs['wind_speed'] = st.number_input("Wind Speed (m/s)", 0.0, 20.0, value=5.0)
        inputs['wind_power_index'] = st.number_input("Wind Power Index", 0.0, 1.0, value=0.5)

    if st.button("Predict"):
        input_df = pd.DataFrame([inputs])
        scaled_input = scaler.transform(input_df[FEATURES])
        prediction = model.predict(scaled_input)[0]
        result = "High" if prediction == 1 else "Low"

        st.success(f"✅ Predicted Output: **{result}**")
        st.dataframe(input_df.assign(Predicted_Output=result))

else:
    st.subheader("📤 Upload CSV File")
    uploaded_file = st.file_uploader("Upload CSV with unscaled inputs", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("📄 Uploaded Data", df.head())

        if all(col in df.columns for col in FEATURES):
            scaled_df = scaler.transform(df[FEATURES])
            predictions = model.predict(scaled_df)
            df['Predicted Output'] = np.where(predictions == 1, 'High', 'Low')

            st.subheader("📊 Prediction Results")
            st.dataframe(df[FEATURES + ['Predicted Output']])
        else:
            missing_cols = [col for col in FEATURES if col not in df.columns]
            st.error(f"❌ Missing required columns in CSV: {missing_cols}")
