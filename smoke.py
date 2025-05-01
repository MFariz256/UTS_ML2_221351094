import streamlit as st
import tensorflow as tf
import numpy as np
import joblib

# Load scaler dan label encoder
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path="smoke-detection-dataset.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Judul Aplikasi
st.title("Smoke Detection Dataset")
st.write("Masukkan informasi lingkungan seperti suhu, kelembaban, dan konsentrasi gas untuk mendeteksi adanya potensi kebakaran atau asap.")

# Form input pengguna
UTC = st.number_input("Timestamp", min_value=0.0, max_value=200.0, value=50.0)
C = st.number_input("Temperature[C]", min_value=0.0, max_value=200.0, value=50.0)
humidity = st.number_input("Kelembapan (%)", min_value=0.0, max_value=200.0, value=50.0)
TVOC = st.number_input("TVOC", min_value=0.0, max_value=50.0, value=0.0)
eCO2 = st.number_input("Konsentrasi karbon dioksida", min_value=0.0, max_value=100.0, value=60.0)
h2 = st.number_input("Gas hidrogen", min_value=0.0, max_value=14.0, value=6.5)
Ethanol = st.number_input("Etanol", min_value=0.0, max_value=300.0, value=100.0)
hpa = st.number_input("Pressure", min_value=0.0, max_value=200.0, value=50.0)
pm1 = st.number_input("PM1.0", min_value=0.0, max_value=200.0, value=0.0)
pm2 = st.number_input("PM2.5", min_value=0.0, max_value=200.0, value=0.0)
nc0 = st.number_input("NC0.5", min_value=0.0, max_value=200.0, value=0.0)
nc1 = st.number_input("NC1.0", min_value=0.0, max_value=200.0, value=0.0)
nc2 = st.number_input("NC2.5", min_value=0.0, max_value=200.0, value=0.0)

if st.button("Rekomendasi"):
    # Preprocessing input
    input_data = np.array([[UTC, C, humidity, TVOC, eCO2, h2, Ethanol, hpa, pm1, pm2, nc0, nc1, nc2]])
    input_scaled = scaler.transform(input_data).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_scaled)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    
    predicted_label = np.argmax(prediction)
    smoke = label_encoder.inverse_transform([predicted_label])[0]

    st.success(f"Rekomendasi: **{smoke}**")