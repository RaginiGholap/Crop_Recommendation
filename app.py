import streamlit as st
import numpy as np
import pickle

# Load saved models
model = pickle.load(open("crop_model.pkl", "rb"))
scaler = pickle.load(open("scalar.pkl", "rb"))
label_encoder = pickle.load(open("label_enocode.pkl", "rb"))

st.title("🌾 Crop Recommendation System")

st.write("Enter soil and climate details to predict the best crop")

# User inputs
N = st.number_input("Nitrogen (N)", min_value=0)
P = st.number_input("Phosphorus (P)", min_value=0)
K = st.number_input("Potassium (K)", min_value=0)
temperature = st.number_input("Temperature (°C)")
humidity = st.number_input("Humidity (%)")
ph = st.number_input("pH value")
rainfall = st.number_input("Rainfall (mm)")

if st.button("Predict Crop"):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    # Scale input data
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)

    # Decode prediction
    crop = label_encoder.inverse_transform(prediction)

    st.success(f"🌱 Recommended Crop: **{crop[0]}**")
