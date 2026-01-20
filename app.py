import streamlit as st
import numpy as np
import pickle

# Load saved files
model = pickle.load(open("crop_model.pkl", "rb"))
scaler = pickle.load(open("scalar.pkl", "rb"))
le = pickle.load(open("label_enocode.pkl", "rb"))

st.title("🌾 Crop Recommendation System")
st.write("Enter Soil and Climate Parameters")

N = st.number_input("Nitrogen (N)", min_value=0.0)
P = st.number_input("Phosphorus (P)", min_value=0.0)
K = st.number_input("Potassium (K)", min_value=0.0)
temperature = st.number_input("Temperature (°C)")
humidity = st.number_input("Humidity (%)")
ph = st.number_input("Soil pH")
rainfall = st.number_input("Rainfall (mm)")

if st.button("Recommend Crop"):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    crop = le.inverse_transform(prediction)
    st.success(f"🌱 Recommended Crop: **{crop[0]}**")
