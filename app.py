import streamlit as st
import numpy as np
import pickle
import os

st.set_page_config(page_title="Crop Recommendation System 🌾", layout="centered")

st.title("🌾 Crop Recommendation System")
st.write("Enter soil and climate parameters to get crop recommendations.")

# -----------------------------
# 1️⃣ Check if all required files exist
# -----------------------------
required_files = ["crop_model.pkl", "scalar.pkl", "label_enocode.pkl"]

for file in required_files:
    if not os.path.exists(file):
        st.error(f"❌ Required file not found: {file}. Please make sure it is in the same folder as app.py")
        st.stop()  # Stop the app if a file is missing

# -----------------------------
# 2️⃣ Load the model, scaler, and label encoder
# -----------------------------
with open("crop_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scalar.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# -----------------------------
# 3️⃣ User input
# -----------------------------
N = st.number_input("Nitrogen (N)", min_value=0.0)
P = st.number_input("Phosphorus (P)", min_value=0.0)
K = st.number_input("Potassium (K)", min_value=0.0)
temperature = st.number_input("Temperature (°C)")
humidity = st.number_input("Humidity (%)")
ph = st.number_input("Soil pH")
rainfall = st.number_input("Rainfall (mm)")

# -----------------------------
# 4️⃣ Predict crop
# -----------------------------
if st.button("Recommend Crop"):
    try:
        # Convert input into numpy array and scale
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        input_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_scaled)

        # Convert numeric label to actual crop name
        crop = le.inverse_transform(prediction)

        st.success(f"🌱 Recommended Crop: **{crop[0]}**")

    except Exception as e:
        st.error(f"⚠️ Something went wrong during prediction: {e}")

