import streamlit as st
import numpy as np
from model import load_model, predict
from preprocess import preprocess_input

scaler, model_rf = load_model()

st.header("🩺 Diabetes Prediction App")
st.write("Enter patient details below to predict the likelihood of diabetes.")

with st.form("prediction_form"):
  name = st.text_input("👤 Patient Name", value="Susy")

  pregnancies = st.number_input("🤰 Number of Pregnancies", min_value=0.0, step=1.0, value=6.0)
  glucose = st.number_input("🍬 Glucose Level", min_value=0.0, value=148.0, step=1.0)
  blood_pressure = st.number_input("🩸 Blood Pressure", min_value=0.0, value=72.0, step=1.0)
  skin_thickness = st.number_input("📏 Skin Thickness", min_value=0.0, value=35.0, step=1.0)
  insulin = st.number_input("💉 Insulin Level", min_value=0.0, value=168.0, step=1.0)
  bmi = st.number_input("⚖️ Body Mass Index (BMI)", min_value=0.0, value=43.1)
  dpf = st.number_input("🧬 Diabetes Pedigree Function (DPF)", min_value=0.0, value=2.288)
  age = st.number_input("📅 Age", min_value=0.0, max_value=120.0, value=33.0, step=1.0)

  submitted = st.form_submit_button("🚀 Predict")

if submitted:
  input_value = preprocess_input(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age)
  prediction = predict(model_rf, scaler, input_value)
  result = "🟥 Diabetic" if prediction == 1 else "🟩 Non-Diabetic"

  st.markdown("---")
  st.subheader("🔎 Prediction Result")
  st.write(f"**Name:** {name if name else 'N/A'}")
  _ = st.error(f"**Prediction:** {result}") if prediction == 1 else st.success(f"**Prediction:** {result}")