import streamlit as st
import numpy as np
from model import load_model, predict
from preprocess import preprocess_input

scaler, model_rf = load_model()

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="centered")
st.header("🩺 Diabetes Prediction App")
st.write("Enter patient details below to predict the likelihood of diabetes.")

with st.form("prediction_form"):
  name = st.text_input("👤 Patient Name", value="Susy")
  
  pregnancies = st.number_input("🤰 Number of Pregnancies", min_value=0, step=1, value=6)
  glucose = st.number_input("🍬 Glucose Level", min_value=0.0, value=148.0, step=0.1)
  blood_pressure = st.number_input("🩸 Blood Pressure", min_value=0.0, value=72.0, step=0.1)
  skin_thickness = st.number_input("📏 Skin Thickness", min_value=0.0, value=35.0, step=0.1)
  insulin = st.number_input("💉 Insulin Level", min_value=0.0, value=168.0, step=0.1)
  bmi = st.number_input("⚖️ Body Mass Index (BMI)", min_value=0.0, value=43.1, step=0.1)
  dpf = st.number_input("🧬 Diabetes Pedigree Function (DPF)", min_value=0.0, value=2.288, step=0.1)
  age = st.number_input("📅 Age", min_value=0, max_value=120, value=33, step=1)

  submitted = st.form_submit_button("🚀 Predict")

if submitted:
  input_value = preprocess_input(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age)
  prediction = predict(model_rf, scaler, input_value)
  result = "🟥 Diabetic" if prediction == 1 else "🟩 Non-Diabetic"

  st.markdown("---")
  st.subheader("🔎 Prediction Result")
  st.write(f"**Name:** {name if name else 'N/A'}")
  _ = st.error(f"**Prediction:** {result}") if prediction == 1 else st.success(f"**Prediction:** {result}")
