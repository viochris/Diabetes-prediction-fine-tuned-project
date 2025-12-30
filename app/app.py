"""
Project: Diabetes Prediction Dashboard (Fine-Tuned Version)
Author: Silvio Christian, Joe
Description:
    Streamlit-based frontend that serves the Robust Random Forest model.
    Allows users to input health data and get real-time predictions.
"""

import streamlit as st
import numpy as np
from model import load_model, predict
from preprocess import preprocess_input

# ==========================================
# 1. Initialization
# ==========================================
# Load the Fine-Tuned Model and Scaler artifacts
scaler, model_rf = load_model()

# Configure page settings
st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="centered")

# ==========================================
# 2. UI Header
# ==========================================
st.header("🩺 Diabetes Prediction App")
st.write("Enter patient details below to predict the likelihood of diabetes.")

# ==========================================
# 3. User Input Form
# ==========================================
# Using st.form to batch inputs and prevent reload on every keystroke
with st.form("prediction_form"):
  name = st.text_input("👤 Patient Name", value="Susy")
  
  # Numeric inputs with constraints matching the dataset's range
  pregnancies = st.number_input("🤰 Number of Pregnancies", min_value=0, step=1, value=6)
  glucose = st.number_input("🍬 Glucose Level", min_value=0.0, value=148.0, step=0.1)
  blood_pressure = st.number_input("🩸 Blood Pressure", min_value=0.0, value=72.0, step=0.1)
  skin_thickness = st.number_input("📏 Skin Thickness", min_value=0.0, value=35.0, step=0.1)
  insulin = st.number_input("💉 Insulin Level", min_value=0.0, value=168.0, step=0.1)
  bmi = st.number_input("⚖️ Body Mass Index (BMI)", min_value=0.0, value=43.1, step=0.1)
  dpf = st.number_input("🧬 Diabetes Pedigree Function (DPF)", min_value=0.0, value=2.288, step=0.1)
  age = st.number_input("📅 Age", min_value=0, max_value=120, value=33, step=1)

  # Form submission button
  submitted = st.form_submit_button("🚀 Predict")

# ==========================================
# 4. Inference Execution
# ==========================================
if submitted:
  # Format the inputs
  input_value = preprocess_input(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age)
  
  # Run prediction using the loaded artifacts
  prediction = predict(model_rf, scaler, input_value)
  
  # Map binary result to readable label
  result = "🟥 Diabetic" if prediction == 1 else "🟩 Non-Diabetic"

  # ==========================================
  # 5. Display Results
  # ==========================================
  st.markdown("---")
  st.subheader("🔎 Prediction Result")
  st.write(f"**Name:** {name if name else 'N/A'}")
    
  # Display dynamic feedback based on the diagnosis:
  # - Class 1 (Diabetes): Use RED (st.error) to indicate health risk/alert.
  # - Class 0 (Healthy): Use GREEN (st.success) to indicate normal condition.
  if prediction == 1:
      st.error(f"**Prediction:** {result}, Confidence: {conf:.2%}")
  else:
      st.success(f"**Prediction:** {result}, Confidence: {conf:.2%}")
