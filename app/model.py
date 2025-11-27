"""
Module: Backend Inference Logic
Author: Silvio Christian, Joe
Description:
    Handles the loading of the Fine-Tuned Machine Learning artifacts (Model & Scaler)
    and executes predictions on new data.
"""

from sklearn.preprocessing import MinMaxScaler
import joblib

def load_model():
    """
    Loads the serialized model artifacts from the 'models/' directory.
    
    Returns:
        scaler: The MinMaxScaler fitted ONLY on training data (preventing leakage).
        model_rf: The Hyperparameter-Tuned Random Forest model.
    """
    # Load the pre-fitted scaler
    scaler = joblib.load("models/scaler.joblib")
    
    # Load the best-performing model found during Hyperparameter Tuning
    model_rf = joblib.load("models/model_hp_tune.joblib")
    
    return scaler, model_rf

def predict(model, scaler, input_data):
    """
    Performs inference on a single instance of patient data.
    
    Args:
        model: The tuned ML model.
        scaler: The fitted scaler.
        input_data (list): Raw feature values.
        
    Returns:
        int: Prediction result (0 = Non-Diabetic, 1 = Diabetic).
    """
    # Normalize input using the same scale as the training data
    input_scaled = scaler.transform([input_data])
    
    # Generate class prediction
    result = model.predict(input_scaled)[0]
    return result
