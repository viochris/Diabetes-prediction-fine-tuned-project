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
    
    # Predict the class label (0 or 1).
    # [0] extracts the single integer result from the array.
    result = model.predict(input_scaled)[0]

    # Calculate the confidence score of the WINNING class.
    # LOGIC: We use .max() because we want to know how sure the model is about its decision.
    # - If it predicts 0 (Healthy), we want the probability of 0.
    # - If it predicts 1 (Diabetes), we want the probability of 1.
    # Example: If Probs are [0.8, 0.2] -> It predicts 0, and Confidence is 0.8 (80%).
    conf = model.predict_proba(input_scaled).max(axis=1)
    
    return result, conf

