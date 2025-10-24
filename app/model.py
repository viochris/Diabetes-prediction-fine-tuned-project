from sklearn.preprocessing import MinMaxScaler
import joblib

def load_model():
    scaler = joblib.load("models/scaler.joblib")
    model_rf = joblib.load("models/model_hp_tune.joblib")
    return scaler, model_rf

def predict(model, scaler, input_data):
    input_scaled = scaler.transform([input_data])
    result = model.predict(input_scaled)[0]
    return result