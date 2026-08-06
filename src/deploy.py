from fastapi import FastAPI
import pandas as pd
import pickle
import json

app = FastAPI()

# Load model and preprocessing objects at startup
with open("models/best_model.txt", "r") as f:
    model_name = f.read().strip()

with open(f"models/{model_name}.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.get("/health")
def health():
    return {"status": "healthy", "model": model_name}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    
    # Apply same preprocessing
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col].astype(str))
    
    df = df[scaler.feature_names_in_]
    df_scaled = scaler.transform(df)
    
    prediction = model.predict(df_scaled)[0]
    proba = model.predict_proba(df_scaled)[0].tolist()
    
    return {
        "prediction": int(prediction),
        "churn_probability": proba[1],
        "model": model_name
    }

@app.get("/metrics")
def get_metrics():
    with open("models/metrics.json", "r") as f:
        return json.load(f)