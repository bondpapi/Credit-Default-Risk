from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf

# === Load saved models and preprocessors ===
model = joblib.load("stacked_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# Load DNN model
dnn_model = tf.keras.models.load_model("dnn_model.h5")


# === Define input schema ===
class CreditApplication(BaseModel):
    data: dict

# === Initialize app ===
app = FastAPI()


# === Endpoint 1: Stacked Model ===
@app.post("/predict")
def predict(application: CreditApplication):
    df = pd.DataFrame([application.data])
    X = preprocessor.transform(df)
    prob = model.predict_proba(X)[0, 1]
    prediction = int(prob >= 0.1523)  # Use your tuned threshold
    return {"model": "stacked_model", "probability": prob, "prediction": prediction}


# === Endpoint 2: Deep Neural Network ===
@app.post("/predict_dnn")
def predict_dnn(application: CreditApplication):
    df = pd.DataFrame([application.data])
    X = preprocessor.transform(df)
    prob = dnn_model.predict(X)[0, 0]
    prediction = int(prob >= 0.1523)  # Same threshold as stacked model for consistency
    return {"model": "dnn_model", "probability": float(prob), "prediction": prediction}

