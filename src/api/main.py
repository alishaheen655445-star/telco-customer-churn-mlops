from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load model
model = joblib.load("model.pkl")


# IMPORTANT: MUST MATCH TRAINING FEATURES EXACTLY
class CustomerData(BaseModel):
    gender: str
    senior_citizen: int
    partner: str
    dependents: str
    tenure_months: int
    phone_service: str
    multiple_lines: str
    internet_service: str
    online_security: str
    online_backup: str
    device_protection: str
    tech_support: str
    streaming_tv: str
    streaming_movies: str
    contract: str
    paperless_billing: str
    payment_method: str
    monthly_charges: float
    total_charges: float


@app.get("/")
def home():
    return {"message": "API is running"}


@app.post("/predict")
def predict(data: CustomerData):

    df = pd.DataFrame([data.dict()])

    prediction = model.predict(df)[0]

    return {
        "prediction": int(prediction)
    }