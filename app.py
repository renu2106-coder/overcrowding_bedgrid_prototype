from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd

# Create app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = joblib.load("model.pkl")

# Input validation
class PatientData(BaseModel):
    Current_Patient_Count: int = Field(..., ge=0)
    Bed_Occupancy_Rate: float = Field(..., ge=0, le=1)
    Staff_Availability: int = Field(..., ge=0)
    Average_Waiting_Time: float = Field(..., ge=0)
    Incoming_Emergency_Cases: int = Field(..., ge=0)

# Root endpoint
@app.get("/")
def home():
    return {"message": "API running 🚀"}

# Prediction endpoint
@app.post("/predict")
def predict(data: PatientData):
    if (
        data.Current_Patient_Count > 200 and
        data.Bed_Occupancy_Rate > 0.9 and
        data.Staff_Availability < 5
    ):
        print("RULE TRIGGERED → OVERCROWDED")
        return {"Overcrowded": 1}
    # Feature engineering
    load_per_staff = data.Current_Patient_Count / (data.Staff_Availability + 1)
    emergency_pressure = data.Incoming_Emergency_Cases / (data.Current_Patient_Count + 1)

    # Create DataFrame (IMPORTANT FIX)
    features = pd.DataFrame([{
        "Current_Patient_Count": data.Current_Patient_Count,
        "Bed_Occupancy_Rate": data.Bed_Occupancy_Rate,
        "Staff_Availability": data.Staff_Availability,
        "Average_Waiting_Time": data.Average_Waiting_Time,
        "Incoming_Emergency_Cases": data.Incoming_Emergency_Cases,
        "load_per_staff": load_per_staff,
        "emergency_pressure": emergency_pressure
    }])

    # Debug logs
    print("INPUT FEATURES:\n", features)
    prediction = model.predict(features)
    print("MODEL OUTPUT:", prediction)

    # Optional: probability (if supported)
    try:
        prob = model.predict_proba(features)
        print("PROBABILITY:", prob)
    except:
        pass

    return {"Overcrowded": int(prediction[0])}