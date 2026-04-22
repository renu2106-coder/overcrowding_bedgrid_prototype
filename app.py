from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
import joblib
import numpy as np

app = FastAPI()

# Load model
model = joblib.load("model.pkl")

# ✅ Input validation (important)
class PatientData(BaseModel):
    Current_Patient_Count: int = Field(..., ge=0)
    Bed_Occupancy_Rate: float = Field(..., ge=0, le=1)
    Staff_Availability: int = Field(..., ge=0)
    Average_Waiting_Time: float = Field(..., ge=0)
    Incoming_Emergency_Cases: int = Field(..., ge=0)

@app.get("/")
def home():
    return {"message": "API running 🚀"}

@app.post("/predict")
def predict(data: PatientData):
    # Feature engineering (same as pipeline)
    load_per_staff = data.Current_Patient_Count / (data.Staff_Availability + 1)
    emergency_pressure = data.Incoming_Emergency_Cases / (data.Current_Patient_Count + 1)

    features = np.array([[
        data.Current_Patient_Count,
        data.Bed_Occupancy_Rate,
        data.Staff_Availability,
        data.Average_Waiting_Time,
        data.Incoming_Emergency_Cases,
        load_per_staff,
        emergency_pressure
    ]])

    prediction = model.predict(features)

    return {"Overcrowded": int(prediction[0])}