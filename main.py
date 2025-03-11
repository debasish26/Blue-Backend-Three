import numpy as np
import joblib
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# Allow CORS for all origins and all ports
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all domains
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# File paths for model files
flood_model_file = "flood_model.pkl"
cyclone_model_file = "cyclone_model.pkl"

# Ensure model files exist before loading
if not os.path.exists(flood_model_file):
    raise FileNotFoundError(f"Model file {flood_model_file} not found! Please upload it.")

if not os.path.exists(cyclone_model_file):
    raise FileNotFoundError(f"Model file {cyclone_model_file} not found! Please upload it.")

# Load pre-trained models
print("Loading pre-trained models...")
flood_model = joblib.load(flood_model_file)
cyclone_model = joblib.load(cyclone_model_file)
print("Models loaded successfully!")

# Prediction Endpoints
@app.get("/predict/flood")
async def predict_flood(river_discharge: float, precipitation: float, lat: float, lon: float):
    data = np.array([[river_discharge, precipitation]])
    prediction = flood_model.predict_proba(data)[0][1] * 100
    return {
        "event": "Flood Risk",
        "probability": round(prediction, 2),
        "expectedDate": "Within 24-48 hours",
        "intensity": "High" if prediction > 50 else "Low",
        "image": "https://source.unsplash.com/800x600/?flood,disaster",
        "details": f"Predicted flood risk based on river discharge ({river_discharge}) and precipitation ({precipitation}).",
        "recommendations": ["Prepare sandbags", "Avoid low-lying areas", "Monitor weather updates"],
        "location": f"Lat: {lat}, Lon: {lon}"
    }

@app.get("/predict/cyclone")
async def predict_cyclone(central_pressure: float, lat: float, lon: float):
    data = np.array([[central_pressure, lat, lon]])
    prediction = cyclone_model.predict_proba(data)[0][1] * 100
    return {
        "event": "Cyclone Severity",
        "probability": round(prediction, 2),
        "expectedDate": "Within 24-48 hours",
        "intensity": "Severe" if prediction > 50 else "Not Severe",
        "image": "https://source.unsplash.com/800x600/?cyclone,storm",
        "details": f"Predicted cyclone severity based on central pressure ({central_pressure}) at coordinates ({lat}, {lon}).",
        "recommendations": ["Stay indoors", "Stock supplies", "Follow evacuation orders"],
        "location": f"Lat: {lat}, Lon: {lon}"
    }
