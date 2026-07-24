from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Dict, Any, List
import predict
import joblib
import os

app = FastAPI(
    title="Shipment Delay Prediction API",
    description="REST API for predicting shipment delays using an XGBoost classification model.",
    version="1.0.0"
)

MODEL_PATH = "best_xgboost_model.pkl"
FEATURES_PATH = "feature_columns.pkl"

# Check assets on startup
if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
    raise RuntimeError(f"Required model assets not found in the working directory. Ensure {MODEL_PATH} and {FEATURES_PATH} exist.")

# Load model and feature columns
model = predict.load_xgboost_model(MODEL_PATH)
feature_columns = joblib.load(FEATURES_PATH)

# Define request schema based on FEATURE_DEFAULTS from predict.py
class ShipmentData(BaseModel):
    shipping_mode: str = Field(default="Sea", description="Air, Rail, Road, Sea")
    shipment_type: str = Field(default="Import", description="Import, Export")
    priority: str = Field(default="Standard", description="Express, Standard, Urgent")
    weight_kg: float = Field(default=5000.0)
    volume_cbm: float = Field(default=15.0)
    declared_value: float = Field(default=10000.0)
    insurance: bool = Field(default=False)
    fragile_x: bool = Field(default=False)
    carrier_type: str = Field(default="Road")
    average_rating: float = Field(default=4.0)
    fleet_size: float = Field(default=100.0)
    years_of_service: int = Field(default=5)
    customer_type: str = Field(default="Business")
    industry: str = Field(default="Retail")
    country: str = Field(default="USA")
    customer_status: str = Field(default="Active")
    customs_required: bool = Field(default=False)
    documentation_complete: bool = Field(default=True)
    inspection_required: bool = Field(default=False)
    cargo_type: str = Field(default="Electronics")
    category: str = Field(default="Electronics")
    hs_code: float = Field(default=8517.0)
    hazardous: bool = Field(default=False)
    perishable: bool = Field(default=False)
    temperature_controlled: bool = Field(default=False)
    fragile_y: bool = Field(default=False)
    weight_per_unit: float = Field(default=5.0)
    distance_km: float = Field(default=1000.0)
    average_transit_days: int = Field(default=5)
    route_risk: str = Field(default="Low")
    traffic_index: float = Field(default=30.0)
    vehicle_type: str = Field(default="Truck")
    capacity_kg: int = Field(default=20000)
    fuel_type: str = Field(default="Diesel")
    maintenance_status: str = Field(default="Good")
    vehicle_age: int = Field(default=4)
    warehouse_capacity: float = Field(default=50000.0)
    current_utilization: float = Field(default=75.0)
    warehouse_type: str = Field(default="Regional")
    weather_condition: str = Field(default="Clear")
    temperature: float = Field(default=20.0)
    rainfall: float = Field(default=0.0)
    humidity: float = Field(default=50.0)
    wind_speed: float = Field(default=10.0)
    visibility: float = Field(default=10.0)
    booking_month: int = Field(default=6)
    booking_day: int = Field(default=15)
    booking_weekday: int = Field(default=2)
    ship_month: int = Field(default=6)
    ship_day: int = Field(default=16)
    ship_weekday: int = Field(default=3)

    class Config:
        extra = "allow"
        json_schema_extra = {
            "example": {
                "shipping_mode": "Sea",
                "shipment_type": "Import",
                "priority": "Standard",
                "weight_kg": 5000.0,
                "distance_km": 1250.0,
                "weather_condition": "Clear",
                "insurance": False
            }
        }

@app.get("/")
def read_root():
    """Health check endpoint confirming API status and model load state."""
    return {
        "status": "online",
        "model_loaded": model is not None,
        "features_loaded": len(feature_columns)
    }

@app.post("/predict")
def predict_shipment_delay(shipment: ShipmentData, threshold: float = Query(default=0.5, ge=0.0, le=1.0)):
    """
    Predicts the delay of a single shipment.
    """
    shipment_dict = shipment.dict()
    res = predict.predict_delay(shipment_dict, model, feature_columns, predict.LABEL_MAPPINGS, threshold)
    if res["Status"] == "Failed":
        raise HTTPException(status_code=400, detail=res["Message"])
    return res

@app.post("/predict/batch")
def predict_batch_shipment_delay(shipments: List[ShipmentData], threshold: float = Query(default=0.5, ge=0.0, le=1.0)):
    """
    Predicts the delays of multiple shipments in batch.
    """
    results = []
    for shipment in shipments:
        shipment_dict = shipment.dict()
        res = predict.predict_delay(shipment_dict, model, feature_columns, predict.LABEL_MAPPINGS, threshold)
        results.append(res)
    return {"predictions": results}
