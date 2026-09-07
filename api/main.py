from fastapi import FastAPI
from pydantic import BaseModel

from src.model_service import predict_case

# ==========================================
# Create FastAPI application
# ==========================================

app = FastAPI(
    title="NexTerra AI API",
    description="Land acquisition delay prediction API",
    version="1.0",
)


# ==========================================
# Input schema
# ==========================================


class LandAcquisitionCase(BaseModel):
    state: str
    district: str
    land_type: str
    acquisition_stage: str

    case_age_days: int
    days_in_stage: int
    number_of_landowners: int
    litigation_present: int
    number_of_objections: int
    compensation_delay_days: int
    previous_stage_delay_days: int
    historical_delay_rate: float
    land_area_hectares: float


# ==========================================
# Health check
# ==========================================


@app.get("/")
def root():

    return {"message": "NexTerra AI API is running"}


# ==========================================
# Prediction endpoint
# ==========================================


@app.post("/predict")
def predict(case: LandAcquisitionCase):

    result = predict_case(case.model_dump())

    return result
