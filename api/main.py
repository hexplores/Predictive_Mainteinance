
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field

# --------------------------------------------------
# 1. Load the trained model and threshold
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
THRESHOLD_PATH = (
    BASE_DIR / "outputs" / "training" / "selected_threshold.txt"
)

model = joblib.load(MODEL_PATH)
threshold = float(THRESHOLD_PATH.read_text().strip())

# --------------------------------------------------
# 2. Create the FastAPI application
# --------------------------------------------------
app = FastAPI(
    title="Predictive Maintenance API",
    description="Predict machine failure from sensor measurements.",
    version="1.0.0",
)

# --------------------------------------------------
# 3. Define the input data
# --------------------------------------------------
class MachineData(BaseModel):
    machine_type: str = Field(
        description="Machine type: L, M or H"
    )
    air_temperature: float = Field(
        alias="Air temperature [K]"
    )
    process_temperature: float = Field(
        alias="Process temperature [K]"
    )
    rotational_speed: float = Field(
        alias="Rotational speed [rpm]"
    )
    torque: float = Field(
        alias="Torque [Nm]"
    )
    tool_wear: float = Field(
        alias="Tool wear [min]"
    )

# --------------------------------------------------
# 4. Health-check endpoint
# --------------------------------------------------
@app.get("/")
def home():
    return {
        "message": "Predictive Maintenance API is running",
        "model": "Random Forest",
        "threshold": threshold,
    }

# --------------------------------------------------
# 5. Prediction endpoint
# --------------------------------------------------
@app.post("/predict")
def predict_failure(data: MachineData):

    # Convert incoming JSON into the model's expected columns
    input_data = pd.DataFrame([{
        "Type": data.machine_type.upper(),
        "Air temperature [K]": data.air_temperature,
        "Process temperature [K]": data.process_temperature,
        "Rotational speed [rpm]": data.rotational_speed,
        "Torque [Nm]": data.torque,
        "Tool wear [min]": data.tool_wear,
    }])

    # Predict probability of machine failure
    probability = float(model.predict_proba(input_data)[0, 1])

    # Apply the threshold selected during validation
    prediction = int(probability >= threshold)

    return {
        "predicted_failure": prediction,
        "failure_probability": round(probability, 4),
        "threshold": threshold,
        "maintenance_alert": (
            "Potential failure detected"
            if prediction == 1
            else "No failure alert"
        ),
        "note": (
            "This is a model prediction, not a confirmed diagnosis "
            "of machine condition."
        ),
    }
