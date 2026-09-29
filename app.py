import pickle
import warnings
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

warnings.filterwarnings("ignore")

app = FastAPI()

# Load model once at startup
with open("model.pkl", "rb") as f:
    model = pickle.load(f)


class ForecastRequest(BaseModel):
    steps: int


@app.get("/")
def health():
    return {"status": "ok", "model": "ARIMA(1,1,1)"}


@app.post("/forecast")
def forecast(request: ForecastRequest):
    predictions = model.forecast(request.steps)
    return {
        "steps": request.steps,
        "forecast": predictions.tolist()
    }
