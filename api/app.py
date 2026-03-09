"""
Minimal FastAPI app for transaction type classification.

Usage:
    uvicorn api.app:app --reload

    User interface:
    http://127.0.0.1:8000/docs

Example request:
    POST /classify
    {
        "purpose_text": "Lidl grocery store purchase"
    }
"""

from pathlib import Path
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_PATH = Path("models/best_model.pkl")

app = FastAPI(
    title="Transaction Classifier API",
    description="Classifies transaction purpose text into transaction type."
)


class ClassificationRequest(BaseModel):
    purpose_text: str = Field(..., min_length=1, description="Transaction purpose text")


class ClassificationResponse(BaseModel):
    predicted_type: str

model = joblib.load("models/best_model.pkl")


@app.get("/")
def root():
    return {
        "message": "Transaction Classifier API is running.",
        "endpoints": {
            "health": "/health",
            "classify": "/classify"
        }
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/classify", response_model=ClassificationResponse)
def classify(request: ClassificationRequest):
    text = request.purpose_text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="purpose_text must not be empty")

    prediction = model.predict([text])[0]

    return ClassificationResponse(
        purpose_text=text,
        predicted_type=prediction
    )