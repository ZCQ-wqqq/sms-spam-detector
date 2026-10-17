import json
import os
from pathlib import Path

import joblib
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model" / "model.joblib"
INFO_PATH = BASE_DIR / "model" / "model_info.json"

app = FastAPI(title="SMS Spam Detector")
model = joblib.load(MODEL_PATH)

model_info = json.loads(INFO_PATH.read_text()) if INFO_PATH.exists() else {}


class Message(BaseModel):
    text: str = Field(min_length=1, max_length=1000)

    @field_validator("text")
    @classmethod
    def not_blank(cls, value):
        if not value.strip():
            raise ValueError("text cannot be blank")
        return value


def check_api_key(key):
    # the key is only required when API_KEY is set (staging / production)
    expected = os.getenv("API_KEY")
    if expected and key != expected:
        raise HTTPException(status_code=401, detail="invalid API key")


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "env": os.getenv("APP_ENV", "dev")}


@app.get("/model-info")
def get_model_info():
    return model_info


@app.post("/predict")
def predict(message: Message, x_api_key: str | None = Header(default=None)):
    check_api_key(x_api_key)
    spam_probability = float(model.predict_proba([message.text])[0][1])
    label = "spam" if spam_probability >= 0.5 else "ham"
    return {"label": label, "spam_probability": round(spam_probability, 3)}
