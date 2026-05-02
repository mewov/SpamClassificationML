import os
import joblib
from pydantic import BaseModel
from pathlib import Path
from fastapi import FastAPI

loaded = joblib.load(Path(os.getenv("MODEL_PATH", "./data/train/model.pkl")))
model = loaded["model"]
vectorizer = loaded["vectorizer"]

app = FastAPI()

class PredictRequest(BaseModel):
    message: str

@app.post("/predict")
def predict_handle(request: PredictRequest):
    X = vectorizer.transform([request.message])
    y_pred = model.predict(X)
    return {"is_ok": True, "is_spam": bool(y_pred[0])}
