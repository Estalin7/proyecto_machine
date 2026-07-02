"""
API de inferencia para el clasificador de noticias falsas (GossipCop).

Ejecutar localmente:
    uvicorn src.app:app --reload --port 8000

Endpoints:
    GET  /health           -> chequeo de vida (para el load balancer / health check)
    POST /predict          -> { "title": "..." } -> { "label": "real|fake", "probability": 0.87 }
"""
import os

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from preprocessing import limpiar_texto

MODEL_DIR = os.getenv("MODEL_DIR", "models")
STATIC_DIR = os.getenv("STATIC_DIR", "static")

app = FastAPI(
    title="Fake News Classifier API",
    description="Clasifica títulos de noticias de entretenimiento como reales o falsos (dataset GossipCop).",
    version="1.0.0",
)

model = None
vectorizer = None


class PredictRequest(BaseModel):
    title: str


class PredictResponse(BaseModel):
    label: str
    probability: float


@app.on_event("startup")
def cargar_modelo():
    global model, vectorizer
    model_path = os.path.join(MODEL_DIR, "model.pkl")
    vectorizer_path = os.path.join(MODEL_DIR, "vectorizer.pkl")

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise RuntimeError(
            f"No se encontraron los artefactos del modelo en '{MODEL_DIR}'. "
            "Ejecuta primero src/train.py o descarga model.pkl y vectorizer.pkl."
        )

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# Sirve cualquier otro asset estático (css/js/imagenes) que agregues luego en static/
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado todavía.")

    if not payload.title or not payload.title.strip():
        raise HTTPException(status_code=400, detail="El campo 'title' no puede estar vacío.")

    texto_limpio = limpiar_texto(payload.title)
    X = vectorizer.transform([texto_limpio])

    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0][pred]

    label = "real" if pred == 1 else "fake"
    return PredictResponse(label=label, probability=round(float(proba), 4))
