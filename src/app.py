"""
API de inferencia para el clasificador de noticias falsas (GossipCop).

Ejecutar localmente:
    uvicorn src.app:app --reload --port 8000

Endpoints:
    GET  /health           -> chequeo de vida (para el load balancer / health check)
    POST /predict          -> { "title": "..." } -> { "label": "real|fake", "probability": 0.87 }
"""
import logging
import os

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from preprocessing import limpiar_texto

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = os.getenv("MODEL_DIR", "models")
STATIC_DIR = os.getenv("STATIC_DIR", "static")

app = FastAPI(
    title="Fake News Classifier API",
    description="Clasifica titulos de noticias de entretenimiento como reales o falsos (dataset GossipCop).",
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

    logger.info(f"Buscando modelos en: {MODEL_DIR}")
    logger.info(f"model.pkl existe: {os.path.exists(model_path)}")
    logger.info(f"vectorizer.pkl existe: {os.path.exists(vectorizer_path)}")

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        logger.error(f"No se encontraron los artefactos del modelo en '{MODEL_DIR}'.")
        return

    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        logger.info(f"Modelo cargado: {type(model).__name__}")
        logger.info(f"Vectorizer cargado: {type(vectorizer).__name__}")
    except Exception as e:
        logger.error(f"Error al cargar el modelo: {e}")
        model = None
        vectorizer = None


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# Sirve cualquier otro asset estatico
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    if model is None or vectorizer is None:
        logger.error("Modelo no cargado - devolviendo 503")
        raise HTTPException(status_code=503, detail="Modelo no cargado todavia.")

    if not payload.title or not payload.title.strip():
        raise HTTPException(status_code=400, detail="El campo 'title' no puede estar vacio.")

    try:
        texto_limpio = limpiar_texto(payload.title)
        X = vectorizer.transform([texto_limpio])

        pred = int(model.predict(X)[0])
        proba_array = model.predict_proba(X)[0]
        proba = float(proba_array[pred])

        label = "real" if pred == 1 else "fake"
        return PredictResponse(label=label, probability=round(proba, 4))
    except Exception as e:
        logger.error(f"Error en /predict: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")
