"""
API de inferencia para el clasificador de noticias falsas (GossipCop).

Ejecutar localmente:
    uvicorn src.app:app --reload --port 8000

Endpoints:
    GET  /health           -> chequeo de vida (para el load balancer / health check)
    POST /predict          -> { "title": "..." } o { "url": "..." } -> resultado del modelo
"""
import json
import logging
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from html import unescape
from typing import Any, Optional

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from .preprocessing import limpiar_texto
except ImportError:  # pragma: no cover - fallback para ejecuciones directas
    from preprocessing import limpiar_texto

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = os.getenv("MODEL_DIR", "models")
STATIC_DIR = os.getenv("STATIC_DIR", "static")

app = FastAPI(
    title="Fake News Classifier API",
    description="Clasifica titulares de noticias de entretenimiento como reales o falsos y permite analizar enlaces de noticias.",
    version="1.1.0",
)

model = None
vectorizer = None


class PredictRequest(BaseModel):
    title: Optional[str] = None
    url: Optional[str] = None
    verify_url: bool = False


class PredictResponse(BaseModel):
    label: str
    probability: float
    source: str
    extracted_title: Optional[str] = None
    url_verification: Optional[dict[str, Any]] = None


def extraer_titulo_html(html: str) -> Optional[str]:
    """Extrae el título visible de una página HTML usando expresiones regulares simples."""
    if not html:
        return None

    for pattern in [
        r"<title[^>]*>(.*?)</title>",
        r"<meta[^>]+property=['\"]og:title['\"][^>]+content=['\"]([^'\"]+)['\"]",
        r"<meta[^>]+name=['\"]twitter:title['\"][^>]+content=['\"]([^'\"]+)['\"]",
    ]:
        match = re.search(pattern, html, re.IGNORECASE | re.DOTALL)
        if match:
            title = unescape(match.group(1))
            title = re.sub(r"<[^>]+>", " ", title)
            title = re.sub(r"\s+", " ", title).strip()
            if title:
                return title

    return None


def extraer_titulo_de_url(url: str) -> Optional[str]:
    """Intenta extraer el título real de una URL usando newspaper4k y, si falla, usa un fallback HTML."""
    if not url:
        return None

    try:
        from newspaper import Article
    except ImportError:
        logger.warning("newspaper4k no está instalado; usando fallback HTML.")
        return None

    try:
        article = Article(url)
        article.download()
        article.parse()
        title = (article.title or "").strip()
        if title:
            return title
    except Exception as exc:
        logger.warning(f"No se pudo extraer el título con newspaper4k para {url}: {exc}")

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
    )

    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            html = response.read().decode(response.headers.get_content_charset() or "utf-8", errors="ignore")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as exc:
        logger.warning(f"No se pudo descargar la URL {url}: {exc}")
        return None

    return extraer_titulo_html(html)


def evaluar_reporte_virustotal(stats: dict[str, Any]) -> dict[str, Any]:
    """Convierte un reporte de VirusTotal en un resumen legible y peligroso/sospechoso/seguro."""
    malicious = int(stats.get("malicious", 0) or 0)
    suspicious = int(stats.get("suspicious", 0) or 0)
    harmless = int(stats.get("harmless", 0) or 0)
    undetected = int(stats.get("undetected", 0) or 0)
    total = malicious + suspicious + harmless + undetected

    if total == 0:
        return {
            "status": "unknown",
            "risk_label": "seguro",
            "is_dangerous": False,
            "stats": stats,
        }

    if malicious > 0 or suspicious > 0:
        risk_label = "peligroso"
        is_dangerous = True
    elif (malicious + suspicious) / total >= 0.05:
        risk_label = "sospechoso"
        is_dangerous = True
    else:
        risk_label = "seguro"
        is_dangerous = False

    return {
        "status": "ok",
        "risk_label": risk_label,
        "is_dangerous": is_dangerous,
        "stats": {
            "malicious": malicious,
            "suspicious": suspicious,
            "harmless": harmless,
            "undetected": undetected,
            "total": total,
        },
    }


def verificar_url_virustotal(url: str, api_key: Optional[str] = None) -> dict[str, Any]:
    """Consulta la API de VirusTotal si se proporcionó una clave válida."""
    api_key = api_key or os.getenv("VIRUSTOTAL_API_KEY")
    if not api_key:
        return {
            "status": "skipped",
            "message": "No se configuró VIRUSTOTAL_API_KEY. La verificación se omitió.",
            "risk_label": "desconocido",
            "is_dangerous": False,
            "stats": {},
        }

    form_data = urllib.parse.urlencode({"url": url}).encode("utf-8")
    request = urllib.request.Request(
        "https://www.virustotal.com/api/v3/urls",
        data=form_data,
        headers={
            "accept": "application/json",
            "x-apikey": api_key,
            "content-type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.load(response)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as exc:
        return {
            "status": "error",
            "message": f"No se pudo consultar a VirusTotal: {exc}",
            "risk_label": "desconocido",
            "is_dangerous": False,
            "stats": {},
        }

    analysis_id = (payload.get("data") or {}).get("id")
    if not analysis_id:
        return {
            "status": "error",
            "message": "VirusTotal no devolvió un identificador de análisis.",
            "risk_label": "desconocido",
            "is_dangerous": False,
            "stats": {},
        }

    analysis_request = urllib.request.Request(
        f"https://www.virustotal.com/api/v3/analyses/{analysis_id}",
        headers={
            "accept": "application/json",
            "x-apikey": api_key,
        },
        method="GET",
    )

    try:
        with urllib.request.urlopen(analysis_request, timeout=20) as response:
            analysis_payload = json.load(response)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as exc:
        return {
            "status": "error",
            "message": f"No se pudo recuperar el análisis de VirusTotal: {exc}",
            "risk_label": "desconocido",
            "is_dangerous": False,
            "stats": {},
        }

    attributes = (analysis_payload.get("data") or {}).get("attributes") or {}
    return evaluar_reporte_virustotal(attributes.get("stats") or {})


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

    title_input = (payload.title or "").strip()
    url_input = (payload.url or "").strip()

    if not title_input and not url_input:
        raise HTTPException(status_code=400, detail="Debe enviar un 'title' o un 'url'.")

    source = "title"
    extracted_title = None

    if url_input:
        source = "url"
        extracted_title = extraer_titulo_de_url(url_input)
        if not extracted_title:
            extracted_title = title_input or None
        if not extracted_title:
            raise HTTPException(status_code=400, detail="No se pudo extraer un título válido desde la URL proporcionada.")
        text_to_classify = extracted_title
    else:
        text_to_classify = title_input

    if not text_to_classify or not text_to_classify.strip():
        raise HTTPException(status_code=400, detail="El texto de entrada no puede estar vacio.")

    try:
        texto_limpio = limpiar_texto(text_to_classify)
        X = vectorizer.transform([texto_limpio])

        pred = int(model.predict(X)[0])
        proba_array = model.predict_proba(X)[0]
        proba = float(proba_array[pred])

        label = "real" if pred == 1 else "fake"
        verification = None
        if payload.verify_url and url_input:
            verification = verificar_url_virustotal(url_input, os.getenv("VIRUSTOTAL_API_KEY"))

        return PredictResponse(
            label=label,
            probability=round(proba, 4),
            source=source,
            extracted_title=extracted_title or text_to_classify,
            url_verification=verification,
        )
    except Exception as e:
        logger.error(f"Error en /predict: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")
