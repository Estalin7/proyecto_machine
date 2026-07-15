FROM python:3.11-slim

WORKDIR /app

# Dependencias del sistema mínimas
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m nltk.downloader punkt punkt_tab stopwords -q

COPY src/ ./src
# Descargar modelos usando gdown
RUN pip install gdown
RUN mkdir -p models && \
    gdown "https://drive.google.com/file/d/1bpIh1EEW-UWZxGOwW1dLqtHOCR4EF3m8/view?usp=sharing" -O models/model.pkl && \
    gdown "https://drive.google.com/file/d/1ytHdWOk9vsntO4iENayroXhWrMVCpN_-/view?usp=sharing" -O models/vectorizer.pkl

COPY static/ ./static

ENV MODEL_DIR=/app/models
ENV STATIC_DIR=/app/static
ENV PYTHONPATH=/app/src
EXPOSE 8000

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
