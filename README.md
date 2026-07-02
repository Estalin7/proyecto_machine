# El Verificador — Detector de Noticias Falsas de Espectáculos

Proyecto de Machine Learning que clasifica titulares de noticias de espectáculos como **reales o falsos**, usando Random Forest + TF-IDF entrenado con el dataset **GossipCop**.

Desplegado en: 🌐 [https://proyecto-machine-70jn.onrender.com](https://proyecto-machine-70jn.onrender.com)

---

## 📁 Estructura del Proyecto

```
proyecto_machine/
├── src/
│   ├── app.py              # API FastAPI (endpoints /predict, /health, /)
│   ├── train.py            # Script de entrenamiento del modelo
│   └── preprocessing.py    # Funciones de limpieza de texto
├── static/
│   └── index.html          # Interfaz web "El Verificador"
├── data/                   # (no incluido en repo — archivos .gitignore)
│   ├── gossipcop_real.csv
│   └── gossipcop_fake.csv
├── models/                 # (no incluido en repo — descargado en Docker build)
│   ├── model.pkl
│   └── vectorizer.pkl
├── Dockerfile
├── requirements.txt
└── .gitignore
```

---

## 🚀 Pasos del Proyecto

### 1. Inicialización del Repositorio en GitHub

Se comenzó iniciando un repositorio local y conectándolo a GitHub:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/Estalin7/proyecto_machine.git
git push -u origin main
```

> **Problema encontrado:** la carpeta `Detector_noticias_falsas_deportivas/` tenía un `.git` propio (submodule). Se resolvió eliminando el `.git` interno y re-agregando los archivos normalmente.

---

### 2. Preprocesamiento y Entrenamiento del Modelo

El modelo fue entrenado con el dataset **GossipCop** (noticias de espectáculos reales y falsas).

**Script de entrenamiento:** [`src/train.py`](src/train.py)

Pasos del entrenamiento:
1. Cargar `gossipcop_real.csv` y `gossipcop_fake.csv`
2. Limpiar texto con `limpiar_texto()`: minúsculas, sin URLs, sin números, sin puntuación
3. Vectorizar con **TF-IDF** (`max_features=5000`, `stop_words="english"`)
4. Entrenar **RandomForestClassifier** (`n_estimators=300`, `class_weight="balanced"`)
5. Guardar `model.pkl` y `vectorizer.pkl` con `joblib`

Para ejecutar el entrenamiento localmente:

```bash
# Instalar dependencias
pip install -r requirements.txt

# Entrenar el modelo (requiere los CSV en la carpeta data/)
$env:PYTHONPATH="src"
python src/train.py --data-dir data --output-dir models
```

**Resultado:** Accuracy en test ≈ **81.25%**

---

### 3. API con FastAPI

**Archivo:** [`src/app.py`](src/app.py)

La API expone tres endpoints:

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/` | Sirve la interfaz web (`index.html`) |
| `GET` | `/health` | Estado de la API y si el modelo está cargado |
| `POST` | `/predict` | Clasifica un titular de noticia |

**Ejemplo de uso del endpoint `/predict`:**

```bash
curl -X POST https://proyecto-machine-70jn.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"title": "Celebrity spotted at secret wedding ceremony"}'
```

**Respuesta esperada:**
```json
{
  "label": "real",
  "probability": 0.7133
}
```

---

### 4. Interfaz Web

**Archivo:** [`static/index.html`](static/index.html)

Diseño estilo periódico clásico ("El Verificador"). El usuario escribe un titular y el modelo devuelve:
- ✅ **Verosímil** (noticia real probable)
- ❌ **Falso probable** (noticia falsa probable)
- Porcentaje de confianza del modelo

---

### 5. Dockerización

**Archivo:** [`Dockerfile`](Dockerfile)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src

# Descargar modelos desde Google Drive durante el build
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
```

Los modelos **no se incluyen en el repositorio** (por su tamaño) y se descargan automáticamente desde Google Drive durante el proceso de build en Docker.

---

### 6. Despliegue en Render

El proyecto se desplegó usando **Render** con Docker:

1. Crear cuenta en [render.com](https://render.com)
2. Conectar el repositorio de GitHub: `Estalin7/proyecto_machine`
3. Seleccionar **"Web Service"** → tipo **"Docker"**
4. Render detecta automáticamente el `Dockerfile` y hace el build

**Problemas encontrados y soluciones:**

| Error | Causa | Solución |
|-------|-------|----------|
| `/models`: not found | La carpeta `models/` no estaba en el repo | Cambiar a descargar modelos con `gdown` en Dockerfile |
| `gdown: error: unrecognized arguments: --id` | gdown v6.x eliminó el flag `--id` | Usar solo el file ID o URL directa |
| `gdown: error: unrecognized arguments: --fuzzy` | gdown v6.x eliminó el flag `--fuzzy` | Pasar URL completa sin flags extra |
| `ModuleNotFoundError: No module named 'preprocessing'` | Python no encontraba `src/preprocessing.py` | Agregar `ENV PYTHONPATH=/app/src` al Dockerfile |
| `idf vector is not fitted` | Modelos guardados con sklearn 1.7.2, Docker usaba 1.4.2 | Actualizar `requirements.txt` a `scikit-learn==1.7.2` |

---

## ⚙️ Ejecución Local

```bash
# 1. Clonar el repositorio
git clone https://github.com/Estalin7/proyecto_machine.git
cd proyecto_machine

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Entrenar el modelo (necesitas los CSV en data/)
$env:PYTHONPATH="src"   # Windows PowerShell
python src/train.py --data-dir data --output-dir models

# 4. Levantar la API
uvicorn src.app:app --reload --port 8000
```

Abrir en el navegador: [http://localhost:8000](http://localhost:8000)

---

## 🛠️ Tecnologías Usadas

| Tecnología | Uso |
|------------|-----|
| Python 3.11 | Lenguaje principal |
| scikit-learn 1.7.2 | Modelo ML (RandomForest + TF-IDF) |
| FastAPI | API REST |
| Uvicorn | Servidor ASGI |
| Joblib | Serialización del modelo |
| Docker | Contenedor para despliegue |
| Render | Plataforma de despliegue |
| GitHub | Control de versiones |
| gdown | Descarga de modelos desde Google Drive |

---

## 📊 Dataset

- **Nombre:** GossipCop
- **Archivos:** `gossipcop_real.csv` y `gossipcop_fake.csv`
- **Formato:** CSV separado por `;`, encoding `latin1`
- **Columnas principales:** `title`, `tipo_noticia` (1=real, 0=fake)

> Los archivos CSV **no se incluyen en el repositorio** (están en `.gitignore`) por su tamaño.

---

## 👨‍💻 Autor

**Estalin** — Proyecto académico UPAO  
*No constituye verificación periodística oficial.*
