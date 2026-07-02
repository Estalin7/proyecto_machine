"""
Funciones de preprocesamiento de texto.
Se extraen del notebook original para poder reutilizarlas
tanto en el entrenamiento (train.py) como en la API de inferencia (app.py).
"""
import re
import string


def limpiar_texto(texto: str) -> str:
    """Normaliza un título de noticia: minúsculas, sin URLs, sin números,
    sin puntuación y sin espacios múltiples."""
    texto = str(texto).lower()
    texto = re.sub(r"http\S+", "", texto)
    texto = re.sub(r"\d+", "", texto)
    texto = texto.translate(str.maketrans("", "", string.punctuation))
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def extraer_features_extra(df, columna_original="title", columna_limpia="title_clean"):
    """Genera las features adicionales usadas en el EDA (opcional para el modelo,
    útil si luego decides enriquecer el vectorizador con features numéricas)."""
    df["longitud_titulo"] = df[columna_limpia].apply(len)
    df["cantidad_palabras"] = df[columna_limpia].apply(lambda x: len(x.split()))
    df["mayusculas"] = df[columna_original].apply(lambda x: sum(1 for c in str(x) if c.isupper()))
    df["exclamaciones"] = df[columna_original].apply(lambda x: str(x).count("!"))
    return df
