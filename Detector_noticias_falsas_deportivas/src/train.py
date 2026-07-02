"""
Script de entrenamiento — versión "productiva" del notebook machine_Learning.ipynb.

Uso:
    python src/train.py --data-dir data/ --output-dir models/

Requiere en --data-dir:
    gossipcop_real.csv
    gossipcop_fake.csv
(mismo formato ; latin1 que se usaba en Colab)
"""
import argparse
import os
import time

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from preprocessing import limpiar_texto


def cargar_datos(data_dir: str) -> pd.DataFrame:
    real = pd.read_csv(os.path.join(data_dir, "gossipcop_real.csv"), sep=";", encoding="latin1")
    fake = pd.read_csv(os.path.join(data_dir, "gossipcop_fake.csv"), sep=";", encoding="latin1")

    real = real.drop(columns=["Unnamed: 4", "Unnamed: 5"], errors="ignore")
    fake = fake.drop(columns=["Unnamed: 4", "Unnamed: 5"], errors="ignore")

    real["tipo_noticia"] = 1
    fake["tipo_noticia"] = 0

    df = pd.concat([real, fake], ignore_index=True)
    df = df.dropna(subset=["title"])
    df = df.drop_duplicates(subset=["title"])
    return df


def entrenar(data_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    print("Cargando datasets...")
    df = cargar_datos(data_dir)

    print("Limpiando texto...")
    df["title_clean"] = df["title"].apply(limpiar_texto)

    print("Vectorizando (TF-IDF)...")
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vectorizer.fit_transform(df["title_clean"])
    y = df["tipo_noticia"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Entrenando RandomForestClassifier con los mejores hiperparámetros...")
    start = time.time()
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=10,
        criterion="gini",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print(f"Entrenamiento finalizado en {(time.time() - start) / 60:.2f} min")

    acc = model.score(X_test, y_test)
    print(f"Accuracy en test: {acc:.4f}")

    print("Guardando artefactos...")
    joblib.dump(model, os.path.join(output_dir, "model.pkl"))
    joblib.dump(vectorizer, os.path.join(output_dir, "vectorizer.pkl"))
    print(f"Listo. Artefactos guardados en: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="models")
    args = parser.parse_args()
    entrenar(args.data_dir, args.output_dir)
