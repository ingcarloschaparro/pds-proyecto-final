"""Clasificador para distinguir entre textos por favor si non-por favor Implementa baseline TF-IDF + Logistic Regression si modelo DistilBERT"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from src.config.mlflow_remote import apply_tracking_uri as _mlf_apply
    _mlf_apply(experiment="E2-Classifier")
except ImportError:
    print("MLflow remote config not found, using local tracking")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import joblib
import os
from pathlib import Path
import yaml
from typing import Dict, Any, Tuple


def preparar_datos_clasificacion(ruta_datos: str) -> Tuple[pd.Series, pd.Series]:
    """Prepara datos para clasificación por favor vs non-por favor Args: ruta_datos: Ruta al archivo CSV con datos procesados Returns: Tuple con features (ex) si labels (si)"""
    print("Cargando datos para clasificación...")

    # Cargar datos
    df = pd.read_csv(ruta_datos, low_memory=False)

    # Filtrar solo registros con labels válidos
    df_valid = df[df["label"].notna()].copy()
    print(f"Registros con labels válidos: {len(df_valid)} de {len(df)}")

    # Preparar features
    textos = []
    labels = []

    print(f"Procesando {len(df_valid)} registros válidos...")
    print("Distribución de labels:", df_valid["label"].value_counts().to_dict())

    # Procesar datos PLS y non-PLS
    for _, row in df_valid.iterrows():
        # Para PLS: usar el resumen como feature (ya que es texto en lenguaje sencillo)
        if row["label"] == "por favor":
            resumen = row["resumen"]
            # Manejar NaN y valores vacíos correctamente
            if pd.isna(resumen):
                texto = ""
            else:
                texto = str(resumen).strip()

            if len(texto) > 10:  # Solo textos con contenido significativo
                textos.append(texto)
                labels.append(1)  # PLS

        # Para non-PLS: usar el texto original
        elif row["label"] == "non_pls":
            texto_original = row["texto_original"]
            if pd.isna(texto_original):
                texto = ""
            else:
                texto = str(texto_original).strip()

            if len(texto) > 10:  # Solo textos con contenido significativo
                textos.append(texto)
                labels.append(0)  # non-PLS

    print(f"Datos preparados: {len(textos)} textos, {len(labels)} labels")
    print(f"Distribución de clases: {pd.Series(labels).value_counts()}")

    return pd.Series(textos), pd.Series(labels)


def entrenar_baseline_tfidf(
    X_train: pd.Series, y_train: pd.Series, X_test: pd.Series, y_test: pd.Series
) -> Dict[str, Any]:
    """Entrena modelo baseline TF-IDF + Logistic Regression Args: X_train, y_train: Datos de entrenamiento X_test, y_test: Datos de validación Returns: Diccionario con modelo, métricas si vectorizer"""

    print("Entrenando modelo baseline TF-IDF + Logistic Regression...")

    # Crear vectorizer TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),  # Unigramas y bigramas
        stop_words="english",
        min_df=5,
        max_df=0.8,
    )

    # Transformar textos
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print(f"Vocabulario creado: {len(vectorizer.get_feature_names_out())} términos")

    # Entrenar modelo
    modelo = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight="balanced",  # Para manejar desbalance de clases
    )

    modelo.fit(X_train_vec, y_train)

    # Predicciones
    y_pred = modelo.predict(X_test_vec)
    y_pred_proba = modelo.predict_proba(X_test_vec)

    # Métricas
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    print(".3f")
    print(".3f")

    # Reporte de clasificación
    print("\nReporte de clasificación:")
    print(
        classification_report(
            y_test, y_pred, target_names=["non-por favor", "por favor"]
        )
    )

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    print("\nMatriz de confusión:")
    print(cm)

    return {
        "modelo": modelo,
        "vectorizer": vectorizer,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=["non-por favor", "por favor"],
            output_dict=True,
        ),
        "confusion_matrix": cm.tolist(),
        "y_pred": y_pred.tolist(),
        "y_pred_proba": y_pred_proba.tolist(),
    }


def guardar_modelo(resultados: Dict[str, Any], ruta_modelo: str) -> None:
    """Guarda el modelo si vectorizer entrenados Args: resultados: Diccionario con modelo si métricas ruta_modelo: Directorio donde guardar el modelo"""
    os.makedirs(ruta_modelo, exist_ok=True)

    # Guardar modelo
    joblib.dump(resultados["modelo"], f"{ruta_modelo}/clasificador_baseline.pkl")

    # Guardar vectorizer
    joblib.dump(resultados["vectorizer"], f"{ruta_modelo}/vectorizer_tfidf.pkl")

    # Guardar métricas
    metricas = {
        "f1_macro": resultados["f1_macro"],
        "f1_weighted": resultados["f1_weighted"],
        "classification_report": resultados["classification_report"],
        "confusion_matrix": resultados["confusion_matrix"],
    }

    with open(f"{ruta_modelo}/metricas_baseline.json", "w") as f:
        import json

        json.dump(metricas, f, indent=2)

    print(f"Modelo guardado en: {ruta_modelo}")


def main():
    """Función principal para entrenar clasificador"""

    # Configuración - usar dataset completo para asegurar datos PLS válidos
    RUTA_DATOS = "data/processed/dataset_clean_v1.csv"
    RUTA_MODELO = "models/clasificador_baseline"
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    print("=== ENTRENAMIENTO CLASIFICADOR BASELINE ===")
    print("Usando dataset completo para asegurar datos por favor válidos...")

    try:
        # Preparar datos
        X, y = preparar_datos_clasificacion(RUTA_DATOS)

        # Verificar que tenemos ambas clases
        unique_labels = y.unique()
        print(f"Clases encontradas: {unique_labels}")

        if len(unique_labels) < 2:
            print("Error: Se necesita al menos a|tambien clases para clasificación")
            print("Distribución de clases:", y.value_counts())
            return None

        # Dividir en train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

        print(f"Conjunto de entrenamiento: {len(X_train)} registros")
        print(f"Conjunto de prueba: {len(X_test)} registros")
        print(f"Distribución train: {y_train.value_counts().to_dict()}")
        print(f"Distribución test: {y_test.value_counts().to_dict()}")

        # Entrenar modelo
        resultados = entrenar_baseline_tfidf(X_train, y_train, X_test, y_test)

        # Guardar modelo
        guardar_modelo(resultados, RUTA_MODELO)

        print("Entrenamiento completado exitosamente!")

        return resultados

    except Exception as e:
        print(f"Error durante el entrenamiento: {str(e)}")
        raise


if __name__ == "__main__":
    main()
