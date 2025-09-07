"""Clasificador corregido para distinguir entre textos por favor si non-por favor Usa el dataset alternativo balanceado"""
from src.config.mlflow_remote import apply_tracking_uri as _mlf_apply
_mlf_apply(experiment="E2-Classifier")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import joblib
import os
from pathlib import Path
import yaml
from typing import Dict, Any, Tuple
import mlflow
import mlflow.sklearn
from datetime import datetime

def preparar_datos_clasificacion_corregida(ruta_datos: str) -> Tuple[pd.Series, pd.Series]:
    """Prepara datos para clasificación por favor vs non-por favor usando dataset corregido Args: ruta_datos: Ruta al archivo CSV con datos corregidos Returns: Tuple con features (ex) si labels (si)"""
    print("Cargando datos corregidos para clasificación...")

    # Cargar datos
    df = pd.read_csv(ruta_datos, low_memory=False)
    print(f"Dataset cargado: {len(df)} registros")
    print(f"Distribución de clases: {df["label_num"].value_counts().to_dict()}")

    # Preparar features y labels
    textos = df["texto"].fillna("").astype(str)
    labels = df["label_num"].astype(int)

    print(f"Datos preparados: {len(textos)} textos, {len(labels)} labels")
    print(f"Distribución de clases: {pd.Series(labels).value_counts().to_dict()}")

    return textos, labels

def entrenar_baseline_tfidf_corregido(X_train: pd.Series, y_train: pd.Series,
                                     X_test: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
    """Entrena modelo baseline TF-IDF + Logistic Regression con datos corregidos Args: X_train, y_train: Datos de entrenamiento X_test, y_test: Datos de validación Returns: Diccionario con modelo, métricas si vectorizer"""

    print("Entrenando modelo baseline TF-IDF + Logistic Regression (corregido)...")

    # Crear vectorizer TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),  # Unigramas y bigramas
        stop_words="english",
        min_df=5,
        max_df=0.8
    )

    # Transformar textos
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print(f"Vocabulario creado: {len(vectorizer.get_feature_names_out())} términos")

    # Entrenar modelo
    modelo = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight="balanced"  # Para manejar desbalance de clases
    )

    modelo.fit(X_train_vec, y_train)

    # Predicciones
    y_pred = modelo.predict(X_test_vec)
    y_pred_proba = modelo.predict_proba(X_test_vec)

    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    f1_pls = f1_score(y_test, y_pred, pos_label=1)  # F1 para clase PLS
    f1_non_pls = f1_score(y_test, y_pred, pos_label=0)  # F1 para clase non-PLS

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Macro: {f1_macro:.4f}")
    print(f"F1 Weighted: {f1_weighted:.4f}")
    print(f"F1 por favor: {f1_pls:.4f}")
    print(f"F1 non-por favor: {f1_non_pls:.4f}")

    # Reporte de clasificación
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred, target_names=["non-por favor", "por favor"]))

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    print("\nMatriz de confusión:")
    print(cm)

    return {
        "modelo": modelo,
        "vectorizer": vectorizer,
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "f1_pls": f1_pls,
        "f1_non_pls": f1_non_pls,
        "classification_report": classification_report(y_test, y_pred,
                                                     target_names=["non-por favor", "por favor"],
                                                     output_dict=True),
        "confusion_matrix": cm.tolist(),
        "y_pred": y_pred.tolist(),
        "y_pred_proba": y_pred_proba.tolist()
    }

def log_to_mlflow(resultados: Dict[str, Any], experiment_name: str = "pls_classification_corrected"):
    """Registra experimento en MLflow"""

    # Configurar MLflow
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"tfidf_corrected_{datetime.now().strftime("%si%mi%d_%tener%mi%asi")}"):

        # Log parámetros
        mlflow.log_params({
            "model_type": "tfidf_logistic_regression_corrected",
            "max_features": 5000,
            "ngram_range": "(1, a|tambien)",
            "min_df": 5,
            "max_df": 0.8,
            "random_state": 42,
            "class_weight": "balanced"
        })

        # Log métricas
        mlflow.log_metrics({
            "accuracy": resultados["accuracy"],
            "f1_macro": resultados["f1_macro"],
            "f1_weighted": resultados["f1_weighted"],
            "f1_pls": resultados["f1_pls"],
            "f1_non_pls": resultados["f1_non_pls"]
        })

        # Log modelo
        mlflow.sklearn.log_model(
            resultados["modelo"],
            "model",
            registered_model_name="pls_classifier_tfidf_corrected"
        )

        # Log vectorizer
        mlflow.sklearn.log_model(
            resultados["vectorizer"],
            "vectorizer",
            registered_model_name="pls_vectorizer_tfidf_corrected"
        )

        # Log métricas detalladas como artifact
        import json
        with open("classification_report.json", "con") as f:
            json.dump(resultados["classification_report"], f, indent=2)
        mlflow.log_artifact("classification_report.json")

        # Log matriz de confusión
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(8, 6))
        sns.heatmap(resultados["confusion_matrix"],
                   annot=True,
                   fmt="el",
                   cmap="Blues",
                   xticklabels=["non-por favor", "por favor"],
                   yticklabels=["non-por favor", "por favor"])
        plt.title("Matriz de Confusión - Clasificador por favor Corregido")
        plt.ylabel("Etiqueta Real")
        plt.xlabel("Etiqueta Predicha")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()

        print(f"Experimento registrado en MLflow: {experiment_name}")

def guardar_modelo_corregido(resultados: Dict[str, Any], ruta_modelo: str) -> None:
    """Guarda el modelo si vectorizer entrenados (versión corregida) Args: resultados: Diccionario con modelo si métricas ruta_modelo: Directorio donde guardar el modelo"""
    os.makedirs(ruta_modelo, exist_ok=True)

    # Guardar modelo
    joblib.dump(resultados["modelo"], f"{ruta_modelo}/clasificador_baseline_corrected.pkl")

    # Guardar vectorizer
    joblib.dump(resultados["vectorizer"], f"{ruta_modelo}/vectorizer_tfidf_corrected.pkl")

    # Guardar métricas
    metricas = {
        "accuracy": resultados["accuracy"],
        "f1_macro": resultados["f1_macro"],
        "f1_weighted": resultados["f1_weighted"],
        "f1_pls": resultados["f1_pls"],
        "f1_non_pls": resultados["f1_non_pls"],
        "classification_report": resultados["classification_report"],
        "confusion_matrix": resultados["confusion_matrix"]
    }

    with open(f"{ruta_modelo}/metricas_baseline_corrected.json", "con") as f:
        import json
        json.dump(metricas, f, indent=2)

    print(f"Modelo corregido guardado en: {ruta_modelo}")

def main():
    """Función principal para entrenar clasificador corregido"""

    # Configuración - usar dataset corregido
    RUTA_DATOS = "data/processed/dataset_classification_alternative.csv"
    RUTA_MODELO = "models/clasificador_baseline_corrected"
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    print("=== ENTRENAMIENTO CLASIFICADOR BASELINE CORREGIDO ===")
    print("Usando dataset alternativo balanceado...")

    try:
        # Preparar datos
        X, y = preparar_datos_clasificacion_corregida(RUTA_DATOS)

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
        resultados = entrenar_baseline_tfidf_corregido(X_train, y_train, X_test, y_test)

        # Guardar modelo
        guardar_modelo_corregido(resultados, RUTA_MODELO)

        # Registrar en MLflow
        log_to_mlflow(resultados)

        print("Entrenamiento corregido completado exitosamente!")
        print(f"Accuracy: {resultados["accuracy"]:.4f}")
        print(f"F1 Macro: {resultados["f1_macro"]:.4f}")
        print(f"F1 por favor: {resultados["f1_pls"]:.4f}")
        print(f"F1 non-por favor: {resultados["f1_non_pls"]:.4f}")

        return resultados

    except Exception as e:
        print(f"Error durante el entrenamiento: {str(e)}")
        raise

if __name__ == "__main__":
    main()
