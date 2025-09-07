#!/usr/bin/env python3
"""Script de prueba para verificar que MLflow logging funciona correctamente (local)"""

import mlflow
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd

# Configurar MLflow local
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("E2-Test-Logging-Local")

def test_mlflow_logging():
    """Prueba el logging de MLflow con un modelo simple"""
    
    with mlflow.start_run(run_name="test_logging_run_local"):
        print("=== PRUEBA DE LOGGING MLFLOW LOCAL ===")
        
        # Generar datos de prueba
        X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Loggear parámetros
        mlflow.log_params({
            "n_samples": 1000,
            "n_features": 20,
            "test_size": 0.2,
            "random_state": 42,
            "model_type": "LogisticRegression"
        })
        
        # Entrenar modelo
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        # Predicciones
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Macro: {f1_macro:.4f}")
        print(f"F1 Weighted: {f1_weighted:.4f}")
        
        # Loggear métricas
        mlflow.log_metrics({
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted
        })
        
        # Loggear modelo
        mlflow.sklearn.log_model(model, "model")
        
        # Loggear reporte de clasificación
        report = classification_report(y_test, y_pred, output_dict=True)
        mlflow.log_text(str(report), "classification_report.txt")
        
        # Loggear algunos ejemplos de predicciones
        examples = pd.DataFrame({
            'true_label': y_test[:10],
            'predicted_label': y_pred[:10],
            'prediction_probability': y_pred_proba[:10, 1]
        })
        mlflow.log_text(examples.to_string(), "prediction_examples.txt")
        
        print("✅ Logging completado exitosamente!")
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print(f"Experimento: {mlflow.active_run().info.experiment_id}")
        
        return {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "model": model
        }

if __name__ == "__main__":
    test_mlflow_logging()

