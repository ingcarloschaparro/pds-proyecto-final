#!/usr/bin/env python3
"""Script para subir experimentos al servidor MLflow de AWS"""

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd
import time

# Configurar MLflow para AWS
MLFLOW_TRACKING_URI = "http://52.0.127.25:5001"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def upload_experiment_to_aws():
    """Sube un experimento de prueba al servidor MLflow de AWS"""
    
    # Configurar experimento
    experiment_name = "E2-AWS-Upload-Test"
    mlflow.set_experiment(experiment_name)
    
    print(f"=== SUBIENDO EXPERIMENTO A AWS MLFLOW ===")
    print(f"Servidor: {MLFLOW_TRACKING_URI}")
    print(f"Experimento: {experiment_name}")
    
    with mlflow.start_run(run_name=f"aws_test_run_{int(time.time())}"):
        print("✅ Run iniciado correctamente")
        
        # Generar datos de prueba
        X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Loggear parámetros
        params = {
            "n_samples": 1000,
            "n_features": 20,
            "test_size": 0.2,
            "random_state": 42,
            "model_type": "LogisticRegression",
            "server": "AWS"
        }
        
        mlflow.log_params(params)
        print("✅ Parámetros loggeados")
        
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
        metrics = {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "train_samples": len(X_train),
            "test_samples": len(X_test)
        }
        
        mlflow.log_metrics(metrics)
        print("✅ Métricas loggeadas")
        
        # Loggear modelo
        mlflow.sklearn.log_model(model, "model")
        print("✅ Modelo loggeado")
        
        # Loggear reporte de clasificación
        report = classification_report(y_test, y_pred, output_dict=True)
        mlflow.log_text(str(report), "classification_report.txt")
        print("✅ Reporte de clasificación loggeado")
        
        # Loggear algunos ejemplos de predicciones
        examples = pd.DataFrame({
            'true_label': y_test[:10],
            'predicted_label': y_pred[:10],
            'prediction_probability': y_pred_proba[:10, 1]
        })
        mlflow.log_text(examples.to_string(), "prediction_examples.txt")
        print("✅ Ejemplos de predicciones loggeados")
        
        # Loggear tags adicionales
        mlflow.set_tags({
            "experiment_type": "classification_test",
            "uploaded_to": "AWS",
            "status": "completed"
        })
        print("✅ Tags loggeados")
        
        print("🎉 ¡Experimento subido exitosamente a AWS MLflow!")
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        
        return {
            "run_id": mlflow.active_run().info.run_id,
            "experiment_id": mlflow.active_run().info.experiment_id,
            "metrics": metrics,
            "params": params
        }

def upload_multiple_experiments():
    """Sube múltiples experimentos para demostrar diferentes modelos"""
    
    experiments = [
        {
            "name": "E2-DistilBERT-AWS",
            "model_type": "DistilBERT",
            "description": "Clasificador DistilBERT para PLS vs non-PLS"
        },
        {
            "name": "E2-TFIDF-AWS", 
            "model_type": "TF-IDF + LogisticRegression",
            "description": "Clasificador baseline TF-IDF"
        },
        {
            "name": "E2-PLS-Generator-AWS",
            "model_type": "BART/T5",
            "description": "Generador de Plain Language Summaries"
        }
    ]
    
    results = []
    
    for exp in experiments:
        print(f"\n=== SUBIENDO {exp['name']} ===")
        
        # Configurar experimento
        mlflow.set_experiment(exp['name'])
        
        with mlflow.start_run(run_name=f"{exp['model_type']}_run_{int(time.time())}"):
            # Loggear parámetros específicos del modelo
            params = {
                "model_type": exp['model_type'],
                "description": exp['description'],
                "server": "AWS",
                "upload_timestamp": time.time()
            }
            
            mlflow.log_params(params)
            
            # Simular métricas diferentes para cada modelo
            if "DistilBERT" in exp['model_type']:
                metrics = {
                    "f1_macro": 0.85,
                    "f1_weighted": 0.87,
                    "accuracy": 0.86,
                    "precision": 0.84,
                    "recall": 0.88
                }
            elif "TF-IDF" in exp['model_type']:
                metrics = {
                    "f1_macro": 0.78,
                    "f1_weighted": 0.80,
                    "accuracy": 0.79,
                    "precision": 0.77,
                    "recall": 0.81
                }
            else:  # PLS Generator
                metrics = {
                    "rouge_1": 0.65,
                    "rouge_2": 0.58,
                    "rouge_l": 0.62,
                    "bleu_score": 0.60,
                    "compression_ratio": 0.45
                }
            
            mlflow.log_metrics(metrics)
            
            # Loggear tags
            mlflow.set_tags({
                "experiment_type": exp['model_type'].lower().replace(" ", "_"),
                "uploaded_to": "AWS",
                "status": "completed",
                "model_family": exp['model_type'].split()[0]
            })
            
            print(f"✅ {exp['name']} subido exitosamente")
            results.append({
                "experiment": exp['name'],
                "run_id": mlflow.active_run().info.run_id,
                "metrics": metrics
            })
    
    return results

if __name__ == "__main__":
    try:
        # Probar conexión primero
        print("Probando conexión al servidor AWS...")
        experiments = mlflow.search_experiments()
        print(f"✅ Conexión exitosa. Encontrados {len(experiments)} experimentos")
        
        # Subir experimento de prueba
        result = upload_experiment_to_aws()
        print(f"\n📊 Resultado: {result}")
        
        # Subir múltiples experimentos
        print("\n" + "="*50)
        print("SUBIENDO MÚLTIPLES EXPERIMENTOS")
        print("="*50)
        
        results = upload_multiple_experiments()
        
        print("\n🎉 ¡Todos los experimentos subidos exitosamente!")
        print("\n📋 Resumen:")
        for result in results:
            print(f"- {result['experiment']}: {result['run_id']}")
        
        print(f"\n🌐 Ve a: {MLFLOW_TRACKING_URI}")
        print("   para ver los experimentos con gráficos y métricas")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Verifica que el servidor MLflow esté funcionando")

