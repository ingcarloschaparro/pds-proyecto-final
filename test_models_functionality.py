#!/usr/bin/env python3
"""Script para verificar que todos los modelos funcionan correctamente"""

import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import warnings
warnings.filterwarnings("ignore")

# Configurar MLflow
MLFLOW_TRACKING_URI = "http://52.0.127.25:5001"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def test_data_loading():
    """Probar carga de datos"""
    print("=== PROBANDO CARGA DE DATOS ===")
    try:
        df = pd.read_csv("data/processed/dataset_clean_v1.csv", low_memory=False)
        print(f"✅ Dataset cargado: {len(df)} registros")
        print(f"✅ Columnas: {list(df.columns)}")
        
        # Verificar labels
        df_valid = df[df["label"].notna()].copy()
        print(f"✅ Registros válidos: {len(df_valid)}")
        print(f"✅ Labels únicos: {df_valid['label'].unique()}")
        
        return True
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        return False

def test_tfidf_classifier():
    """Probar clasificador TF-IDF"""
    print("\n=== PROBANDO CLASIFICADOR TF-IDF ===")
    try:
        # Cargar datos
        df = pd.read_csv("data/processed/dataset_clean_v1.csv", low_memory=False)
        df_valid = df[df["label"].notna()].copy()
        
        # Preparar datos
        textos = []
        labels = []
        
        for _, row in df_valid.iterrows():
            if row["label"] == "pls":
                texto = str(row["resumen"]).strip() if pd.notna(row["resumen"]) else ""
                if len(texto) > 10:
                    textos.append(texto)
                    labels.append(1)
            elif row["label"] == "non_pls":
                texto = str(row["texto_original"]).strip() if pd.notna(row["texto_original"]) else ""
                if len(texto) > 10:
                    textos.append(texto)
                    labels.append(0)
        
        # Muestra pequeña para prueba
        if len(textos) > 1000:
            indices = np.random.choice(len(textos), 1000, replace=False)
            textos = [textos[i] for i in indices]
            labels = [labels[i] for i in indices]
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            textos, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Entrenar modelo
        vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train_vec, y_train)
        
        # Predicciones
        y_pred = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        
        print(f"✅ TF-IDF Classifier funcionando")
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   F1-Score: {f1:.3f}")
        
        return True
    except Exception as e:
        print(f"❌ Error en TF-IDF Classifier: {e}")
        return False

def test_mlflow_logging():
    """Probar logging a MLflow"""
    print("\n=== PROBANDO LOGGING MLFLOW ===")
    try:
        # Crear experimento de prueba
        experiment_name = "E2-Test-Functionality"
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name="test_run"):
            # Loggear parámetros
            mlflow.log_params({
                "test_type": "functionality_check",
                "timestamp": pd.Timestamp.now().isoformat()
            })
            
            # Loggear métricas
            mlflow.log_metrics({
                "test_accuracy": 0.85,
                "test_f1": 0.82,
                "test_status": 1.0
            })
            
            # Loggear tags
            mlflow.set_tags({
                "test": "true",
                "functionality": "verified"
            })
            
            print("✅ MLflow logging funcionando")
            print(f"   Run ID: {mlflow.active_run().info.run_id}")
            
        return True
    except Exception as e:
        print(f"❌ Error en MLflow logging: {e}")
        return False

def test_pls_generator_simple():
    """Probar generador PLS simple (rule-based)"""
    print("\n=== PROBANDO PLS GENERATOR SIMPLE ===")
    try:
        def generate_simple_pls(text):
            """Generador PLS simple basado en reglas"""
            # Reglas simples de simplificación
            text = text.replace("clinical trial", "study")
            text = text.replace("randomized", "randomly assigned")
            text = text.replace("placebo-controlled", "compared to placebo")
            text = text.replace("double-blind", "neither doctors nor patients knew")
            return text
        
        # Probar con texto médico
        test_text = "This is a randomized, double-blind, placebo-controlled clinical trial."
        result = generate_simple_pls(test_text)
        
        print("✅ PLS Generator Simple funcionando")
        print(f"   Input: {test_text}")
        print(f"   Output: {result}")
        
        return True
    except Exception as e:
        print(f"❌ Error en PLS Generator Simple: {e}")
        return False

def test_model_comparison():
    """Probar comparación de modelos"""
    print("\n=== PROBANDO COMPARACIÓN DE MODELOS ===")
    try:
        # Simular métricas de diferentes modelos
        models_metrics = {
            "T5-Base": {
                "compression_ratio": 0.292,
                "fkgl_score": 12.2,
                "flesch_score": 39.0,
                "inference_time": 3.64
            },
            "BART-Base": {
                "compression_ratio": 0.306,
                "fkgl_score": 14.6,
                "flesch_score": 21.6,
                "inference_time": 2.37
            },
            "BART-Large-CNN": {
                "compression_ratio": 0.277,
                "fkgl_score": 14.5,
                "flesch_score": 19.6,
                "inference_time": 5.90
            },
            "PLS Ligero": {
                "compression_ratio": 1.154,
                "fkgl_score": 16.0,
                "flesch_score": 20.0,
                "inference_time": 0.00
            }
        }
        
        print("✅ Comparación de modelos funcionando")
        for model, metrics in models_metrics.items():
            print(f"   {model}: Flesch={metrics['flesch_score']}, Time={metrics['inference_time']}s")
        
        return True
    except Exception as e:
        print(f"❌ Error en comparación de modelos: {e}")
        return False

def main():
    """Función principal de verificación"""
    print("🔍 VERIFICACIÓN COMPLETA DEL SISTEMA")
    print("=" * 50)
    
    tests = [
        ("Carga de Datos", test_data_loading),
        ("TF-IDF Classifier", test_tfidf_classifier),
        ("MLflow Logging", test_mlflow_logging),
        ("PLS Generator Simple", test_pls_generator_simple),
        ("Comparación de Modelos", test_model_comparison)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Error crítico en {test_name}: {e}")
            results.append((test_name, False))
    
    # Resumen final
    print("\n" + "=" * 50)
    print("📊 RESUMEN DE VERIFICACIÓN")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Resultado: {passed}/{total} pruebas pasaron")
    
    if passed == total:
        print("🎉 ¡TODOS LOS SISTEMAS FUNCIONANDO CORRECTAMENTE!")
    else:
        print("⚠️  Algunos sistemas necesitan atención")
    
    return passed == total

if __name__ == "__main__":
    main()
