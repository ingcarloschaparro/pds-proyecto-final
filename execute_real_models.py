#!/usr/bin/env python3
"""Script para ejecutar los 4 modelos reales y obtener métricas genuinas"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import time
import warnings
import os
from pathlib import Path
from typing import Dict, List, Any
import re

warnings.filterwarnings("ignore")

# Configurar MLflow para AWS
MLFLOW_TRACKING_URI = "http://52.0.127.25:5001"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def load_test_data(sample_size=100):
    """Cargar datos de prueba para los modelos"""
    print("Cargando datos de prueba...")
    
    df = pd.read_csv("data/processed/dataset_clean_v1.csv", low_memory=False)
    df_valid = df[df["label"].notna()].copy()
    
    # Preparar textos para generación PLS
    pls_texts = []
    for _, row in df_valid.iterrows():
        if row["label"] == "non_pls" and pd.notna(row["texto_original"]):
            texto = str(row["texto_original"]).strip()
            if len(texto) > 50:  # Textos suficientemente largos
                pls_texts.append(texto)
    
    # Tomar muestra
    if len(pls_texts) > sample_size:
        pls_texts = np.random.choice(pls_texts, sample_size, replace=False).tolist()
    
    print(f"✅ {len(pls_texts)} textos cargados para generación PLS")
    return pls_texts

def calculate_readability_metrics(text):
    """Calcular métricas de legibilidad"""
    # Flesch Reading Ease (simplificado)
    sentences = len(re.split(r'[.!?]+', text))
    words = len(text.split())
    syllables = sum([len(re.findall(r'[aeiouAEIOU]', word)) for word in text.split()])
    
    if sentences > 0 and words > 0:
        flesch = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
    else:
        flesch = 0
    
    # FKGL (simplificado)
    if sentences > 0 and words > 0:
        fkgl = 0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59
    else:
        fkgl = 0
    
    return {
        "flesch_score": max(0, min(100, flesch)),
        "fkgl_score": max(0, fkgl),
        "word_count": words,
        "sentence_count": sentences
    }

def execute_t5_base(texts):
    """Ejecutar modelo T5-Base real"""
    print("\n=== EJECUTANDO T5-BASE ===")
    
    experiment_name = "E2-T5-Base-Real"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f"t5_base_real_{int(time.time())}"):
        try:
            from transformers import pipeline
            
            # Cargar modelo T5-Base
            print("Cargando T5-Base...")
            summarizer = pipeline(
                "summarization",
                model="t5-base",
                max_length=100,
                min_length=30,
                num_beams=4,
                temperature=0.8,
                do_sample=True
            )
            
            # Loggear parámetros
            mlflow.log_params({
                "model_name": "t5-base",
                "model_type": "T5-Base",
                "architecture": "transformer",
                "task": "summarization",
                "max_length": 100,
                "min_length": 30,
                "num_beams": 4,
                "temperature": 0.8,
                "do_sample": True,
                "test_samples": len(texts)
            })
            
            # Procesar textos
            results = []
            total_time = 0
            
            for i, text in enumerate(texts[:10]):  # Procesar solo 10 para prueba
                print(f"Procesando texto {i+1}/10...")
                
                start_time = time.time()
                try:
                    # Truncar texto si es muy largo
                    if len(text) > 512:
                        text = text[:512]
                    
                    summary = summarizer(text, max_length=100, min_length=30)[0]['summary_text']
                    processing_time = time.time() - start_time
                    total_time += processing_time
                    
                    # Calcular métricas
                    original_metrics = calculate_readability_metrics(text)
                    summary_metrics = calculate_readability_metrics(summary)
                    
                    compression_ratio = len(summary) / len(text) if len(text) > 0 else 0
                    
                    results.append({
                        "original_length": len(text),
                        "summary_length": len(summary),
                        "compression_ratio": compression_ratio,
                        "original_flesch": original_metrics["flesch_score"],
                        "summary_flesch": summary_metrics["flesch_score"],
                        "original_fkgl": original_metrics["fkgl_score"],
                        "summary_fkgl": summary_metrics["fkgl_score"],
                        "processing_time": processing_time
                    })
                    
                except Exception as e:
                    print(f"Error procesando texto {i+1}: {e}")
                    continue
            
            if results:
                # Calcular métricas promedio
                avg_compression = np.mean([r["compression_ratio"] for r in results])
                avg_flesch = np.mean([r["summary_flesch"] for r in results])
                avg_fkgl = np.mean([r["summary_fkgl"] for r in results])
                avg_time = np.mean([r["processing_time"] for r in results])
                
                # Loggear métricas
                mlflow.log_metrics({
                    "compression_ratio": avg_compression,
                    "flesch_score": avg_flesch,
                    "fkgl_score": avg_fkgl,
                    "avg_processing_time": avg_time,
                    "total_processing_time": total_time,
                    "successful_generations": len(results),
                    "success_rate": len(results) / min(10, len(texts))
                })
                
                print(f"✅ T5-Base completado")
                print(f"   Compresión promedio: {avg_compression:.3f}")
                print(f"   Flesch promedio: {avg_flesch:.1f}")
                print(f"   FKGL promedio: {avg_fkgl:.1f}")
                print(f"   Tiempo promedio: {avg_time:.2f}s")
                
                return True
            else:
                print("❌ No se pudieron procesar textos con T5-Base")
                return False
                
        except Exception as e:
            print(f"❌ Error ejecutando T5-Base: {e}")
            mlflow.log_param("error", str(e))
            return False

def execute_bart_base(texts):
    """Ejecutar modelo BART-Base real"""
    print("\n=== EJECUTANDO BART-BASE ===")
    
    experiment_name = "E2-BART-Base-Real"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f"bart_base_real_{int(time.time())}"):
        try:
            from transformers import pipeline
            
            # Cargar modelo BART-Base
            print("Cargando BART-Base...")
            summarizer = pipeline(
                "summarization",
                model="facebook/bart-base",
                max_length=120,
                min_length=40,
                num_beams=4,
                temperature=0.7,
                do_sample=True
            )
            
            # Loggear parámetros
            mlflow.log_params({
                "model_name": "facebook/bart-base",
                "model_type": "BART-Base",
                "architecture": "transformer",
                "task": "summarization",
                "max_length": 120,
                "min_length": 40,
                "num_beams": 4,
                "temperature": 0.7,
                "do_sample": True,
                "test_samples": len(texts)
            })
            
            # Procesar textos
            results = []
            total_time = 0
            
            for i, text in enumerate(texts[:10]):
                print(f"Procesando texto {i+1}/10...")
                
                start_time = time.time()
                try:
                    if len(text) > 512:
                        text = text[:512]
                    
                    summary = summarizer(text, max_length=120, min_length=40)[0]['summary_text']
                    processing_time = time.time() - start_time
                    total_time += processing_time
                    
                    # Calcular métricas
                    original_metrics = calculate_readability_metrics(text)
                    summary_metrics = calculate_readability_metrics(summary)
                    
                    compression_ratio = len(summary) / len(text) if len(text) > 0 else 0
                    
                    results.append({
                        "original_length": len(text),
                        "summary_length": len(summary),
                        "compression_ratio": compression_ratio,
                        "original_flesch": original_metrics["flesch_score"],
                        "summary_flesch": summary_metrics["flesch_score"],
                        "original_fkgl": original_metrics["fkgl_score"],
                        "summary_fkgl": summary_metrics["fkgl_score"],
                        "processing_time": processing_time
                    })
                    
                except Exception as e:
                    print(f"Error procesando texto {i+1}: {e}")
                    continue
            
            if results:
                # Calcular métricas promedio
                avg_compression = np.mean([r["compression_ratio"] for r in results])
                avg_flesch = np.mean([r["summary_flesch"] for r in results])
                avg_fkgl = np.mean([r["summary_fkgl"] for r in results])
                avg_time = np.mean([r["processing_time"] for r in results])
                
                # Loggear métricas
                mlflow.log_metrics({
                    "compression_ratio": avg_compression,
                    "flesch_score": avg_flesch,
                    "fkgl_score": avg_fkgl,
                    "avg_processing_time": avg_time,
                    "total_processing_time": total_time,
                    "successful_generations": len(results),
                    "success_rate": len(results) / min(10, len(texts))
                })
                
                print(f"✅ BART-Base completado")
                print(f"   Compresión promedio: {avg_compression:.3f}")
                print(f"   Flesch promedio: {avg_flesch:.1f}")
                print(f"   FKGL promedio: {avg_fkgl:.1f}")
                print(f"   Tiempo promedio: {avg_time:.2f}s")
                
                return True
            else:
                print("❌ No se pudieron procesar textos con BART-Base")
                return False
                
        except Exception as e:
            print(f"❌ Error ejecutando BART-Base: {e}")
            mlflow.log_param("error", str(e))
            return False

def execute_bart_large_cnn(texts):
    """Ejecutar modelo BART-Large-CNN real"""
    print("\n=== EJECUTANDO BART-LARGE-CNN ===")
    
    experiment_name = "E2-BART-Large-CNN-Real"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f"bart_large_cnn_real_{int(time.time())}"):
        try:
            from transformers import pipeline
            
            # Cargar modelo BART-Large-CNN
            print("Cargando BART-Large-CNN...")
            summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                max_length=150,
                min_length=50,
                num_beams=4,
                temperature=0.6,
                do_sample=True
            )
            
            # Loggear parámetros
            mlflow.log_params({
                "model_name": "facebook/bart-large-cnn",
                "model_type": "BART-Large-CNN",
                "architecture": "transformer",
                "task": "summarization",
                "max_length": 150,
                "min_length": 50,
                "num_beams": 4,
                "temperature": 0.6,
                "do_sample": True,
                "test_samples": len(texts)
            })
            
            # Procesar textos
            results = []
            total_time = 0
            
            for i, text in enumerate(texts[:10]):
                print(f"Procesando texto {i+1}/10...")
                
                start_time = time.time()
                try:
                    if len(text) > 1024:
                        text = text[:1024]
                    
                    summary = summarizer(text, max_length=150, min_length=50)[0]['summary_text']
                    processing_time = time.time() - start_time
                    total_time += processing_time
                    
                    # Calcular métricas
                    original_metrics = calculate_readability_metrics(text)
                    summary_metrics = calculate_readability_metrics(summary)
                    
                    compression_ratio = len(summary) / len(text) if len(text) > 0 else 0
                    
                    results.append({
                        "original_length": len(text),
                        "summary_length": len(summary),
                        "compression_ratio": compression_ratio,
                        "original_flesch": original_metrics["flesch_score"],
                        "summary_flesch": summary_metrics["flesch_score"],
                        "original_fkgl": original_metrics["fkgl_score"],
                        "summary_fkgl": summary_metrics["fkgl_score"],
                        "processing_time": processing_time
                    })
                    
                except Exception as e:
                    print(f"Error procesando texto {i+1}: {e}")
                    continue
            
            if results:
                # Calcular métricas promedio
                avg_compression = np.mean([r["compression_ratio"] for r in results])
                avg_flesch = np.mean([r["summary_flesch"] for r in results])
                avg_fkgl = np.mean([r["summary_fkgl"] for r in results])
                avg_time = np.mean([r["processing_time"] for r in results])
                
                # Loggear métricas
                mlflow.log_metrics({
                    "compression_ratio": avg_compression,
                    "flesch_score": avg_flesch,
                    "fkgl_score": avg_fkgl,
                    "avg_processing_time": avg_time,
                    "total_processing_time": total_time,
                    "successful_generations": len(results),
                    "success_rate": len(results) / min(10, len(texts))
                })
                
                print(f"✅ BART-Large-CNN completado")
                print(f"   Compresión promedio: {avg_compression:.3f}")
                print(f"   Flesch promedio: {avg_flesch:.1f}")
                print(f"   FKGL promedio: {avg_fkgl:.1f}")
                print(f"   Tiempo promedio: {avg_time:.2f}s")
                
                return True
            else:
                print("❌ No se pudieron procesar textos con BART-Large-CNN")
                return False
                
        except Exception as e:
            print(f"❌ Error ejecutando BART-Large-CNN: {e}")
            mlflow.log_param("error", str(e))
            return False

def execute_pls_ligero(texts):
    """Ejecutar modelo PLS Ligero (rule-based) real"""
    print("\n=== EJECUTANDO PLS LIGERO ===")
    
    experiment_name = "E2-PLS-Ligero-Real"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f"pls_ligero_real_{int(time.time())}"):
        try:
            # Implementar reglas de simplificación
            def simplify_text(text):
                """Simplificar texto usando reglas"""
                # Diccionario de simplificaciones
                simplifications = {
                    r'\brandomized\b': 'randomly assigned',
                    r'\bplacebo-controlled\b': 'compared to placebo',
                    r'\bdouble-blind\b': 'neither doctors nor patients knew',
                    r'\bclinical trial\b': 'study',
                    r'\bparticipants\b': 'people',
                    r'\bintervention\b': 'treatment',
                    r'\boutcome\b': 'result',
                    r'\bstatistically significant\b': 'meaningful difference',
                    r'\bcohort\b': 'group',
                    r'\bprotocol\b': 'plan',
                    r'\brandomization\b': 'random assignment',
                    r'\bplacebo\b': 'inactive treatment',
                    r'\bprimary endpoint\b': 'main goal',
                    r'\bsecondary endpoint\b': 'additional goal',
                    r'\binclusion criteria\b': 'requirements to join',
                    r'\bexclusion criteria\b': 'reasons not to join',
                    r'\badverse events\b': 'side effects',
                    r'\bsafety profile\b': 'safety information',
                    r'\befficacy\b': 'effectiveness',
                    r'\btolerability\b': 'how well it was tolerated'
                }
                
                simplified = text
                for pattern, replacement in simplifications.items():
                    simplified = re.sub(pattern, replacement, simplified, flags=re.IGNORECASE)
                
                return simplified
            
            # Loggear parámetros
            mlflow.log_params({
                "model_name": "pls_lightweight",
                "model_type": "PLS Ligero",
                "architecture": "rule_based",
                "task": "text_simplification",
                "rules_count": 20,
                "processing_type": "regex_based",
                "test_samples": len(texts)
            })
            
            # Procesar textos
            results = []
            total_time = 0
            
            for i, text in enumerate(texts[:10]):
                print(f"Procesando texto {i+1}/10...")
                
                start_time = time.time()
                try:
                    simplified = simplify_text(text)
                    processing_time = time.time() - start_time
                    total_time += processing_time
                    
                    # Calcular métricas
                    original_metrics = calculate_readability_metrics(text)
                    simplified_metrics = calculate_readability_metrics(simplified)
                    
                    expansion_ratio = len(simplified) / len(text) if len(text) > 0 else 0
                    
                    results.append({
                        "original_length": len(text),
                        "simplified_length": len(simplified),
                        "expansion_ratio": expansion_ratio,
                        "original_flesch": original_metrics["flesch_score"],
                        "simplified_flesch": simplified_metrics["flesch_score"],
                        "original_fkgl": original_metrics["fkgl_score"],
                        "simplified_fkgl": simplified_metrics["fkgl_score"],
                        "processing_time": processing_time
                    })
                    
                except Exception as e:
                    print(f"Error procesando texto {i+1}: {e}")
                    continue
            
            if results:
                # Calcular métricas promedio
                avg_expansion = np.mean([r["expansion_ratio"] for r in results])
                avg_flesch = np.mean([r["simplified_flesch"] for r in results])
                avg_fkgl = np.mean([r["simplified_fkgl"] for r in results])
                avg_time = np.mean([r["processing_time"] for r in results])
                
                # Loggear métricas
                mlflow.log_metrics({
                    "expansion_ratio": avg_expansion,
                    "flesch_score": avg_flesch,
                    "fkgl_score": avg_fkgl,
                    "avg_processing_time": avg_time,
                    "total_processing_time": total_time,
                    "successful_generations": len(results),
                    "success_rate": len(results) / min(10, len(texts))
                })
                
                print(f"✅ PLS Ligero completado")
                print(f"   Expansión promedio: {avg_expansion:.3f}")
                print(f"   Flesch promedio: {avg_flesch:.1f}")
                print(f"   FKGL promedio: {avg_fkgl:.1f}")
                print(f"   Tiempo promedio: {avg_time:.4f}s")
                
                return True
            else:
                print("❌ No se pudieron procesar textos con PLS Ligero")
                return False
                
        except Exception as e:
            print(f"❌ Error ejecutando PLS Ligero: {e}")
            mlflow.log_param("error", str(e))
            return False

def main():
    """Función principal para ejecutar los 4 modelos reales"""
    print("🚀 EJECUTANDO LOS 4 MODELOS REALES")
    print("=" * 50)
    
    # Cargar datos de prueba
    texts = load_test_data(sample_size=50)
    
    if not texts:
        print("❌ No se pudieron cargar datos de prueba")
        return
    
    # Ejecutar cada modelo
    models = [
        ("T5-Base", execute_t5_base),
        ("BART-Base", execute_bart_base),
        ("BART-Large-CNN", execute_bart_large_cnn),
        ("PLS Ligero", execute_pls_ligero)
    ]
    
    results = []
    
    for model_name, model_func in models:
        try:
            print(f"\n{'='*20} {model_name} {'='*20}")
            success = model_func(texts)
            results.append((model_name, success))
        except Exception as e:
            print(f"❌ Error crítico en {model_name}: {e}")
            results.append((model_name, False))
    
    # Resumen final
    print("\n" + "=" * 50)
    print("📊 RESUMEN DE EJECUCIÓN")
    print("=" * 50)
    
    successful = 0
    for model_name, success in results:
        status = "✅ EXITOSO" if success else "❌ FALLÓ"
        print(f"{status} {model_name}")
        if success:
            successful += 1
    
    print(f"\n🎯 Resultado: {successful}/{len(models)} modelos ejecutados exitosamente")
    
    if successful > 0:
        print(f"\n🌐 Ve a: {MLFLOW_TRACKING_URI}")
        print("   para ver los experimentos reales con métricas genuinas")
    
    return successful == len(models)

if __name__ == "__main__":
    main()
