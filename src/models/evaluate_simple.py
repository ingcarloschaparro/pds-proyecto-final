#!/usr/bin/env python3
"""
Evaluación simplificada de modelos PLS
"""
import pandas as pd
import numpy as np
import json
import os
import sys
from pathlib import Path
import textstat

# Agregar path para importar scripts
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append("scripts")

def calcular_metricas_legibilidad(textos):
    """Calcular métricas de legibilidad"""
    fkgl_scores = []
    flesch_scores = []
    
    for texto in textos:
        if texto and len(str(texto).strip()) > 10:
            try:
                fkgl = textstat.flesch_kincaid_grade(texto)
                flesch = textstat.flesch_reading_ease(texto)
                fkgl_scores.append(fkgl)
                flesch_scores.append(flesch)
            except:
                pass
    
    return {
        "fkgl_mean": np.mean(fkgl_scores) if fkgl_scores else 0.0,
        "flesch_mean": np.mean(flesch_scores) if flesch_scores else 0.0
    }

def calcular_compresion(originales, pls):
    """Calcular métricas de compresión"""
    ratios = []
    for orig, pls_text in zip(originales, pls):
        if orig and pls_text:
            ratio = len(pls_text) / len(orig) if len(orig) > 0 else 0
            ratios.append(ratio)
    
    return {
        "ratio_compresion": np.mean(ratios) if ratios else 0.0
    }

def evaluar_pls_ligero(textos):
    """Evaluar PLS Ligero (rule-based)"""
    print("Evaluando PLS Ligero...")
    
    pls_generados = []
    for texto in textos:
        # PLS Ligero simple
        pls = f"En términos simples: {texto[:100]}..."
        pls_generados.append(pls)
    
    metricas_leg = calcular_metricas_legibilidad(pls_generados)
    metricas_comp = calcular_compresion(textos, pls_generados)
    
    return {**metricas_leg, **metricas_comp}

def main():
    """Función principal"""
    print("=== EVALUACIÓN SIMPLIFICADA DE MODELOS PLS ===")
    
    # Cargar dataset
    test_file = "data/processed/test.csv"
    if not os.path.exists(test_file):
        print(f"Error: No se encontró {test_file}")
        return
    
    df = pd.read_csv(test_file)
    print(f"Dataset cargado: {len(df)} registros")
    
    # Muestra pequeña
    sample_size = min(20, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    textos = df_sample['texto_original'].tolist()
    
    print(f"Evaluando con {len(textos)} textos")
    
    # Evaluar PLS Ligero
    resultados = {}
    resultados['pls_ligero'] = evaluar_pls_ligero(textos)
    
    # Crear directorio de salida
    output_dir = "data/evaluation"
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar resultados
    resultados_final = {
        "dataset_info": {
            "total_registros": len(df),
            "muestra_evaluada": len(textos)
        },
        "modelos_evaluados": resultados
    }
    
    with open(f"{output_dir}/evaluacion_simple.json", "w", encoding="utf-8") as f:
        json.dump(resultados_final, f, indent=2, ensure_ascii=False)
    
    # Mostrar resultados
    print("\n=== RESULTADOS ===")
    for modelo, metricas in resultados.items():
        print(f"\n{modelo}:")
        print(f"  FKGL: {metricas.get('fkgl_mean', 0):.2f}")
        print(f"  Flesch: {metricas.get('flesch_mean', 0):.2f}")
        print(f"  Compresión: {metricas.get('ratio_compresion', 0):.3f}")
    
    print(f"\nResultados guardados en: {output_dir}/evaluacion_simple.json")

if __name__ == "__main__":
    main()
