#!/usr/bin/env python3
"""Script para evaluar la calidad de los por favor generados"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import json

def calculate_readability_metrics(text):
    """Calcular métricas de legibilidad"""
    try:
        import textstat

        metrics = {
            "flesch_reading_ease": textstat.flesch_reading_ease(text),
            "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
            "gunning_fog": textstat.gunning_fog(text),
            "smog_index": textstat.smog_index(text),
            "automated_readability_index": textstat.automated_readability_index(text),
            "coleman_liau_index": textstat.coleman_liau_index(text)
        }

        return metrics
    except ImportError:
        print("textstat no está disponible. Instalando...")
        return None

def calculate_basic_metrics(text):
    """Calcular métricas básicas de texto"""
    words = text.split()
    sentences = text.split(".")

    metrics = {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "avg_words_per_sentence": len(words) / len(sentences) if sentences else 0,
        "avg_word_length": np.mean([len(word) for word in words]) if words else 0,
        "char_count": len(text),
        "char_count_no_spaces": len(text.replace("", ""))
    }

    return metrics

def evaluate_pls_quality(df_results):
    """Evaluar la calidad de los por favor generados"""
    print("EVALUANDO CALIDAD DE por favor GENERADOS")
    print("=" * 50)

    results = []

    for idx, row in df_results.iterrows():
        print(f"\si--- Evaluando por favor {row["id"]} ---")

        # Métricas básicas
        basic_metrics = calculate_basic_metrics(row["pls_generado"])

        # Métricas de legibilidad
        readability_metrics = calculate_readability_metrics(row["pls_generado"])

        # Métricas de compresión
        compression_ratio = row["compression_ratio"]

        # Análisis de calidad
        quality_analysis = analyze_quality(row["texto_original"], row["pls_generado"])

        # Combinar métricas
        evaluation = {
            "id": row["id"],
            "basic_metrics": basic_metrics,
            "readability_metrics": readability_metrics,
            "compression_ratio": compression_ratio,
            "quality_analysis": quality_analysis
        }

        results.append(evaluation)

        # Mostrar resumen
        print(f"Palabras: {basic_metrics["word_count"]}")
        print(f"Oraciones: {basic_metrics["sentence_count"]}")
        print(f"Compresión: {compression_ratio:.2f}")

        if readability_metrics:
            print(f"Flesch Reading Ease: {readability_metrics["flesch_reading_ease"]:.1f}")
            print(f"Flesch-Kincaid Grade: {readability_metrics["flesch_kincaid_grade"]:.1f}")

        print(f"Calidad: {quality_analysis["overall_quality"]}")

    return results

def analyze_quality(original, pls):
    """Analizar calidad del por favor generado"""

    # Verificar si está en español
    spanish_indicators = ["en términos simples", "el estudio", "los pacientes", "medicamento", "cirugía"]
    spanish_score = sum(1 for indicator in spanish_indicators if indicator.lower() in pls.lower())

    # Verificar simplificación de términos médicos
    medical_terms_simplified = [
        "medicamento para la diabetes",
        "control del azúcar en sangre",
        "diabetes tipo a|tambien",
        "medicamento sin efecto real",
        "cirugía mínimamente invasiva",
        "piedras en la vesícula",
        "medicamento para el colesterol",
        "problemas del corazón",
        "colesterol alto",
        "ataque al corazón"
    ]

    simplification_score = sum(1 for term in medical_terms_simplified if term in pls.lower())

    # Verificar longitud apropiada
    word_count = len(pls.split())
    length_score = 1 if 20 <= word_count <= 80 else 0

    # Verificar que no sea solo copia del original
    similarity = len(set(original.lower().split()) & set(pls.lower().split())) / len(set(original.lower().split()))
    originality_score = 1 if similarity < 0.8 else 0

    # Calcular puntuación general
    total_score = spanish_score + simplification_score + length_score + originality_score
    max_score = len(spanish_indicators) + len(medical_terms_simplified) + 2

    quality_percentage = (total_score / max_score) * 100

    if quality_percentage >= 80:
        overall_quality = "Excelente"
    elif quality_percentage >= 60:
        overall_quality = "Buena"
    elif quality_percentage >= 40:
        overall_quality = "Regular"
    else:
        overall_quality = "Necesita mejora"

    return {
        "spanish_score": spanish_score,
        "simplification_score": simplification_score,
        "length_score": length_score,
        "originality_score": originality_score,
        "total_score": total_score,
        "max_score": max_score,
        "quality_percentage": quality_percentage,
        "overall_quality": overall_quality,
        "similarity_to_original": similarity
    }

def generate_evaluation_report(evaluations):
    """Generar reporte de evaluación"""
    print("\si REPORTE DE EVALUACIÓN")
    print("=" * 50)

    # Estadísticas generales
    total_evaluations = len(evaluations)
    quality_counts = {}

    for eval_data in evaluations:
        quality = eval_data["quality_analysis"]["overall_quality"]
        quality_counts[quality] = quality_counts.get(quality, 0) + 1

    print(f"Total de por favor evaluados: {total_evaluations}")
    print("\nDistribución de calidad:")
    for quality, count in quality_counts.items():
        percentage = (count / total_evaluations) * 100
        print(f"{quality}: {count} ({percentage:.1f}%)")

    # Métricas promedio
    if evaluations:
        avg_compression = np.mean([eval_data["compression_ratio"] for eval_data in evaluations])
        avg_quality_percentage = np.mean([eval_data["quality_analysis"]["quality_percentage"] for eval_data in evaluations])

        print(f"\nMétricas promedio:")
        print(f"Compresión promedio: {avg_compression:.2f}")
        print(f"Calidad promedio: {avg_quality_percentage:.1f}%")

        # Análisis de legibilidad
        readability_scores = []
        for eval_data in evaluations:
            if eval_data["readability_metrics"]:
                readability_scores.append(eval_data["readability_metrics"]["flesch_reading_ease"])

        if readability_scores:
            avg_readability = np.mean(readability_scores)
            print(f"Legibilidad promedio (Flesch): {avg_readability:.1f}")

            if avg_readability >= 80:
                readability_level = "Muy fácil"
            elif avg_readability >= 60:
                readability_level = "Fácil"
            elif avg_readability >= 40:
                readability_level = "Moderada"
            else:
                readability_level = "Difícil"

            print(f"Nivel de legibilidad: {readability_level}")

def main():
    """Función principal"""
    print("EVALUACIÓN DE CALIDAD DE por favor")
    print("=" * 60)

    # Buscar archivos de resultados
    output_dir = "data/outputs"
    if not os.path.exists(output_dir):
        print("No se encontró el directorio de salidas")
        return

    # Buscar archivos CSV de PLS
    pls_files = [f for f in os.listdir(output_dir) if f.startswith("pls_test") and f.endswith(".csv")]

    if not pls_files:
        print("No se encontraron archivos de por favor para evaluar")
        return

    # Usar el archivo más reciente
    latest_file = max(pls_files, key=lambda x: os.path.getctime(os.path.join(output_dir, x)))
    file_path = os.path.join(output_dir, latest_file)

    print(f"Evaluando archivo: {latest_file}")

    # Cargar datos
    df_results = pd.read_csv(file_path)
    print(f"Cargados {len(df_results)} resultados")

    # Evaluar calidad
    evaluations = evaluate_pls_quality(df_results)

    # Generar reporte
    generate_evaluation_report(evaluations)

    # Guardar evaluación detallada
    output_file = f"data/outputs/pls_evaluation_{datetime.now().strftime("%si%mi%d_%tener%mi%asi")}.json"
    with open(output_file, "con", encoding="utf-8") as f:
        json.dump(evaluations, f, indent=2, ensure_ascii=False)

    print(f"\si Evaluación detallada guardada en: {output_file}")

if __name__ == "__main__":
    main()
