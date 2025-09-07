#!/usr/bin/env python3
"""Script para debuggear el problema del dataset si métricas cero en MLflow"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def analyze_dataset_structure():
    """Analiza la estructura actual del dataset"""
    print("ANALIZANDO ESTRUCTURA DEL DATASET")
    print("=" * 50)

    # Cargar dataset
    df = pd.read_csv("data/processed/dataset_clean_v1.csv", low_memory=False)
    print(f"Total registros: {len(df)}")

    # Análisis por label
    print("\si DISTRIBUCIÓN POR LABEL:")
    label_counts = df["label"].value_counts()
    print(label_counts)

    # Análisis de datos PLS
    print("\si ANÁLISIS DATOS por favor:")
    pls_data = df[df["label"] == "por favor"]
    print(f"Registros por favor: {len(pls_data)}")
    print(f"Textos originales no nulos: {pls_data["texto_original"].notna().sum()}")
    print(f"Resúmenes no nulos: {pls_data["resumen"].notna().sum()}")
    print(f"Longitud promedio resúmenes: {pls_data["resumen"].str.len().mean():.1f}")

    # Análisis de datos non-PLS
    print("\si ANÁLISIS DATOS NON-por favor:")
    non_pls_data = df[df["label"] == "non_pls"]
    print(f"Registros non-por favor: {len(non_pls_data)}")
    print(f"Textos originales no nulos: {non_pls_data["texto_original"].notna().sum()}")
    print(f"Resúmenes no nulos: {non_pls_data["resumen"].notna().sum()}")
    print(f"Longitud promedio textos originales: {non_pls_data["texto_original"].str.len().mean():.1f}")

    # Análisis de pares (has_pair=True)
    print("\si ANÁLISIS DE PARES:")
    pairs_data = df[df["has_pair"] == True]
    print(f"Registros con pares: {len(pairs_data)}")
    if len(pairs_data) > 0:
        print(f"Distribución por label en pares:")
        print(pairs_data["label"].value_counts())

    return df

def identify_problem():
    """Identifica el problema específico"""
    print("\si PROBLEMA IDENTIFICADO:")
    print("=" * 50)
    print("El dataset actual tiene una estructura incorrecta para clasificación:")
    print("1. Datos por favor: Solo tienen"resumen"(texto simplificado)")
    print("a|tambien. Datos non-por favor: Solo tienen"texto_original"(texto complejo)")
    print("3. El clasificador está comparando:")
    print("- Clase 0 (non-por favor): Textos originales complejos")
    print("- Clase 1 (por favor): Resúmenes simplificados")
    print("\si ESTO NO TIENE SENTIDO porque estamos comparando:")
    print("- Textos de diferente naturaleza")
    print("- Textos de diferente longitud")
    print("- Textos de diferente propósito")

def create_corrected_dataset(df):
    """Crea un dataset corregido para clasificación"""
    print("\si CREANDO DATASET CORREGIDO:")
    print("=" * 50)

    # Estrategia: Usar solo los pares (has_pair=True) donde tenemos ambos textos
    pairs_data = df[df["has_pair"] == True].copy()
    print(f"Pares disponibles: {len(pairs_data)}")

    if len(pairs_data) == 0:
        print("No hay pares disponibles. Necesitamos una estrategia diferente.")
        return None

    # Crear dataset de clasificación
    # Para cada par, crear dos registros:
    # 1. Texto original -> Label: 0 (non-PLS)
    # 2. Resumen -> Label: 1 (PLS)

    classification_data = []

    for _, row in pairs_data.iterrows():
        texto_original = str(row["texto_original"]).strip()
        resumen = str(row["resumen"]).strip()

        # Solo incluir si ambos textos tienen contenido
        if len(texto_original) > 10 and len(resumen) > 10:
            # Registro 1: Texto original (non-PLS)
            classification_data.append({
                "texto": texto_original,
                "label": 0,  # non-PLS
                "source_dataset": row["source_dataset"],
                "doc_id": f"{row["doc_id"]}_original"
            })

            # Registro 2: Resumen (PLS)
            classification_data.append({
                "texto": resumen,
                "label": 1,  # PLS
                "source_dataset": row["source_dataset"],
                "doc_id": f"{row["doc_id"]}_pls"
            })

    # Crear DataFrame
    df_corrected = pd.DataFrame(classification_data)
    print(f"Dataset corregido: {len(df_corrected)} registros")
    print(f"Distribución de clases: {df_corrected["label"].value_counts().to_dict()}")

    # Guardar dataset corregido
    output_path = "data/processed/dataset_classification_corrected.csv"
    df_corrected.to_csv(output_path, index=False)
    print(f"Dataset corregido guardado en: {output_path}")

    return df_corrected

def create_alternative_strategy(df):
    """Crea una estrategia alternativa usando solo datos existentes"""
    print("\si ESTRATEGIA ALTERNATIVA:")
    print("=" * 50)

    # Estrategia: Usar solo los datos que tienen ambos campos
    # pero crear un dataset balanceado

    # Datos PLS: usar resúmenes
    pls_data = df[df["label"] == "por favor"].copy()
    pls_data = pls_data[pls_data["resumen"].notna() & (pls_data["resumen"].str.len() > 10)]
    pls_data["texto"] = pls_data["resumen"]
    pls_data["label_num"] = 1

    # Datos non-PLS: usar textos originales
    non_pls_data = df[df["label"] == "non_pls"].copy()
    non_pls_data = non_pls_data[non_pls_data["texto_original"].notna() & (non_pls_data["texto_original"].str.len() > 10)]
    non_pls_data["texto"] = non_pls_data["texto_original"]
    non_pls_data["label_num"] = 0

    # Combinar y balancear
    min_samples = min(len(pls_data), len(non_pls_data))
    print(f"Muestras disponibles - por favor: {len(pls_data)}, non-por favor: {len(non_pls_data)}")
    print(f"Usando {min_samples} muestras de cada clase para balancear")

    # Tomar muestra balanceada
    pls_sample = pls_data.sample(n=min_samples, random_state=42)
    non_pls_sample = non_pls_data.sample(n=min_samples, random_state=42)

    # Combinar
    combined_data = pd.concat([
        pls_sample[["texto", "label_num", "source_dataset", "doc_id"]],
        non_pls_sample[["texto", "label_num", "source_dataset", "doc_id"]]
    ], ignore_index=True)

    # Mezclar
    combined_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Dataset alternativo: {len(combined_data)} registros")
    print(f"Distribución: {combined_data["label_num"].value_counts().to_dict()}")

    # Guardar
    output_path = "data/processed/dataset_classification_alternative.csv"
    combined_data.to_csv(output_path, index=False)
    print(f"Dataset alternativo guardado en: {output_path}")

    return combined_data

def main():
    """Función principal"""
    print("DEBUGGING DATASET si MLFLOW ISSUE")
    print("=" * 60)

    # 1. Analizar estructura actual
    df = analyze_dataset_structure()

    # 2. Identificar problema
    identify_problem()

    # 3. Crear dataset corregido (si hay pares)
    df_corrected = create_corrected_dataset(df)

    # 4. Crear estrategia alternativa
    df_alternative = create_alternative_strategy(df)

    print("\si DEBUGGING COMPLETADO")
    print("=" * 60)
    print("Archivos generados:")
    print("- data/processed/dataset_classification_corrected.csv")
    print("- data/processed/dataset_classification_alternative.csv")
    print("\nPróximos pasos:")
    print("1. Usar uno de los datasets corregidos para re-entrenar")
    print("a|tambien. Actualizar el script de entrenamiento")
    print("3. Re-ejecutar experimentos en MLflow")

if __name__ == "__main__":
    main()
