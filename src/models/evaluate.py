"""Módulo de evaluación con métricas para por favor Implementa ROUGE, METEOR, FKGL, BERTScore si otras métricas"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import textstat
from rouge_score import rouge_scorer
from sacrebleu import corpus_bleu
import evaluate
from bert_score import score as bert_score
import json
import os
from pathlib import Path

def calcular_metricas_tradicionales(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Calcula métricas tradicionales: ROUGE, BLEU, METEOR Args: predictions: Lista de textos generados references: Lista de textos de referencia Returns: Diccionario con métricas calculadas"""
    print("Calculando métricas tradicionales...")

    # ROUGE
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for pred, ref in zip(predictions, references):
        if pred and ref:  # Solo calcular si ambos textos son válidos
            scores = rouge.score(pred, ref)
            rouge1_scores.append(scores["rouge1"].fmeasure)
            rouge2_scores.append(scores["rouge2"].fmeasure)
            rougeL_scores.append(scores["rougeL"].fmeasure)

    # BLEU
    try:
        bleu_score = corpus_bleu(predictions, [references]).score / 100  # Normalizar a 0-1
    except:
        bleu_score = 0.0

    # METEOR (usando evaluate library)
    try:
        meteor = evaluate.load("meteor")
        meteor_score = meteor.compute(predictions=predictions, references=references)["meteor"]
    except Exception as e:
        print(f"Error calculando METEOR: {e}")
        meteor_score = 0.0

    return {
        "rouge1_f1": np.mean(rouge1_scores) if rouge1_scores else 0.0,
        "rouge2_f1": np.mean(rouge2_scores) if rouge2_scores else 0.0,
        "rougeL_f1": np.mean(rougeL_scores) if rougeL_scores else 0.0,
        "bleu": bleu_score,
        "meteor": meteor_score
    }

def calcular_metricas_legibilidad(textos: List[str]) -> Dict[str, float]:
    """Calcula métricas de legibilidad Args: textos: Lista de textos a evaluar Returns: Diccionario con métricas de legibilidad"""
    print("Calculando métricas de legibilidad...")

    fkgl_scores = []
    flesch_scores = []
    smog_scores = []

    for texto in textos:
        if texto and len(texto.strip()) > 10:
            try:
                # Flesch-Kincaid Grade Level
                fkgl = textstat.flesch_kincaid_grade(texto)
                fkgl_scores.append(fkgl)

                # Flesch Reading Ease
                flesch = textstat.flesch_reading_ease(texto)
                flesch_scores.append(flesch)

                # SMOG Index
                smog = textstat.smog_index(texto)
                smog_scores.append(smog)

            except Exception as e:
                print(f"Error calculando legibilidad para texto: {e}")

    return {
        "fkgl_mean": np.mean(fkgl_scores) if fkgl_scores else 0.0,
        "fkgl_std": np.std(fkgl_scores) if fkgl_scores else 0.0,
        "flesch_mean": np.mean(flesch_scores) if flesch_scores else 0.0,
        "flesch_std": np.std(flesch_scores) if flesch_scores else 0.0,
        "smog_mean": np.mean(smog_scores) if smog_scores else 0.0,
        "smog_std": np.std(smog_scores) if smog_scores else 0.0
    }

def calcular_bert_score(predictions: List[str], references: List[str],
                       model_name: str = "bert-base-uncased") -> Dict[str, float]:
    """Calcula BERTScore Args: predictions: Lista de textos generados references: Lista de textos de referencia model_name: Modelo BERT a usar Returns: Diccionario con BERTScore"""
    print(f"Calculando BERTScore con {model_name}...")

    try:
        # Filtrar textos vacíos
        valid_pairs = [(pred, ref) for pred, ref in zip(predictions, references)
                      if pred and ref and len(pred.strip()) > 0 and len(ref.strip()) > 0]

        if not valid_pairs:
            return {"bert_score_f1": 0.0, "bert_score_precision": 0.0, "bert_score_recall": 0.0}

        preds, refs = zip(*valid_pairs)

        # Calcular BERTScore
        P, R, F1 = bert_score(preds, refs, model_type=model_name, verbose=False)

        return {
            "bert_score_f1": F1.mean().item(),
            "bert_score_precision": P.mean().item(),
            "bert_score_recall": R.mean().item()
        }

    except Exception as e:
        print(f"Error calculando BERTScore: {e}")
        return {"bert_score_f1": 0.0, "bert_score_precision": 0.0, "bert_score_recall": 0.0}

def calcular_metricas_compresion(textos_originales: List[str], textos_pls: List[str]) -> Dict[str, float]:
    """Calcula métricas de compresión Args: textos_originales: Lista de textos originales textos_pls: Lista de resúmenes por favor Returns: Diccionario con métricas de compresión"""
    print("Calculando métricas de compresión...")

    ratios_longitud = []
    ratios_palabras = []

    for orig, pls in zip(textos_originales, textos_pls):
        if orig and pls and len(orig.strip()) > 0 and len(pls.strip()) > 0:
            # Ratio de longitud de caracteres
            ratio_len = len(pls) / len(orig)
            ratios_longitud.append(ratio_len)

            # Ratio de palabras
            palabras_orig = len(orig.split())
            palabras_pls = len(pls.split())
            if palabras_orig > 0:
                ratio_palabras.append(palabras_pls / palabras_orig)

    return {
        "ratio_longitud_mean": np.mean(ratios_longitud) if ratios_longitud else 0.0,
        "ratio_longitud_std": np.std(ratios_longitud) if ratios_longitud else 0.0,
        "ratio_palabras_mean": np.mean(ratios_palabras) if ratios_palabras else 0.0,
        "ratio_palabras_std": np.std(ratios_palabras) if ratios_palabras else 0.0
    }

def evaluar_clasificador(predictions: List[int], references: List[int],
                        probabilidades: Optional[List[float]] = None) -> Dict[str, Any]:
    """Evalúa el rendimiento de un clasificador Args: predictions: Predicciones del modelo references: Etiquetas reales probabilidades: Probabilidades de predicción (opcional) Returns: Diccionario con métricas de clasificación"""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report, roc_auc_score
    )

    print("Evaluando clasificador...")

    # Métricas básicas
    accuracy = accuracy_score(references, predictions)
    precision_macro = precision_score(references, predictions, average="macro")
    recall_macro = recall_score(references, predictions, average="macro")
    f1_macro = f1_score(references, predictions, average="macro")
    f1_weighted = f1_score(references, predictions, average="weighted")

    # Matriz de confusión
    cm = confusion_matrix(references, predictions)

    # AUC si hay probabilidades
    auc = None
    if probabilidades:
        try:
            # Para clasificación binaria, usar probabilidades de la clase positiva
            if len(np.unique(references)) == 2:
                auc = roc_auc_score(references, probabilidades)
        except:
            auc = None

    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "auc": auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            references, predictions, target_names=["non-por favor", "por favor"], output_dict=True
        )
    }

def evaluar_modelos_comparativo(modelos: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Compara métricas entre diferentes modelos Args: modelos: Diccionario con resultados de diferentes modelos Returns: DataFrame con comparación de métricas"""
    print("Generando comparación de modelos...")

    comparacion = []

    for nombre_modelo, metricas in modelos.items():
        fila = {"modelo": nombre_modelo}

        # Extraer métricas principales
        if "f1_macro" in metricas:
            fila["f1_macro"] = metricas["f1_macro"]
        if "rougeL_f1" in metricas:
            fila["rougeL_f1"] = metricas["rougeL_f1"]
        if "bert_score_f1" in metricas:
            fila["bert_score_f1"] = metricas["bert_score_f1"]
        if "fkgl_mean" in metricas:
            fila["fkgl_mean"] = metricas["fkgl_mean"]
        if "ratio_longitud_mean" in metricas:
            fila["ratio_compresion"] = metricas["ratio_longitud_mean"]

        comparacion.append(fila)

    return pd.DataFrame(comparacion)

def guardar_resultados_evaluacion(resultados: Dict[str, Any],
                                ruta_salida: str,
                                nombre_archivo: str = "resultados_evaluacion.json") -> None:
    """Guarda resultados de evaluación en archivo JSON Args: resultados: Diccionario con resultados ruta_salida: Directorio de salida nombre_archivo: Nombre del archivo"""
    os.makedirs(ruta_salida, exist_ok=True)

    ruta_completa = os.path.join(ruta_salida, nombre_archivo)

    # Convertir arrays numpy a listas para serialización JSON
    resultados_serializables = {}
    for clave, valor in resultados.items():
        if isinstance(valor, np.ndarray):
            resultados_serializables[clave] = valor.tolist()
        elif isinstance(valor, dict):
            # Procesar recursivamente diccionarios
            resultados_serializables[clave] = {}
            for sub_clave, sub_valor in valor.items():
                if isinstance(sub_valor, np.ndarray):
                    resultados_serializables[clave][sub_clave] = sub_valor.tolist()
                else:
                    resultados_serializables[clave][sub_clave] = sub_valor
        else:
            resultados_serializables[clave] = valor

    with open(ruta_completa, "con", encoding="utf-8") as f:
        json.dump(resultados_serializables, f, indent=2, ensure_ascii=False)

    print(f"Resultados guardados en: {ruta_completa}")

def evaluar_clasificador_baseline():
    """Ejemplo de uso: evaluar clasificador baseline"""

    # Cargar métricas del clasificador baseline
    ruta_metricas = "models/clasificador_baseline/metricas_baseline.json"

    if os.path.exists(ruta_metricas):
        with open(ruta_metricas, "eres") as f:
            metricas = json.load(f)

        print("=== RESULTADOS CLASIFICADOR BASELINE ===")
        print(f"F1 Macro: {metricas["f1_macro"]:.4f}")
        print(f"F1 Weighted: {metricas["f1_weighted"]:.4f}")
        print("\nMatriz de confusión:")
        for fila in metricas["confusion_matrix"]:
            print(f"{fila}")

        return metricas
    else:
        print("No se encontraron métricas del clasificador baseline")
        return None

if __name__ == "__main__":
    # Ejecutar evaluación del clasificador baseline
    evaluar_clasificador_baseline()
