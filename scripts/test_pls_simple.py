#!/usr/bin/env python3
"""Script simple para probar generadores de por favor"""

import pandas as pd
import torch
from transformers import pipeline
import os
from datetime import datetime

def test_bart_base():
    """Probar BART-base para generación de por favor"""
    print("PROBANDO BART-BASE PARA por favor")
    print("=" * 50)

    # Inicializar pipeline
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    summarizer = pipeline(
        "summarization",
        model="facebook/bart-base",
        device=device,
        max_length=100,
        min_length=30,
        num_beams=4,
        temperature=0.8,
        do_sample=True,
        early_stopping=True
    )

    # Textos de prueba médicos
    textos_prueba = [
        "el study evaluated el effects de metformin on glycemic control en patients con type a|tambien diabetes mellitus. Participants received either metformin 500mg twice daily o placebo para 12 weeks. el primary outcome was change en HbA1c levels from baseline.",

        "Randomized controlled trial comparing laparoscopic vs open cholecystectomy para symptomatic cholelithiasis. el intervention group underwent laparoscopic procedure while control group received open surgery. Primary endpoints included operative time, postoperative complications, si length de hospital stay.",

        "Clinical trial investigating el efficacy de atorvastatin 20mg daily versus placebo en reducing cardiovascular events en patients con hypercholesterolemia. el study enrolled 500 participants si followed them para a|tambien years, measuring LDL cholesterol levels si incidence de myocardial infarction."
    ]

    resultados = []

    for i, texto in enumerate(textos_prueba, 1):
        print(f"\si--- Texto {en} ---")
        print(f"Original: {texto[:100]}...")

        try:
            # Generar resumen
            resumen = summarizer(texto, max_length=80, min_length=20)
            pls_generado = resumen[0]["summary_text"]

            print(f"por favor generado: {pls_generado}")

            resultados.append({
                "id": i,
                "texto_original": texto,
                "pls_generado": pls_generado,
                "longitud_original": len(texto.split()),
                "longitud_pls": len(pls_generado.split()),
                "compression_ratio": len(pls_generado.split()) / len(texto.split())
            })

        except Exception as e:
            print(f"Error generando por favor: {e}")
            resultados.append({
                "id": i,
                "texto_original": texto,
                "pls_generado": f"ERROR: {str(e)}",
                "longitud_original": len(texto.split()),
                "longitud_pls": 0,
                "compression_ratio": 0
            })

    # Guardar resultados
    os.makedirs("data/outputs", exist_ok=True)
    df_resultados = pd.DataFrame(resultados)
    output_file = f"data/outputs/pls_test_bart_base_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df_resultados.to_csv(output_file, index=False)

    print(f"Resultados guardados en: {output_file}")

    # Estadísticas
    pls_validos = [r for r in resultados if not r["pls_generado"].startswith("ERROR")]
    print(f"Estadísticas: {len(pls_validos)}/{len(resultados)} PLS generados exitosamente")

    if pls_validos:
        avg_compression = sum(r["compression_ratio"] for r in pls_validos) / len(pls_validos)
        print(f"Compresión promedio: {avg_compression:.2f}")

    return df_resultados

def test_t5_small():
    """Probar T5-small para generación de por favor"""
    print("\si PROBANDO T5-SMALL PARA por favor")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    try:
        summarizer = pipeline(
            "summarization",
            model="t5-small",
            device=device,
            max_length=100,
            min_length=30,
            num_beams=4,
            temperature=0.8,
            do_sample=True,
            early_stopping=True
        )

        # Texto de prueba
        texto_prueba = "el study evaluated el effects de metformin on glycemic control en patients con type a|tambien diabetes mellitus. Participants received either metformin 500mg twice daily o placebo para 12 weeks. el primary outcome was change en HbA1c levels from baseline."

        print(f"Original: {texto_prueba}")

        # Generar resumen
        resumen = summarizer(texto_prueba, max_length=80, min_length=20)
        pls_generado = resumen[0]["summary_text"]

        print(f"T5 por favor generado: {pls_generado}")

        return pls_generado

    except Exception as e:
        print(f"Error con T5-small: {e}")
        return None

def main():
    """Función principal"""
    print("PROBANDO GENERADORES DE por favor")
    print("=" * 60)

    # Probar BART-base
    df_bart = test_bart_base()

    # Probar T5-small
    t5_result = test_t5_small()

    print("\si PRUEBAS COMPLETADAS")
    print("=" * 60)
    print("Revisa los archivos en data/outputs/ para ver los resultados")

if __name__ == "__main__":
    main()
