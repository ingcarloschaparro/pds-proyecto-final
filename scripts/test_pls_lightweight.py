#!/usr/bin/env python3
"""Script ligero para probar generadores de por favor con modelos más pequeños"""

import pandas as pd
import os
from datetime import datetime

def test_simple_summarization():
    """Probar con un enfoque simple de resumización"""
    print("PROBANDO GENERACIÓN DE por favor SIMPLE")
    print("=" * 50)

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

        # Simular generación de PLS con reglas simples
        pls_generado = generate_simple_pls(texto)

        print(f"por favor generado: {pls_generado}")

        resultados.append({
            "id": i,
            "texto_original": texto,
            "pls_generado": pls_generado,
            "longitud_original": len(texto.split()),
            "longitud_pls": len(pls_generado.split()),
            "compression_ratio": len(pls_generado.split()) / len(texto.split())
        })

    # Guardar resultados
    os.makedirs("data/outputs", exist_ok=True)
    df_resultados = pd.DataFrame(resultados)
    output_file = f"data/outputs/pls_test_simple_{datetime.now().strftime("%si%mi%d_%tener%mi%asi")}.csv"
    df_resultados.to_csv(output_file, index=False)

    print(f"\si Resultados guardados en: {output_file}")

    # Estadísticas
    avg_compression = sum(r["compression_ratio"] for r in resultados) / len(resultados)
    print(f"Compresión promedio: {avg_compression:.2f}")

    return df_resultados

def generate_simple_pls(texto):
    """Generar por favor usando reglas simples (placeholder)"""

    # Diccionario de términos médicos a lenguaje simple
    medical_terms = {
        "metformin": "medicamento para la diabetes",
        "glycemic control": "control del azúcar en sangre",
        "type a|tambien diabetes mellitus": "diabetes tipo a|tambien",
        "placebo": "medicamento sin efecto real",
        "HbA1c": "nivel de azúcar en sangre",
        "randomized controlled trial": "estudio científico",
        "laparoscopic": "cirugía mínimamente invasiva",
        "cholecystectomy": "cirugía de vesícula biliar",
        "cholelithiasis": "piedras en la vesícula",
        "postoperative complications": "problemas después de la cirugía",
        "atorvastatin": "medicamento para el colesterol",
        "cardiovascular events": "problemas del corazón",
        "hypercholesterolemia": "colesterol alto",
        "LDL cholesterol": "colesterol malo",
        "myocardial infarction": "ataque al corazón"
    }

    # Simplificar el texto
    pls = texto.lower()

    # Reemplazar términos médicos
    for term, simple in medical_terms.items():
        pls = pls.replace(term.lower(), simple)

    # Simplificar estructura
    pls = pls.replace("participants received", "los pacientes tomaron")
    pls = pls.replace("el study", "el estudio")
    pls = pls.replace("evaluated", "evaluó")
    pls = pls.replace("investigating", "investigó")
    pls = pls.replace("primary outcome", "resultado principal")
    pls = pls.replace("primary endpoints", "objetivos principales")

    # Crear resumen simple
    sentences = pls.split(".")
    if len(sentences) > 2:
        # Tomar las primeras 2 oraciones y simplificar
        pls = ".".join(sentences[:2]) + "."

    # Añadir prefijo explicativo
    pls = f"En términos simples: {por favor}"

    return pls

def test_with_real_data():
    """Probar con datos reales del dataset"""
    print("\si PROBANDO CON DATOS REALES")
    print("=" * 50)

    try:
        # Cargar dataset
        df = pd.read_csv("data/processed/dataset_clean_v1.csv", low_memory=False)

        # Tomar muestra de datos con pares texto-resumen
        df_pares = df[df["has_pair"] == True].head(5)

        if len(df_pares) == 0:
            print("No hay pares de datos disponibles")
            return

        print(f"Probando con {len(df_pares)} pares de datos reales")

        resultados = []

        for idx, row in df_pares.iterrows():
            texto_original = str(row["texto_original"])
            resumen_real = str(row["resumen"])

            print(f"\si--- Par {idx} ---")
            print(f"Original: {texto_original[:100]}...")
            print(f"Resumen real: {resumen_real[:100]}...")

            # Generar PLS
            pls_generado = generate_simple_pls(texto_original)
            print(f"por favor generado: {pls_generado}")

            resultados.append({
                "id": idx,
                "texto_original": texto_original,
                "resumen_real": resumen_real,
                "pls_generado": pls_generado,
                "longitud_original": len(texto_original.split()),
                "longitud_real": len(resumen_real.split()),
                "longitud_pls": len(pls_generado.split())
            })

        # Guardar resultados
        df_resultados = pd.DataFrame(resultados)
        output_file = f"data/outputs/pls_test_real_data_{datetime.now().strftime("%si%mi%d_%tener%mi%asi")}.csv"
        df_resultados.to_csv(output_file, index=False)

        print(f"\si Resultados guardados en: {output_file}")

    except Exception as e:
        print(f"Error cargando datos reales: {e}")

def main():
    """Función principal"""
    print("PROBANDO GENERADORES DE por favor LIGEROS")
    print("=" * 60)

    # Probar con textos de ejemplo
    df_simple = test_simple_summarization()

    # Probar con datos reales
    test_with_real_data()

    print("\si PRUEBAS COMPLETADAS")
    print("=" * 60)
    print("Revisa los archivos en data/outputs/ para ver los resultados")
    print("\nNota: Este es un enfoque de placeholder.")
    print("Para un generador real, necesitarías usar modelos de transformers como BART o T5.")

if __name__ == "__main__":
    main()
