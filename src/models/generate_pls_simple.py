"""
Generador simplificado de PLS con BART-base
"""
import pandas as pd
import numpy as np
import torch
from transformers import pipeline
import os
import json
from typing import List, Dict, Any

class PLSGeneratorSimple:
    """Generador simplificado de PLS usando BART"""

    def __init__(self):
        self.pipeline = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def cargar_modelo(self):
        """Cargar modelo BART-base"""
        print("Cargando modelo BART-base...")
        try:
            self.pipeline = pipeline(
                "summarization",
                model="facebook/bart-base",
                device=self.device,
                max_length=150,
                min_length=50,
                num_beams=4,
                temperature=0.8,
                do_sample=True,
                early_stopping=True
            )
            print("Modelo cargado exitosamente")
            return True
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            return False

    def generar_pls(self, texto: str, instrucciones: str = None) -> str:
        """Generar PLS para un texto individual"""
        if not texto or pd.isna(texto):
            return ""

        # Instrucciones por defecto
        if instrucciones is None:
            instrucciones = "Explica con lenguaje cotidiano y claro: "

        # Preparar input
        input_text = f"{instrucciones}\n\n{str(texto)[:1000]}"  # Limitar longitud

        try:
            resultados = self.pipeline(input_text)
            return resultados[0]['summary_text'] if resultados else ""
        except Exception as e:
            print(f"Error generando PLS: {e}")
            return ""

def main():
    """Función principal para probar el generador"""
    print("PROBANDO GENERADOR PLS CON BART-BASE")
    print("=" * 50)

    # Inicializar generador
    generador = PLSGeneratorSimple()

    if not generador.cargar_modelo():
        return

    # Textos de prueba
    textos_prueba = [
        "The study evaluated the effects of metformin on glycemic control in patients with type 2 diabetes mellitus. Participants received either metformin 500mg twice daily or placebo for 12 weeks. The primary outcome was change in HbA1c levels from baseline.",

        "Randomized controlled trial comparing laparoscopic vs open cholecystectomy for symptomatic cholelithiasis. The intervention group underwent laparoscopic procedure while control group received open surgery. Primary endpoints included operative time, postoperative complications, and length of hospital stay.",

        "Clinical trial investigating the efficacy of atorvastatin 20mg daily versus placebo in reducing cardiovascular events in patients with hypercholesterolemia. The study enrolled 500 participants and followed them for 2 years, measuring LDL cholesterol levels and incidence of myocardial infarction."
    ]

    # Generar PLS
    print("\n GENERANDO PLS...")
    resultados = []

    for i, texto in enumerate(textos_prueba, 1):
        print(f"\n--- Texto {i} ---")
        print(f"Original: {texto[:100]}...")

        pls = generador.generar_pls(texto)
        print(f"PLS generado: {pls}")

        resultados.append({
            'id': i,
            'texto_original': texto,
            'pls_generado': pls,
            'longitud_original': len(texto),
            'longitud_pls': len(pls)
        })

    # Guardar resultados
    os.makedirs("data/outputs", exist_ok=True)
    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv("data/outputs/pls_prueba_bart.csv", index=False)

    print("\nPrueba completada!")
    print(f"Resultados guardados en: data/outputs/pls_prueba_bart.csv")
    # Estadísticas
    pls_validos = [r for r in resultados if r['pls_generado']]
    print(f"Estadísticas: {len(pls_validos)}/{len(resultados)} PLS generados exitosamente")

if __name__ == "__main__":
    main()
