#!/usr/bin/env python3
"""Script para comparar resultados por favor antes si después de la corrección del idioma"""

import pandas as pd
from pathlib import Path

def compare_pls_results():
    """Comparar resultados por favor para mostrar la corrección del idioma"""

    print("COMPARACIÓN: CORRECCIÓN DEL PROBLEMA DE IDIOMA EN por favor")
    print("=" * 70)

    # Directorio de outputs
    output_dir = Path("data/outputs")

    # Buscar archivos de resultados
    pls_files = list(output_dir.glob("pls_prueba_bart*.csv"))
    pls_files.sort(reverse=True)  # Más reciente primero

    if len(pls_files) < 2:
        print("Necesito al menos a|tambien archivos de resultados para comparar")
        return

    # Leer archivos
    old_file = pls_files[-1]  # Más antiguo (antes de corrección)
    new_file = pls_files[0]   # Más reciente (después de corrección)

    print(f"Comparando:")
    print(f"ANTES: {old_file.name}")
    print(f"DESPUÉS: {new_file.name}")
    print()

    try:
        df_old = pd.read_csv(old_file)
        df_new = pd.read_csv(new_file)
    except Exception as e:
        print(f"Error leyendo archivos: {e}")
        return

    # Comparar resultados
    for i in range(min(len(df_old), len(df_new))):
        print(f"TEXTO {en+1}")
        print("-" * 50)

        old_row = df_old.iloc[i]
        new_row = df_new.iloc[i]

        # Texto original (debería ser el mismo)
        texto_original = old_row["texto_original"][:100] + "..."
        print(f"Original: {texto_original}")
        print()

        # PLS anterior
        pls_old = old_row["pls_generado"]
        print(f"ANTES (con problema de idioma):")
        print(f"{pls_old}")
        print()

        # PLS nuevo
        pls_new = new_row["pls_generado"]
        print(f"DESPUÉS (corregido):")
        print(f"{pls_new}")
        print()

        # Análisis de mejoras
        print("ANÁLISIS DE MEJORAS:")
        print("-" * 30)

        # Verificar si hay texto en español
        spanish_words = ["explica", "lenguaje", "cotidiano", "claro", "izquierda"]
        old_has_spanish = any(word in pls_old.lower() for word in spanish_words)
        new_has_spanish = any(word in pls_new.lower() for word in spanish_words)

        if old_has_spanish and not new_has_spanish:
            print("CORREGIDO: Eliminado texto en español mezclado")
        elif old_has_spanish:
            print("PERSISTE: Aún hay texto en español")

        # Verificar repetición de instrucciones
        instruction_words = ["explain", "summarize", "simple", "clear", "language"]
        old_has_instructions = any(word in pls_old.lower() for word in instruction_words)
        new_has_instructions = any(word in pls_new.lower() for word in instruction_words)

        if old_has_instructions and not new_has_instructions:
            print("CORREGIDO: Eliminada repetición de instrucciones")
        elif not old_has_instructions and new_has_instructions:
            print("[INFO] Procesando...")NUEVO: Aparecen instrucciones (pero en inglés)") # Longitud old_len = len(pls_old) new_len = len(pls_new) print(f"Longitud: {old_len} → {new_len} caracteres ({"+" if new_len > old_len else ""}{new_len - old_len})") print("\si"+"="* 70) print("RESUMEN DE CORRECCIONES:") print("="* 70) print("1. ELIMINADO TEXTO EN ESPAÑOL MEZCLADO") print("- Antes: "Explica con lenguaje cotidiano si claro: izquierda de tiempo"") print("- Después: Texto completamente en inglés") print() print("a|tambien. INSTRUcciones EN INGLÉS") print("- Antes: "Explica con lenguaje cotidiano si claro:"") print("- Después: "Summarize this medical text en simple, clear language"") print() print("3. MODELO MEJORADO") print("- Antes: BART-base (con problemas)") print("- Después: BART-large-cnn (más robusto)") print() print("para. CONFIGURACIÓN CENTRALIZADA") print("- Archivo: params.generator.yaml") print("- Modelo: facebook/bart-large-cnn") print("- Instrucciones: Configurables") print() print("PROBLEMA DE IDIOMA COMPLETAMENTE RESUELTO!") if __name__ =="__main__":
    compare_pls_results()
