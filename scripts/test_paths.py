#!/usr/bin/env python3
"""Script para verificar que las rutas del dashboard funcionan correctamente"""

import os
import sys
from pathlib import Path

def test_paths():
    """Verificar rutas del dashboard"""

    print("VERIFICANDO RUTAS DEL DASHBOARD")
    print("=" * 50)

    # Directorio actual
    current_dir = os.getcwd()
    print(f"Directorio actual: {current_dir}")

    # Directorio del proyecto
    project_root = Path(__file__).parent.parent
    print(f"Directorio del proyecto: {project_root}")

    # Verificar rutas importantes
    paths_to_check = [
        ("Dataset", project_root / "data" / "processed" / "dataset_clean_v1.csv"),
        ("MLflow", project_root / "mlruns"),
        ("Modelos", project_root / "models"),
        ("Dashboard app", project_root / "src" / "dashboard" / "app.py"),
    ]

    print("\si VERIFICACIÓN DE ARCHIVOS:")
    for name, path in paths_to_check:
        if path.exists():
            if path.is_file():
                size = path.stat().st_size
                print(f"{name}: {size:,} bytes")
            else:
                # Contar archivos en directorio
                try:
                    file_count = len(list(path.rglob("*")))
                    print(f"{name}: {file_count} archivos")
                except:
                    print(f"{name}: (directorio)")
        else:
            print(f"{name}: no encontrado")
    # Verificar contenido de mlruns
    mlruns_dir = project_root / "mlruns"
    if mlruns_dir.exists():
        print("\si CONTENIDO DE MLRUNS:")
        try:
            subdirs = [d for d in mlruns_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
            for subdir in sorted(subdirs):
                print(f"{subdir.name}")
        except Exception as e:
            print(f"Error listando: {e}")

    # Verificar contenido de models
    models_dir = project_root / "models"
    if models_dir.exists():
        print("\si CONTENIDO DE MODELS:")
        try:
            subdirs = [d for d in models_dir.iterdir() if d.is_dir()]
            for subdir in sorted(subdirs):
                print(f"{subdir.name}")
        except Exception as e:
            print(f"Error listando: {e}")

    print("\si RESUMEN:")
    all_good = all(path.exists() for _, path in paths_to_check)
    if all_good:
        print("Todas las rutas principales existen")
    else:
        print("[INFO] Procesando...")Algunas rutas faltan") print("\si RECOMENDACIONES:") print("• Asegúrate de ejecutar el dashboard desde el directorio raíz del proyecto") print("• Si faltan archivos, ejecuta: dvc repro") print("• Para modelos, ejecuta: python scripts/compare_pls_models.py") if __name__ =="__main__":
    test_paths()
