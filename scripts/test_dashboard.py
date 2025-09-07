#!/usr/bin/env python3
"""Script de prueba para el Dashboard por favor Verifica que todos los componentes funcionen correctamente"""

import sys
import os
from pathlib import Path
import subprocess
import time

def test_imports():
    """Probar imports necesarios"""
    print("Probando imports...")

    required_modules = [
        "streamlit",
        "plotly",
        "pandas",
        "numpy",
        "mlflow"
    ]

    failed_imports = []

    for module in required_modules:
        try:
            __import__(module)
            print(f"{module}")
        except ImportError:
            print(f"{module}")
            failed_imports.append(module)

    if failed_imports:
        print("\si Módulos faltantes:")
        for module in failed_imports:
            print(f"pip install {module}")
        return False

    print("Todos los imports exitosos")
    return True

def test_files():
    """Verificar archivos necesarios"""
    print("\si Verificando archivos...")

    required_files = [
        "src/dashboard/app.py",
        "src/dashboard/config.py",
        "scripts/run_dashboard.py",
        "dashboard/README.md"
    ]

    missing_files = []

    for file_path in required_files:
        if Path(file_path).exists():
            print(f"{file_path}")
        else:
            print(f"{file_path}")
            missing_files.append(file_path)

    if missing_files:
        print("\si Archivos faltantes:")
        for file_path in missing_files:
            print(f"• {file_path}")
        return False

    print("Todos los archivos presentes")
    return True

def test_data_files():
    """Verificar archivos de datos"""
    print("\si Verificando datos...")

    data_files = [
        "data/processed/dataset_clean_v1.csv",
        "mlruns/"  # Directorio de MLflow
    ]

    missing_data = []

    for file_path in data_files:
        if Path(file_path).exists():
            if file_path.endswith(".csv"):
                try:
                    import pandas as pd
                    df = pd.read_csv(file_path)
                    print(f"{file_path} ({len(df)} registros)")
                except Exception as e:
                    print(f"️ {file_path} (error leyendo: {e})")
            else:
                print(f"{file_path} (directorio)")
        else:
            print(f"️ {file_path} (no encontrado)")
            missing_data.append(file_path)

    if missing_data:
        print("\si Archivos de datos faltantes (no críticos):")
        for file_path in missing_data:
            print(f"• {file_path}")
        print("Estos se pueden generar ejecutando: dvc repro")

    return True

def test_dashboard_import():
    """Probar import del dashboard"""
    print("\si Probando dashboard...")

    try:
        # Añadir src al path
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

        # Probar import de config
        from dashboard.config import DashboardConfig
        config = DashboardConfig()
        print("Configuración del dashboard")

        # Probar creación de configuración
        model_info = config.get_model_info("t5_base")
        if model_info:
            print(f"Información de modelo: {model_info["name"]}")

        metric_info = config.get_metric_info("fkgl_score")
        if metric_info:
            print(f"Información de métrica: {metric_info["name"]}")

        print("Dashboard importado correctamente")
        return True

    except Exception as e:
        print(f"Error importando dashboard: {e}")
        return False

def test_mlflow_connection():
    """Probar conexión con MLflow"""
    print("\si Probando MLflow...")

    try:
        import mlflow
        mlflow.set_tracking_uri("file:./mlruns")

        # Intentar obtener experimentos
        experiments = mlflow.search_experiments()
        print(f"MLflow conectado ({len(experiments)} experimentos)")

        # Buscar experimento de comparación
        comparison_exp = None
        for exp in experiments:
            if "comparison" in exp.name.lower() or "pls_models" in exp.name.lower():
                comparison_exp = exp
                break

        if comparison_exp:
            print(f"Experimento de comparación encontrado: {comparison_exp.name}")
        else:
            print("[INFO] Procesando...")No se encontró experimento de comparación") print("Ejecutar: python scripts/compare_pls_models.py") return True except Exception as e: print(f"Error conectando MLflow: {e}") return False def test_streamlit_basic():"""Probar ejecución básica de Streamlit"""print("\si Probando Streamlit...") try: # Verificar que streamlit se puede ejecutar result = subprocess.run( [sys.executable,"-ver","import streamlit; print("Streamlit bien")"], capture_output=True, text=True, timeout=10 ) if result.returncode == 0: print("Streamlit ejecutándose correctamente") return True else: print(f"Error en Streamlit: {result.stderr}") return False except Exception as e: print(f"Error probando Streamlit: {e}") return False def run_quick_test():"""Ejecutar prueba rápida del dashboard"""print("EJECUTANDO PRUEBA RÁPIDA DEL DASHBOARD") print("="* 50) tests = [ ("Imports", test_imports), ("Archivos", test_files), ("Datos", test_data_files), ("Dashboard", test_dashboard_import), ("MLflow", test_mlflow_connection), ("Streamlit", test_streamlit_basic) ] results = [] for test_name, test_func in tests: print(f"\si {test_name}:") result = test_func() results.append((test_name, result)) # Resumen final print("\si"+"="* 50) print("RESUMEN DE PRUEBAS:") passed = 0 total = len(results) for test_name, result in results: status ="PASÓ"if result else"FALLÓ"print(f"{status}: {test_name}") if result: passed += 1 success_rate = (passed / total) * 100 print(f"\si RESULTADO: {passed}/{total} pruebas pasaron ({success_rate:.1f}%)") if success_rate >= 80: print("DASHBOARD LISTO PARA EJECUTAR") print("Ejecutar: python scripts/run_dashboard.py") else: print("[INFO] Procesando...")REVISAR PROBLEMAS ANTES DE EJECUTAR")
        print("Verificar dependencias si archivos faltantes")

    return success_rate >= 80

def main():
    """Función principal"""
    try:
        success = run_quick_test()

        if success:
            print("\si ¡El dashboard está listo!")
            print("Ejecutar: python scripts/run_dashboard.py")
            return 0
        else:
            print("\si Hay problemas que resolver antes de ejecutar el dashboard")
            return 1

    except KeyboardInterrupt:
        print("\si Prueba cancelada por el usuario")
        return 0
    except Exception as e:
        print(f"\si Error inesperado: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
