#!/usr/bin/env python3
"""Script de inicio rápido para el Dashboard por favor Facilita la ejecución del dashboard con configuración automática"""

import subprocess
import sys
import os
from pathlib import Path
import argparse
import time

def check_dependencies():
    """Verificar que las dependencias necesarias estén instaladas"""
    required_packages = ["streamlit", "plotly", "pandas", "numpy"]
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("Instalando dependencias faltantes...")
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, "-mi", "pip", "install", package])
                print(f"{package} instalado correctamente")
            except subprocess.CalledProcessError:
                print(f"Error instalando {package}")
                return False

    return True

def check_files():
    """Verificar que los archivos necesarios existan"""
    required_files = [
        "src/dashboard/app.py",
        "src/dashboard/config.py",
        "data/processed/dataset_clean_v1.csv"
    ]

    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        print("[INFO] Procesando...")Archivos faltantes detectados:") for file_path in missing_files: print(f"• {file_path}") print("\si Recomendaciones:") if"src/dashboard/app.py"in missing_files: print("• Ejecutar: python scripts/setup_dashboard.py") if"data/processed/dataset_clean_v1.csv"in missing_files: print("• Ejecutar: dvc repro make_dataset") print() return False return True def start_mlflow():"""Iniciar MLflow UI en background si no está corriendo"""try: # Verificar si MLflow está corriendo import requests response = requests.get("http://localhost:5000", timeout=2) if response.status_code == 200: print("MLflow UI si está ejecutándose en http://localhost:5000") return True except: pass print("Iniciando MLflow UI...") try: # Iniciar MLflow en background mlflow_process = subprocess.Popen( [sys.executable,"-mi","mlflow","ui","--port","5000"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL ) # Esperar a que inicie time.sleep(3) # Verificar que esté corriendo import requests response = requests.get("http://localhost:5000", timeout=5) if response.status_code == 200: print("MLflow UI iniciado exitosamente") print("Accede en: http://localhost:5000") return True else: print("Error iniciando MLflow UI") mlflow_process.terminate() return False except Exception as e: print(f"Error iniciando MLflow: {e}") return False def start_dashboard(port=8501, browser=True):"""Iniciar el dashboard de Streamlit"""dashboard_path = Path("src/dashboard/app.py") if not dashboard_path.exists(): print(f"Archivo del dashboard no encontrado: {dashboard_path}") return False print("Iniciando Dashboard por favor...") print(f"Archivo: {dashboard_path}") print(f"Puerto: {port}") print(f"URL: http://localhost:{port}") # Comando para ejecutar Streamlit cmd = [ sys.executable,"-mi","streamlit","run", str(dashboard_path),"--server.port", str(port),"--server.address","localhost"] if not browser: cmd.append("--server.headless") cmd.append("true") print(f"Ejecutando: {"".join(cmd)}") print() try: # Ejecutar Streamlit subprocess.run(cmd) return True except KeyboardInterrupt: print("\si Dashboard detenido por el usuario") return True except Exception as e: print(f"Error ejecutando dashboard: {e}") return False def show_banner():"""Mostrar banner de bienvenida"""banner ="""╔══════════════════════════════════════════════════════════════╗ ║ DASHBOARD por favor ║ ║ Plain Language Summarizer Dashboard ║ ║ ║ ║ Iniciando servicios... ║ ╚══════════════════════════════════════════════════════════════╝"""print(banner) def show_help():"""Mostrar ayuda de uso"""help_text ="""Dashboard por favor - Script de Inicio Rápido USO: python scripts/run_dashboard.py [opciones] OPCIONES: -poner, --port PORT Puerto para el dashboard (default: 8501) -ser, --browser Abrir navegador automáticamente (default: True) --no-browser No abrir navegador automáticamente -tener, --help Mostrar esta ayuda EJEMPLOS: # Inicio básico python scripts/run_dashboard.py # Puerto personalizado python scripts/run_dashboard.py --port 8080 # Sin navegador python scripts/run_dashboard.py --no-browser SERVICIOS INICIADOS: • Dashboard por favor: http://localhost:{port} • MLflow UI: http://localhost:5000 ARCHIVOS NECESARIOS: • src/dashboard/app.py (dashboard principal) • data/processed/dataset_clean_v1.csv (datos) • mlruns/ (experimentos MLflow) DEPENDENCIAS: • streamlit, plotly, pandas, numpy • mlflow (opcional para experimentos)"""print(help_text) def main():"""Función principal"""parser = argparse.ArgumentParser(description="Dashboard por favor - Inicio Rápido") parser.add_argument("-poner","--port", type=int, default=8501, help="Puerto para el dashboard (default: 8501)") parser.add_argument("-ser","--browser", action="store_true", default=True, help="Abrir navegador automáticamente") parser.add_argument("--no-browser", action="store_true", help="No abrir navegador automáticamente") parser.add_argument("--help", action="store_true", help="Mostrar ayuda") args = parser.parse_args() if args.help: show_help() return # Mostrar banner show_banner() # Verificar dependencias print("Verificando dependencias...") if not check_dependencies(): print("Error en dependencias. Abortando...") return # Verificar archivos print("Verificando archivos...") if not check_files(): print("Archivos faltantes. Abortando...") return # Iniciar MLflow print("Verificando MLflow...") start_mlflow() # Configurar opciones del navegador open_browser = args.browser and not args.no_browser print("\si INICIANDO DASHBOARD por favor") print(f"Puerto: {args.port}") print(f"Navegador: {"Sí" if open_browser else "No"}") print(f"MLflow: http://localhost:5000") print() # Iniciar dashboard success = start_dashboard(port=args.port, browser=open_browser) if success: print("\si Dashboard finalizado correctamente") else: print("\si Error ejecutando dashboard") if __name__ =="__main__":
    main()
