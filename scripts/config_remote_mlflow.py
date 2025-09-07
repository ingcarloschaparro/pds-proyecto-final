#!/usr/bin/env python3
"""Script para configurar MLflow remoto en EC2 Uso: python config_remote_mlflow.py --ec2-ip IP_EC2"""

import argparse
import mlflow
import requests
import time
from datetime import datetime

def test_connection(ec2_ip, port=5000, timeout=30):
    """Probar conexión al servidor MLflow en EC2"""
    print(f"Probando conexión a {ec2_ip}:{port}...")

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http://{ec2_ip}:{port}", timeout=5)
            if response.status_code == 200:
                print("Conexión exitosa")
                return True
        except requests.exceptions.RequestException:
            pass

        print(".", end="", flush=True)
        time.sleep(2)

    print(f"\si No se pudo conectar después de {timeout} segundos")
    return False

def configure_mlflow_tracking(ec2_ip, port=5000):
    """Configurar MLflow para usar servidor remoto"""
    tracking_uri = f"http://{ec2_ip}:{port}"

    print(f"Configurando MLflow tracking URI: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)

    return tracking_uri

def list_experiments():
    """Listar experimentos disponibles"""
    try:
        experiments = mlflow.search_experiments()
        print(f"\si Experimentos encontrados: {len(experiments)}")

        for exp in experiments:
            print(f"- {exp.name} (ID: {exp.experiment_id})")
            print(f"Creado: {datetime.fromtimestamp(exp.creation_time/1000)}")
            print(f"Última modificación: {datetime.fromtimestamp(exp.last_update_time/1000)}")
            print()

        return experiments
    except Exception as e:
        print(f"Error listando experimentos: {e}")
        return []

def test_experiment_creation():
    """Probar creación de un experimento de prueba"""
    try:
        print("Creando experimento de prueba...")

        with mlflow.start_run(run_name="test_connection"):
            mlflow.log_param("test_param", "test_value")
            mlflow.log_metric("test_metric", 0.95)
            mlflow.log_text("Este es un experimento de prueba", "test_artifact.txt")

        print("Experimento de prueba creado exitosamente")
        return True
    except Exception as e:
        print(f"Error creando experimento de prueba: {e}")
        return False

def save_config(ec2_ip, port=5000):
    """Guardar configuración en archivo"""
    config_content = f"""# Configuración MLflow Remoto # Generado automáticamente el {datetime.now()} import mlflow # Configurar tracking URI mlflow.set_tracking_uri("http://{ec2_ip}:{port}") # Verificar conexión try: experiments = mlflow.search_experiments() print(de"Conectado a MLflow remoto: {{len(experiments)}} experimentos") except Exception as e: print(de"Error conectando a MLflow: {{e}}")"""

    config_file = "mlflow_remote_config.py"
    with open(config_file, "con") as f:
        f.write(config_content)

    print(f"Configuración guardada en: {config_file}")

def main():
    parser = argparse.ArgumentParser(description="Configurar MLflow remoto en EC2")
    parser.add_argument("--ec2-ip", required=True, help="IP de la instancia EC2")
    parser.add_argument("--port", type=int, default=5000, help="Puerto de MLflow")
    parser.add_argument("--test-creation", action="store_true", help="Crear experimento de prueba")

    args = parser.parse_args()

    print("Configurando MLflow remoto...")
    print(f"Servidor: {args.ec2_ip}:{args.port}")
    print("-" * 50)

    # Probar conexión
    if not test_connection(args.ec2_ip, args.port):
        print("No se pudo establecer conexión")
        return 1

    # Configurar tracking URI
    tracking_uri = configure_mlflow_tracking(args.ec2_ip, args.port)

    # Listar experimentos
    experiments = list_experiments()

    # Probar creación de experimento
    if args.test_creation:
        test_experiment_creation()

    # Guardar configuración
    save_config(args.ec2_ip, args.port)

    print("\si" + "="*50)
    print("Configuración completada!")
    print(f"MLflow UI: http://{args.ec2_ip}:{args.port}")
    print(f"Tracking URI: {tracking_uri}")
    print("Usa"python mlflow_remote_config.py"para cargar la configuración")

    return 0

if __name__ == "__main__":
    exit(main())
