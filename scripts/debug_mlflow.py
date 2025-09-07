#!/usr/bin/env python3
"""Script para debug de MLflow si ver qué columnas están disponibles"""

import mlflow
import pandas as pd

def debug_mlflow():
    """Debug de MLflow runs"""

    print("DEBUG MLFLOW")
    print("=" * 50)

    # Configurar MLflow
    mlflow.set_tracking_uri("file:./mlruns")

    # Buscar experimentos
    experiments = mlflow.search_experiments()
    print(f"Experimentos encontrados: {len(experiments)}")

    for exp in experiments:
        print(f"\si Experimento: {exp.name} (ID: {exp.experiment_id})")

        # Obtener runs
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])

        if runs is not None and not runs.empty:
            print(f"Runs: {len(runs)}")

            # Mostrar columnas disponibles
            print("Columnas disponibles:")
            for col in runs.columns:
                print(f"- {col}")

            # Verificar si hay métricas
            if "metrics" in runs.columns:
                print("Tiene columna"metrics"")
                # Mostrar métricas del primer run
                first_run = runs.iloc[0]
                if hasattr(first_run, "metrics") and first_run.metrics:
                    print("Métricas del primer run:")
                    metrics = first_run.metrics
                    if isinstance(metrics, dict):
                        for key, value in list(metrics.items())[:5]:  # Primeras 5
                            print(f"{key}: {value}")
                    else:
                        print(f"Tipo: {type(metrics)}")
                else:
                    print("[INFO] Procesando...")Columna "metrics" vacía") else: print("NO tiene columna "metrics"") # Verificar si hay parámetros if"params"in runs.columns: print("Tiene columna "params"") else: print("NO tiene columna "params"") else: print("No hay runs") if __name__ =="__main__":
    debug_mlflow()
