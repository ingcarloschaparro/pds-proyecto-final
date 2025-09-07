#!/usr/bin/env python3
"""
Script simple para verificar experimentos en MLflow
"""

import mlflow

# Configurar MLflow para servidor remoto
MLFLOW_TRACKING_URI = "http://52.0.127.25:5001"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def verify_experiments():
    """Verificar experimentos en MLflow"""
    print("🔍 VERIFICANDO EXPERIMENTOS EN MLFLOW")
    print("=" * 50)
    print(f"Servidor: {MLFLOW_TRACKING_URI}")
    print()
    
    try:
        # Obtener todos los experimentos
        experiments = mlflow.search_experiments()
        print(f"✅ Conexión exitosa. Encontrados {len(experiments)} experimentos")
        print()
        
        # Lista de experimentos esperados
        expected_experiments = [
            "E2-T5-Base-Real-Data",
            "E2-BART-Base-Real-Data", 
            "E2-BART-Large-CNN-Real-Data",
            "E2-PLS-Ligero-Real-Data"
        ]
        
        print("📊 EXPERIMENTOS CON DATOS REALES:")
        print("-" * 40)
        
        found_experiments = []
        for exp in experiments:
            exp_name = exp.name
            if "Real-Data" in exp_name:
                found_experiments.append(exp_name)
                print(f"✅ {exp_name} (ID: {exp.experiment_id})")
                
                # Obtener runs del experimento
                runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
                print(f"   - Runs: {len(runs)}")
        
        print()
        print("🎯 VERIFICACIÓN:")
        print("-" * 20)
        
        all_found = True
        for expected in expected_experiments:
            if expected in found_experiments:
                print(f"✅ {expected}")
            else:
                print(f"❌ {expected} - NO ENCONTRADO")
                all_found = False
        
        print()
        if all_found:
            print("🎉 ¡TODOS LOS EXPERIMENTOS ESTÁN DISPONIBLES!")
        else:
            print("⚠️ Algunos experimentos no se encontraron")
        
        print()
        print("🌐 Accede al servidor MLflow:")
        print(f"   URL: {MLFLOW_TRACKING_URI}")
        
        return all_found
        
    except Exception as e:
        print(f"❌ Error verificando experimentos: {e}")
        return False

if __name__ == "__main__":
    verify_experiments()
