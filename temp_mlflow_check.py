import mlflow
import mlflow.sklearn
import pandas as pd

# Configurar conexión remota
mlflow.set_tracking_uri('http://52.0.127.25:5001')

try:
    # Obtener detalles de cada experimento
    experiments = mlflow.search_experiments()

    print("📊 DETALLES DE EXPERIMENTOS MLFLOW REMOTO")
    print("=" * 50)

    for exp in experiments:
        if exp.name != 'Default':  # Saltar Default
            print(f'\n📊 Experimento: {exp.name} (ID: {exp.experiment_id})')

            # Obtener runs del experimento
            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
            if runs is not None and not runs.empty:
                print(f'  📈 Runs: {len(runs)}')

                # Mostrar métricas del primer run
                first_run = runs.iloc[0]
                print(f'  📏 Estado: {first_run.get("status", "N/A")}')

                # Mostrar algunas métricas si existen
                metrics_cols = [col for col in runs.columns if col.startswith('metrics.')]
                if metrics_cols:
                    print(f'  📊 Métricas disponibles: {len(metrics_cols)}')
                    for col in metrics_cols[:3]:  # Primeras 3 métricas
                        metric_name = col.replace('metrics.', '')
                        value = first_run.get(col, 'N/A')
                        if pd.notna(value):
                            print(f'    • {metric_name}: {value:.3f}')
                else:
                    print('  📊 Sin métricas disponibles')
            else:
                print('  📈 Runs: 0')

    print("\n✅ Validación completada exitosamente!")

except Exception as e:
    print(f'❌ Error: {str(e)}')
