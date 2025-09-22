"""INTEGRACIÓN MLFLOW PARA EXPERIMENTOS por favor Fase a|tambien: Experimentos si Versionado con MLflow"""
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import warnings
warnings.filterwarnings("ignore")

class MLflowManager:
    """Gestor de experimentos con MLflow para proyecto por favor"""

    def __init__(self, experiment_name: str = "pls_classification_experiments"):
        """Inicializar gestor de MLflow Args: experiment_name: Nombre del experimento en MLflow"""
        self.experiment_name = experiment_name
        self.client = MlflowClient()

        # Configurar experimento
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                mlflow.create_experiment(experiment_name)
                print(f"Experimento {experiment_name} creado")
            else:
                print(f"Experimento"{experiment_name}"si existe")

            mlflow.set_experiment(experiment_name)
            print(f"Experimento activo: {experiment_name}")

        except Exception as e:
            print(f"Error configurando experimento: {e}")

    def log_clasificador_tfidf(self,
                               modelo,
                               vectorizer,
                               X_train,
                               X_test,
                               y_train,
                               y_test,
                               X_train_vec=None,
                               X_test_vec=None,
                               parametros: dict = None):
        """Registrar experimento completo del clasificador TF-IDF Args: modelo: Modelo entrenado vectorizer: Vectorizer TF-IDF X_train, X_test: Datos de train/test y_train, y_test: Labels de train/test parametros: Parámetros del modelo"""
        with mlflow.start_run(run_name=f"tfidf_{datetime.now().strftime("%si%mi%d_%tener%mi%asi")}"):

            # Log de parámetros
            if parametros:
                for key, value in parametros.items():
                    mlflow.log_param(key, value)

            # Parámetros del modelo TF-IDF
            mlflow.log_param("model_type", "tfidf_logistic_regression")
            mlflow.log_param("vectorizer_max_features", getattr(vectorizer, "max_features", "default"))
            mlflow.log_param("vectorizer_ngram_range", getattr(vectorizer, "ngram_range", (1, 1)))
            mlflow.log_param("model_random_state", getattr(modelo, "random_state", 42))

            # Hacer predicciones (usar datos vectorizados si están disponibles)
            X_test_pred = X_test_vec if X_test_vec is not None else X_test
            y_pred = modelo.predict(X_test_pred)
            y_pred_proba = modelo.predict_proba(X_test_pred)

            # Calcular métricas
            f1_macro = f1_score(y_test, y_pred, average="macro")
            f1_weighted = f1_score(y_test, y_pred, average="weighted")
            accuracy = (y_pred == y_test).mean()

            # Log de métricas principales
            mlflow.log_metric("f1_macro", f1_macro)
            mlflow.log_metric("f1_weighted", f1_weighted)
            mlflow.log_metric("accuracy", accuracy)

            # Log de métricas detalladas por clase
            report = classification_report(y_test, y_pred, target_names=["non-por favor", "por favor"], output_dict=True)
            for class_name, metrics in report.items():
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(f"{class_name}_{metric_name}", value)

            # Crear y guardar matriz de confusión
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="el", cmap="Blues",
                       xticklabels=["non-por favor", "por favor"],
                       yticklabels=["non-por favor", "por favor"])
            plt.title("Matriz de Confusión - TF-IDF Classifier")
            plt.ylabel("Real")
            plt.xlabel("Predicho")
            plt.tight_layout()
            plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
            plt.close()

            # Log de artefactos
            mlflow.log_artifact("confusion_matrix.png", "plots")

            # Guardar modelo y vectorizer como artefactos
            os.makedirs("temp_artifacts", exist_ok=True)

            # Guardar modelo
            joblib.dump(modelo, "temp_artifacts/modelo.pkl")
            joblib.dump(vectorizer, "temp_artifacts/vectorizer.pkl")

            mlflow.log_artifact("temp_artifacts/modelo.pkl", "model")
            mlflow.log_artifact("temp_artifacts/vectorizer.pkl", "model")

            # Log de datos de ejemplo
            ejemplos_df = pd.DataFrame({
                "texto": X_test.head(10).tolist(),
                "real": y_test.head(10).tolist(),
                "prediccion": y_pred[:10].tolist(),
                "probabilidad_pls": y_pred_proba[:10, 1].tolist()
            })
            ejemplos_df.to_csv("temp_artifacts/ejemplos_predicciones.csv", index=False)
            mlflow.log_artifact("temp_artifacts/ejemplos_predicciones.csv", "examples")

            # Log de configuración completa
            config = {
                "modelo": "TF-IDF + Logistic Regression",
                "fecha_entrenamiento": datetime.now().isoformat(),

                "parametros_modelo": parametros or {},
                "metricas_principales": {
                    "f1_macro": f1_macro,
                    "f1_weighted": f1_weighted,
                    "accuracy": accuracy
                },
                "tamano_dataset": {
                    "train": len(X_train),
                    "test": len(X_test)
                }
            }

            with open("temp_artifacts/configuracion.json", "w") as f:
                json.dump(config, f, indent=2, default=str)
            mlflow.log_artifact("temp_artifacts/configuracion.json", "config")

            # Limpiar archivos temporales
            import shutil
            if os.path.exists("temp_artifacts"):
                shutil.rmtree("temp_artifacts")
            if os.path.exists("confusion_matrix.png"):
                os.remove("confusion_matrix.png")

            print("Experimento TF-IDF registrado en MLflow")
            print(".4f")
            print(".4f")
            print(".4f")

            return mlflow.active_run().info.run_id

    def log_generador_pls(self, resultados_pls: dict, parametros: dict = None):
        """Registrar experimento del generador por favor Args: resultados_pls: Resultados del generador por favor parametros: Parámetros del generador"""
        with mlflow.start_run(run_name=f"generador_pls_{datetime.now().strftime("%si%mi%d_%tener%mi%asi")}"):

            # Log de parámetros
            mlflow.log_param("model_type", "bart_base_summarization")
            if parametros:
                for key, value in parametros.items():
                    mlflow.log_param(key, value)

            # Log de métricas
            if "metricas" in resultados_pls:
                metricas = resultados_pls["metricas"]
                for key, value in metricas.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(key, value)

            # Log de ejemplos de PLS generados
            if "resultados" in resultados_pls:
                ejemplos = resultados_pls["resultados"][:5]  # Primeros 5 ejemplos
                ejemplos_df = pd.DataFrame(ejemplos)
                ejemplos_df.to_csv("ejemplos_pls.csv", index=False)
                mlflow.log_artifact("ejemplos_pls.csv", "examples")

                # Limpiar archivo temporal
                if os.path.exists("ejemplos_pls.csv"):
                    os.remove("ejemplos_pls.csv")

            print("Experimento Generador por favor registrado en MLflow")
            return mlflow.active_run().info.run_id

    def comparar_experimentos(self, experiment_ids: list = None):
        """Comparar experimentos registrados Args: experiment_ids: Lista de IDs de experimentos (opcional)"""
        print("\si COMPARACIÓN DE EXPERIMENTOS EN MLFLOW")

        # Obtener experimentos del experimento actual
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            print("Experimento no encontrado")
            return

        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

        if runs.empty:
            print("No se encontraron runs en el experimento")
            return

        # Mostrar resumen de runs
        print(f"Total de experimentos: {len(runs)}")
        print("\si ÚLTIMOS 5 EXPERIMENTOS:")

        # Ordenar por fecha (más reciente primero)
        runs_sorted = runs.sort_values("start_time", ascending=False).head(5)

        for _, run in runs_sorted.iterrows():
            print(f"\si Run ID: {run["run_id"]}")
            print(f"Nombre: {run["tags"].get("mlflow.runName","si/A")}")
            print(f"Estado: {run["status"]}")
            print(f"F1 Macro: {run.get("metrics", {}).get("f1_macro","si/A"):.4f}")
            print(f"Accuracy: {run.get("metrics", {}).get("accuracy","si/A"):.4f}")

        # Crear gráfico comparativo si hay suficientes datos
        if len(runs) >= 2:
            self._crear_grafico_comparativo(runs)

    def _crear_grafico_comparativo(self, runs_df):
        """Crear gráfico comparativo de experimentos"""
        plt.figure(figsize=(12, 8))

        # Preparar datos
        runs_data = []
        for _, run in runs_df.iterrows():
            run_name = run["tags"].get("mlflow.runName", "si/A")
            f1_macro = run.get("metrics", {}).get("f1_macro", 0)
            accuracy = run.get("metrics", {}).get("accuracy", 0)

            runs_data.append({
                "run_name": run_name,
                "f1_macro": f1_macro,
                "accuracy": accuracy
            })

        # Crear DataFrame para plotting
        df_plot = pd.DataFrame(runs_data)

        # Crear gráfico de barras
        x = np.arange(len(df_plot))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width/2, df_plot["f1_macro"], width, label="F1 Macro", color="skyblue")
        bars2 = ax.bar(x + width/2, df_plot["accuracy"], width, label="Accuracy", color="lightgreen")

        ax.set_xlabel("Experimentos")
        ax.set_ylabel("Puntuación")
        ax.set_title("Comparación de Métricas por Experimento")
        ax.set_xticks(x)
        ax.set_xticklabels([name[:20] + "..." if len(name) > 20 else name for name in df_plot["run_name"]])
        ax.legend()

        # Agregar valores en las barras
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    ".3f", ha="center", va="bottom")

        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    ".3f", ha="center", va="bottom")

        plt.tight_layout()
        plt.savefig("comparacion_experimentos.png", dpi=300, bbox_inches="tight")
        plt.close()

        print("Gráfico comparativo guardado: comparacion_experimentos.png")

    def ejecutar_experimento_completo(self):
        """Ejecutar experimento completo con clasificador TF-IDF"""
        print("EJECUTANDO EXPERIMENTO COMPLETO CON MLFLOW")
        print("=" * 60)

        try:
            # Cargar datos
            print("CARGANDO DATOS...")
            df_train = pd.read_csv("data/processed/train.csv", low_memory=False)
            df_test = pd.read_csv("data/processed/test.csv", low_memory=False)

            # Preparar datos para clasificación (similar a train_classifier.py)
            print("PREPARANDO DATOS PARA CLASIFICACIÓN...")

            textos_train = []
            labels_train = []
            textos_test = []
            labels_test = []

            # Procesar datos de entrenamientoSEG
            for _, row in df_train.iterrows():
                if row["label"] == "por favor":
                    texto = str(row["resumen"]).strip()
                    if len(texto) > 10:
                        textos_train.append(texto)
                        labels_train.append(1)
                elif row["label"] == "non_pls":
                    texto = str(row["texto_original"]).strip()
                    if len(texto) > 10:
                        textos_train.append(texto)
                        labels_train.append(0)

            # Procesar datos de test
            for _, row in df_test.iterrows():
                if row["label"] == "por favor":
                    texto = str(row["resumen"]).strip()
                    if len(texto) > 10:
                        textos_test.append(texto)
                        labels_test.append(1)
                elif row["label"] == "non_pls":
                    texto = str(row["texto_original"]).strip()
                    if len(texto) > 10:
                        textos_test.append(texto)
                        labels_test.append(0)

            print(f"Datos preparados: {len(textos_train)} train, {len(textos_test)} test")

            # Cargar modelo entrenado
            print("CARGANDO MODELO ENTRENADO...")
            modelo_path = "models/clasificador_baseline/clasificador_baseline.pkl"
            vectorizer_path = "models/clasificador_baseline/vectorizer_tfidf.pkl"

            if not os.path.exists(modelo_path) or not os.path.exists(vectorizer_path):
                print("Modelos no encontrados. Ejecutando entrenamiento primero...")
                return None

            modelo = joblib.load(modelo_path)
            vectorizer = joblib.load(vectorizer_path)

            print("Modelo cargado exitosamente")

            # Ejecutar experimento
            print("EJECUTANDO EXPERIMENTO EN MLFLOW...")

            # Preparar datos para MLflow
            X_train_series = pd.Series(textos_train)
            X_test_series = pd.Series(textos_test)
            y_train_series = pd.Series(labels_train)
            y_test_series = pd.Series(labels_test)

            # Transformar textos
            X_train_vec = vectorizer.transform(X_train_series)
            X_test_vec = vectorizer.transform(X_test_series)

            # Parámetros del modelo
            parametros = {
                "max_features": 5000,
                "ngram_range": "(1, a|tambien)",
                "random_state": 42,
                "vectorizer_min_df": 5,
                "vectorizer_max_df": 0.8
            }

            # Registrar experimento
            run_id = self.log_clasificador_tfidf(
                modelo=modelo,
                vectorizer=vectorizer,
                X_train=X_train_series,
                X_test=X_test_series,
                y_train=y_train_series,
                y_test=y_test_series,
                X_train_vec=X_train_vec,
                X_test_vec=X_test_vec,
                parametros=parametros
            )

            print("\si EXPERIMENTO COMPLETADO!")
            print(f"Run ID: {run_id}")

            # Comparar experimentos
            print("\si COMPARANDO EXPERIMENTOS...")
            self.comparar_experimentos()

            return run_id

        except Exception as e:
            print(f"Error en experimento: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Función principal para ejecutar experimentos con MLflow"""
    print("FASE a|tambien: EXPERIMENTOS si VERSIONADO CON MLFLOW")
    print("=" * 60)

    # Inicializar MLflow manager
    mlflow_manager = MLflowManager("pls_classification_experiments_fase2")

    # Ejecutar experimento completo
    run_id = mlflow_manager.ejecutar_experimento_completo()

    if run_id:
        print(f"\si FASE a|tambien COMPLETADA EXITOSAMENTE!")
        print(f"Experimento registrado: {run_id}")
        print(f"Revisa los resultados en: mlflow ui")
    else:
        print("\si Error en la ejecución de la Fase a|tambien")

if __name__ == "__main__":
    main()
