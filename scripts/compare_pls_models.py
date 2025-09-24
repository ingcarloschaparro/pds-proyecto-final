#!/usr/bin/env python3
"""Comparación completa de modelos PLS en MLflow Ejecuta y compara: BART-Base, BART-Large-CNN, T5-Base, PLS Ligero"""


import pandas as pd
import numpy as np
import torch
from transformers import pipeline
import mlflow
import mlflow.sklearn
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Dict, List, Any
import warnings
warnings.filterwarnings("ignore")

class PLSModelComparator:
    """Comparador de modelos PLS"""

    def __init__(self):
        self.models = {}
        self.test_texts = self._load_test_texts()
        self.setup_mlflow()
        self.initialize_models()

    def _load_test_texts(self) -> List[str]:
        """Cargar textos de prueba médicos"""
        return [
            "The study evaluated the effects of metformin on glycemic control in patients with type 2 diabetes mellitus. Participants received either metformin 500mg twice daily or placebo for 12 weeks. The primary outcome was change in HbA1c levels from baseline.",

            "Randomized controlled trial comparing laparoscopic vs open cholecystectomy for symptomatic cholelithiasis. The intervention group underwent laparoscopic procedure while control group received open surgery. Primary endpoints included operative time, postoperative complications, and length of hospital stay.",

            "Clinical trial investigating the efficacy of atorvastatin 20mg daily versus placebo in reducing cardiovascular events in patients with hypercholesterolemia. The study enrolled 500 participants and followed them for 2 years, measuring LDL cholesterol levels and incidence of myocardial infarction.",

            "A prospective cohort study examining the relationship between vitamin D deficiency and bone mineral density in postmenopausal women. The researchers measured serum 25-hydroxyvitamin D levels and performed dual-energy x-ray absorptiometry scans to assess BMD at the lumbar spine and femoral neck.",

            "Double-blind placebo-controlled trial assessing the effectiveness of cognitive behavioral therapy combined with selective serotonin reuptake inhibitors for treatment-resistant major depressive disorder. Patients were randomized to receive either CBT plus SSRI or SSRI plus placebo for 16 weeks."
        ]

    def setup_mlflow(self):
        """Configurar MLflow con fallback automático"""
        # Intentar MLflow remoto primero
        try:
            print("🔗 Intentando conectar a MLflow remoto (AWS)...")
            mlflow.set_tracking_uri("http://52.0.127.25:5001")
            # Probar la conexión
            client = mlflow.tracking.MlflowClient()
            client.list_experiments()
            mlflow.set_experiment("E2-PLS-Models-Comparison")
            print("✅ Conectado a MLflow remoto (AWS)")
        except Exception as e:
            print(f"❌ Error con MLflow remoto: {e}")
            print("🔄 Cambiando a MLflow local...")
            mlflow.set_tracking_uri("file:./mlruns")
            mlflow.set_experiment("E2-PLS-Models-Comparison")
            print("✅ Conectado a MLflow local")

    def initialize_models(self):
        """Inicializar todos los modelos PLS"""
        print("INICIALIZANDO MODELOS PLS")
        print("=" * 50)

        # Determinar dispositivo
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Dispositivo: {device}")

        # 1. BART-Base
        try:
            print("Cargando BART-Base...")
            self.models["bart_base"] = {
                "model": pipeline(
                    "summarization",
                    model="facebook/bart-base",
                    device=device,
                    max_length=100,
                    min_length=30,
                    num_beams=4,
                    temperature=0.8,
                    do_sample=True,
                    early_stopping=True
                ),
                "type": "transformer",
                "model_name": "BART-Base",
                "size": "Base"
            }
            print("BART-Base cargado")
        except Exception as e:
            print(f"Error cargando BART-Base: {e}")

        # 2. BART-Large-CNN
        try:
            print("Cargando BART-Large-CNN...")
            self.models["bart_large_cnn"] = {
                "model": pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn",
                    device=device,
                    max_length=100,
                    min_length=30,
                    num_beams=4,
                    temperature=0.8,
                    do_sample=True,
                    early_stopping=True
                ),
                "type": "transformer",
                "model_name": "BART-Large-CNN",
                "size": "Large"
            }
            print("BART-Large-CNN cargado")
        except Exception as e:
            print(f"Error cargando BART-Large-CNN: {e}")

        # 3. T5-Base
        try:
            print("Cargando T5-Base...")
            self.models["t5_base"] = {
                "model": pipeline(
                    "summarization",
                    model="t5-base",
                    device=device,
                    max_length=100,
                    min_length=30,
                    num_beams=4,
                    temperature=0.8,
                    do_sample=True,
                    early_stopping=True
                ),
                "type": "transformer",
                "model_name": "T5-Base",
                "size": "Base"
            }
            print("T5-Base cargado")
        except Exception as e:
            print(f"Error cargando T5-Base: {e}")

        # 4. PLS Ligero (Rule-based)
        self.models["pls_lightweight"] = {
            "model": None,  # Se manejará en el código de generación
            "type": "rule_based",
            "model_name": "PLS Ligero",
            "size": "Light"
        }
        print("PLS Ligero inicializado")

        print(f" -  Total modelos inicializados: {len(self.models)}")
        for name, config in self.models.items():
            print(f"• {name}: {config['model_name']} ({config['type']})")

    def generate_simple_pls(self, text: str) -> str:
        """Generar PLS usando reglas simples"""
        # Diccionario de términos médicos a lenguaje simple
        medical_terms = {
            "metformin": "medicamento para la diabetes",
            "glycemic control": "control del azúcar en sangre",
            "type 2 diabetes mellitus": "diabetes tipo 2",
            "placebo": "medicamento sin efecto real",
            "HbA1c": "nivel de azúcar en sangre",
            "randomized controlled trial": "estudio científico",
            "laparoscopic": "cirugía mínimamente invasiva",
            "cholecystectomy": "cirugía de vesícula biliar",
            "cholelithiasis": "piedras en la vesícula",
            "postoperative complications": "problemas después de la cirugía",
            "atorvastatin": "medicamento para el colesterol",
            "cardiovascular events": "problemas del corazón",
            "hypercholesterolemia": "colesterol alto",
            "LDL cholesterol": "colesterol malo",
            "myocardial infarction": "ataque al corazón",
            "vitamin d deficiency": "falta de vitamina D",
            "bone mineral density": "densidad ósea",
            "postmenopausal women": "mujeres después de la menopausia",
            "dual-energy ex-ray absorptiometry": "escáner especial de huesos",
            "cognitive behavioral therapy": "terapia cognitivo-conductual",
            "selective serotonin reuptake inhibitors": "antidepresivos",
            "major depressive disorder": "depresión mayor",
            "treatment-resistant": "resistente al tratamiento"
        }

        # Simplificar el texto
        pls = text.lower()

        # Reemplazar términos médicos
        for term, simple in medical_terms.items():
            pls = pls.replace(term.lower(), simple)

        # Simplificar estructura
        pls = pls.replace("participants received", "los pacientes tomaron")
        pls = pls.replace("the study", "el estudio")
        pls = pls.replace("evaluated", "evaluó")
        pls = pls.replace("investigating", "investigó")
        pls = pls.replace("primary outcome", "resultado principal")
        pls = pls.replace("primary endpoints", "objetivos principales")
        pls = pls.replace("prospective cohort study", "estudio de seguimiento")
        pls = pls.replace("double-blind placebo-controlled trial", "estudio clínico controlado")
        pls = pls.replace("clinical trial", "estudio clínico")

        # Crear resumen simple (primeras 2 oraciones)
        sentences = pls.split(".")
        if len(sentences) > 2:
            pls = ".".join(sentences[:2]) + "."

        # Añadir prefijo explicativo
        pls = f"En términos simples: {pls}"

        return pls

    def calculate_metrics(self, original_text: str, generated_text: str) -> Dict[str, float]:
        """Calcular métricas de calidad PLS"""
        try:
            import textstat

            # Longitudes
            orig_words = len(original_text.split())
            gen_words = len(generated_text.split())

            # Métricas básicas
            compression_ratio = gen_words / orig_words if orig_words > 0 else 0

            # Métricas de legibilidad (aproximadas)
            fkgl = textstat.flesch_kincaid_grade(generated_text) if gen_words > 10 else 12.0
            flesch_ease = textstat.flesch_reading_ease(generated_text) if gen_words > 10 else 50.0

            # Similitud simple (overlap de palabras)
            orig_words_set = set(original_text.lower().split())
            gen_words_set = set(generated_text.lower().split())
            overlap = len(orig_words_set.intersection(gen_words_set))
            total_words = len(orig_words_set.union(gen_words_set))
            word_overlap_ratio = overlap / total_words if total_words > 0 else 0

            return {
                "compression_ratio": compression_ratio,
                "fkgl_score": fkgl,
                "flesch_reading_ease": flesch_ease,
                "word_overlap_ratio": word_overlap_ratio,
                "original_length": orig_words,
                "generated_length": gen_words
            }

        except Exception as e:
            print(f"️ Error calculando métricas: {e}")
            return {
                "compression_ratio": 0.0,
                "fkgl_score": 12.0,
                "flesch_reading_ease": 50.0,
                "word_overlap_ratio": 0.0,
                "original_length": len(original_text.split()),
                "generated_length": len(generated_text.split())
            }

    def run_model_experiment(self, model_name: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecutar experimento completo para un modelo"""
        print(f" -  EJECUTANDO EXPERIMENTO: {model_name.upper()}")
        print("=" * 60)

        model = model_config["model"]
        results = []
        total_time = 0

        with mlflow.start_run(run_name=f"pls_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):

            # Log parámetros del modelo
            mlflow.log_params({
                "model_name": model_config["model_name"],
                "model_type": model_config["type"],
                "model_size": model_config["size"],
                "device": "cuda" if torch.cuda.is_available() else "cpu"
            })

            # Procesar cada texto de prueba
            for i, text in enumerate(self.test_texts, 1):
                print(f"Procesando texto {i}/5...")

                try:
                    start_time = datetime.now()

                    # Generar PLS
                    if model_config["type"] == "transformer":
                        # Para transformers, ajustar parámetros según el texto
                        text_words = len(text.split())
                        max_len = min(100, max(30, text_words // 2))
                        min_len = min(20, max(10, text_words // 8))
                        
                        # Asegurar que min_len < max_len
                        if min_len >= max_len:
                            min_len = max(10, max_len - 5)

                        summary = model(text, max_length=max_len, min_length=min_len)[0]["summary_text"]
                    else:
                        # Para rule-based (PLS Ligero)
                        summary = self.generate_simple_pls(text)

                    end_time = datetime.now()
                    processing_time = (end_time - start_time).total_seconds()

                    total_time += processing_time

                    # Calcular métricas
                    metrics = self.calculate_metrics(text, summary)

                    # Resultado individual
                    result = {
                        "text_id": i,
                        "original_text": text,
                        "generated_pls": summary,
                        "processing_time": processing_time,
                        **metrics
                    }

                    results.append(result)

                    print(f"Texto {i}: {len(summary.split())} palabras, {processing_time:.2f}s")

                except Exception as e:
                    print(f"Error en texto {i}: {e}")
                    results.append({
                        "text_id": i,
                        "original_text": text,
                        "generated_pls": f"Error: {e}",
                        "processing_time": 0.0,
                        "compression_ratio": 0.0,
                        "fkgl_score": 12.0,
                        "flesch_reading_ease": 50.0,
                        "word_overlap_ratio": 0.0,
                        "original_length": len(text.split()),
                        "generated_length": 0
                    })

            # Calcular métricas agregadas
            if results:
                compression_ratios = [r["compression_ratio"] for r in results]
                fkgl_scores = [r["fkgl_score"] for r in results]
                flesch_scores = [r["flesch_reading_ease"] for r in results]
                overlap_ratios = [r["word_overlap_ratio"] for r in results]
                processing_times = [r["processing_time"] for r in results]

                aggregated_metrics = {
                    "avg_compression_ratio": np.mean(compression_ratios),
                    "avg_fkgl_score": np.mean(fkgl_scores),
                    "avg_flesch_reading_ease": np.mean(flesch_scores),
                    "avg_word_overlap_ratio": np.mean(overlap_ratios),
                    "avg_processing_time": np.mean(processing_times),
                    "total_processing_time": total_time,
                    "texts_processed": len(results),
                    "successful_generations": len([r for r in results if not r["generated_pls"].startswith("Error")])
                }

                # Log métricas en MLflow
                mlflow.log_metrics(aggregated_metrics)

                # Log artifacts
                self._log_artifacts(model_name, results, aggregated_metrics)

                print(" -  MÉTRICAS FINALES:")
                print(f"• Compresión promedio: {aggregated_metrics['avg_compression_ratio']:.3f}")
                print(f"• FKGL promedio: {aggregated_metrics['avg_fkgl_score']:.1f}")
                print(f"• Flesch Reading Ease: {aggregated_metrics['avg_flesch_reading_ease']:.1f}")
                print(f"• Tiempo promedio: {aggregated_metrics['avg_processing_time']:.2f}s")
                print(f"• Generaciones exitosas: {aggregated_metrics['successful_generations']}/5")

                return {
                    "model_name": model_name,
                    "results": results,
                    "aggregated_metrics": aggregated_metrics,
                    "status": "completed"
                }

        return {
            "model_name": model_name,
            "results": results,
            "aggregated_metrics": {},
            "status": "failed"
        }

    def _log_artifacts(self, model_name: str, results: List[Dict], metrics: Dict):
        """Log artifacts en MLflow"""
        try:
            # Crear directorio para artifacts
            artifact_dir = f"artifacts/{model_name}"
            os.makedirs(artifact_dir, exist_ok=True)

            # Guardar resultados detallados
            results_df = pd.DataFrame(results)
            results_file = f"{artifact_dir}/detailed_results.csv"
            results_df.to_csv(results_file, index=False)

            # Guardar métricas
            metrics_file = f"{artifact_dir}/metrics.json"
            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)

            # Guardar ejemplos de generación
            examples = []
            for result in results[:3]:  # Primeros 3 ejemplos
                examples.append({
                    "original": result["original_text"][:200] + "...",
                    "generated": result["generated_pls"],
                    "compression": result["compression_ratio"]
                })

            examples_file = f"{artifact_dir}/examples.json"
            with open(examples_file, "w", encoding="utf-8") as f:
                json.dump(examples, f, indent=2, ensure_ascii=False)

            # Log archivos en MLflow
            mlflow.log_artifacts(artifact_dir)

        except Exception as e:
            print(f"️ Error guardando artifacts: {e}")

    def run_comparison(self):
        """Ejecutar comparación completa de todos los modelos"""
        print("INICIANDO COMPARACIÓN COMPLETA DE MODELOS PLS")
        print("=" * 70)

        if not self.models:
            print("No hay modelos inicializados")
            return

        all_results = []

        # Ejecutar cada modelo
        for model_name, model_config in self.models.items():
            try:
                result = self.run_model_experiment(model_name, model_config)
                all_results.append(result)
                print(f"Experimento {model_name} completado")
            except Exception as e:
                print(f"Error en experimento {model_name}: {e}")
                all_results.append({
                    "model_name": model_name,
                    "results": [],
                    "aggregated_metrics": {},
                    "status": "error",
                    "error": str(e)
                })

        # Crear reporte de comparación
        self.create_comparison_report(all_results)

        print(" -  COMPARACIÓN COMPLETA FINALIZADA")
        print("=" * 70)

        return all_results

    def create_comparison_report(self, all_results: List[Dict]):
        """Crear reporte de comparación"""
        print(" -  GENERANDO REPORTE DE COMPARACIÓN")
        print("=" * 50)

        # Filtrar resultados exitosos
        successful_results = [r for r in all_results if r["status"] == "completed"]

        if not successful_results:
            print("No hay resultados exitosos para comparar")
            return

        # Crear tabla comparativa
        comparison_data = []
        for result in successful_results:
            metrics = result["aggregated_metrics"]
            comparison_data.append({
                "Modelo": result["model_name"],
                "Tipo": self.models[result['model_name']]['type'],
                "Tamaño": self.models[result['model_name']]['size'],
                "Compresión": f"{metrics.get('avg_compression_ratio', 0):.3f}",
                "FKGL": f"{metrics.get('avg_fkgl_score', 0):.1f}",
                "Flesch": f"{metrics.get('avg_flesch_reading_ease', 0):.1f}",
                "Tiempo (s)": f"{metrics.get('avg_processing_time', 0):.2f}",
                "Éxitos": f"{metrics.get('successful_generations', 0)}/5"
            })

        comparison_df = pd.DataFrame(comparison_data)

        # Mostrar tabla
        print(" -  TABLA COMPARATIVA:")
        print(comparison_df.to_string(index=False))

        # Guardar reporte
        report_dir = "reports"
        os.makedirs(report_dir, exist_ok=True)

        report_file = f"{report_dir}/pls_models_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        comparison_df.to_csv(report_file, index=False)

        print(f" -  Reporte guardado en: {report_file}")

        # Determinar mejor modelo
        best_model = self._determine_best_model(successful_results)
        if best_model:
            print(f" -  MEJOR MODELO: {best_model}")

    def _determine_best_model(self, results: List[Dict]) -> str:
        """Determinar el mejor modelo basado en métricas"""
        try:
            scores = []
            for result in results:
                metrics = result["aggregated_metrics"]

                # Calcular score compuesto (menor es mejor para FKGL, mayor para otros)
                score = (
                    metrics.get("avg_compression_ratio", 0) * 0.3 +  # Compresión razonable
                    (1 - metrics.get("avg_fkgl_score", 12) / 12) * 0.4 +  # Menor FKGL es mejor
                    (metrics.get("avg_flesch_reading_ease", 50) / 100) * 0.3  # Mayor Flesch es mejor
                )

                scores.append((result["model_name"], score))

            # Ordenar por score descendente
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[0][0]

        except Exception as e:
            print(f"️ Error determinando mejor modelo: {e}")
            return None

def main():
    """Función principal"""
    print("COMPARADOR DE MODELOS PLS - MLFLOW")
    print("=" * 60)

    comparator = PLSModelComparator()

    # Inicializar modelos
    comparator.initialize_models()

    # Ejecutar comparación
    results = comparator.run_comparison()

    # Resumen final
    print(" -  RESUMEN DE EXPERIMENTOS:")
    successful = len([r for r in results if r["status"] == "completed"])
    total = len(results)
    print(f"• Modelos probados: {total}")
    print(f"• Modelos exitosos: {successful}")
    print(f"• Modelos fallidos: {total - successful}")

    if successful > 0:
        print(f"• Resultados disponibles en MLflow UI: http://localhost:5000")
        print(f"• Experimento: pls_models_comparison")

if __name__ == "__main__":
    main()
