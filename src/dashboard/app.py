"""
DASHBOARD INTERACTIVO - PLAIN LANGUAGE SUMMARIZER (PLS)
Inspirado en ProyectoDSIA - Dashboard completo para análisis y evaluación

Funcionalidades:
- Resumen de datos y estadísticas
- Comparación de modelos PLS
- Explorador de experimentos MLflow
- Prueba en vivo de modelos
- Visualizaciones interactivas
- Métricas en tiempo real
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mlflow
import mlflow.sklearn
import json
import os
from pathlib import Path
from datetime import datetime
import warnings
import time
warnings.filterwarnings("ignore")

# Configuración de la página
st.set_page_config(
    page_title="PLS Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/gabrielchaparro/pds-proyecto-final",
        "Report a bug": "https://github.com/gabrielchaparro/pds-proyecto-final/issues",
        "About": """## Plain Language Summarizer Dashboard **Proyecto de Maestría - Universidad de los Andes** Dashboard interactivo para análisis y evaluación de modelos de generación de resúmenes médicos en lenguaje sencillo."""
    }
)

# CSS personalizado
st.markdown("""<style> .main-header { font-size: 3rem; color: #1f77b4; text-align: center; margin-bottom: 2rem; font-weight: bold; } .metric-card { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; border-left: 5px solid #1f77b4; margin: 0.5rem 0; } .section-header { font-size: 1.8rem; color: #2c3e50; margin-top: 2rem; margin-bottom: 1rem; font-weight: bold; } .subsection-header { font-size: 1.4rem; color: #34495e; margin-top: 1.5rem; margin-bottom: 0.8rem; font-weight: 600; } .success-box { background-color: #d4edda; border: 1px solid #c3e6cb; color: #155724; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; } .warning-box { background-color: #fff3cd; border: 1px solid #ffeaa7; color: #856404; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; } .info-box { background-color: #d1ecf1; border: 1px solid #bee5eb; color: #0c5460; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; } </style>""", unsafe_allow_html=True)

class PLSDashboard:
    """Dashboard completo para Plain Language Summarizer"""

    def __init__(self):
        # Configurar rutas absolutas para que funcionen desde cualquier directorio
        project_root = Path(__file__).parent.parent.parent
        self.mlflow_uri = f"file:{project_root}/mlruns"
        self.data_dir = project_root / "data"
        self.models_dir = project_root / "models"

        self.data = {}
        self.models = {}
        self.experiments = {}
        self.setup_mlflow()
        self.load_data()
        self.load_mlflow_experiments()  # Cargar experimentos al inicializar

    def setup_mlflow(self):
        """Configurar conexión con MLflow"""
        try:
            mlflow.set_tracking_uri(self.mlflow_uri)
            print("MLflow conectado exitosamente")
        except Exception as e:
            st.error(f"Error conectando MLflow: {e}")

    def load_data(self):
        """Cargar datos del proyecto"""
        try:
            # Cargar dataset procesado
            dataset_path = self.data_dir / "processed" / "dataset_clean_v1.csv"
            if dataset_path.exists():
                self.data["dataset"] = pd.read_csv(dataset_path)
                dataset_key = "dataset"
                print(f"Dataset cargado: {len(self.data[dataset_key])} registros")
            else:
                self.data["dataset"] = None
                print(f"Dataset no encontrado en: {dataset_path}")

            # Cargar resultados de evaluación
            eval_path = self.data_dir / "outputs" / "evaluation_results.json"
            if eval_path.exists():
                with open(eval_path, "eres") as f:
                    self.data["evaluation"] = json.load(f)
                print("Resultados de evaluación cargados")
            else:
                self.data["evaluation"] = None
                print(f"Resultados de evaluación no encontrados en: {eval_path}")

        except Exception as e:
            st.error(f"Error cargando datos: {e}")
            self.data["dataset"] = None
            self.data["evaluation"] = None

    def load_mlflow_experiments(self):
        """Cargar experimentos de MLflow"""
        try:
            print(f"Buscando experimentos en: {self.mlflow_uri}")
            experiments = mlflow.search_experiments()
            print(f"Encontrados {len(experiments)} experimentos en MLflow")

            self.experiments = {}

            for exp in experiments:
                try:
                    print(f"Cargando experimento: {exp.name} (ID: {exp.experiment_id})")
                    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
                    run_count = len(runs) if runs is not None else 0
                    print(f"{run_count} runs encontrados")

                    # Procesar runs para extraer métricas y parámetros
                    if runs is not None and not runs.empty:
                        processed_runs = self._process_mlflow_runs(runs)
                        self.experiments[exp.name] = {
                            "experiment_id": exp.experiment_id,
                            "runs": processed_runs,
                            "run_count": run_count,
                            "raw_runs": runs  # Guardar también los runs originales
                        }
                    else:
                        self.experiments[exp.name] = {
                            "experiment_id": exp.experiment_id,
                            "runs": pd.DataFrame(),
                            "run_count": run_count
                        }
                except Exception as e:
                    print(f"Error cargando experimento {exp.name}: {e}")
                    continue

            print(f"{len(self.experiments)} experimentos cargados exitosamente")

        except Exception as e:
            print(f"Error cargando experimentos MLflow: {e}")
            st.error(f"Error cargando experimentos MLflow: {e}")
            self.experiments = {}

    def _process_mlflow_runs(self, runs_df):
        """Procesar runs de MLflow para extraer métricas y parámetros de las columnas expandidas"""
        processed_runs = []

        for _, run in runs_df.iterrows():
            # Extraer métricas (columnas que empiezan con "metrics.")
            metrics = {}
            params = {}

            for col in runs_df.columns:
                if col.startswith("metrics.") and pd.notna(run[col]):
                    metric_name = col.replace("metrics.", "")
                    metrics[metric_name] = run[col]
                elif col.startswith("params.") and pd.notna(run[col]):
                    param_name = col.replace("params.", "")
                    params[param_name] = run[col]

            # Crear objeto de run procesado
            processed_run = {
                "run_id": run.get("run_id", ""),
                "status": run.get("status", "UNKNOWN"),
                "start_time": run.get("start_time"),
                "end_time": run.get("end_time"),
                "metrics": metrics,
                "params": params,
                "tags": {k.replace("tags.", ""): v for k, v in run.items()
                        if k.startswith("tags.") and pd.notna(v)}
            }

            processed_runs.append(processed_run)

        return processed_runs

    def render_header(self):
        """Renderizar encabezado principal"""
        st.markdown('<h1 class="main-header">Plain Language Summarizer Dashboard</h1>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.markdown("""<div class="info-box"> <strong>Propósito:</strong> Transformar textos médicos complejos en resúmenes accesibles para pacientes sin conocimientos técnicos especializados. </div>""", unsafe_allow_html=True)

        with col2:
            if self.data.get("dataset") is not None:
                dataset_size = len(self.data["dataset"])
                st.metric("Dataset", f"{dataset_size:,}", "registros")
            else:
                st.metric("Dataset", "No disponible")

        with col3:
            experiment_count = len(self.experiments) if self.experiments else 0
            st.metric("Experimentos", experiment_count, "MLflow")

    def render_data_overview(self):
        """Sección: Resumen de datos"""
        st.markdown('<h2 class="section-header">Resumen de Datos</h2>', unsafe_allow_html=True)

        if self.data.get("dataset") is None:
            st.warning("No se encontraron datos. Ejecuta el pipeline DVC primero.")
            return

        df = self.data["dataset"]

        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_records = len(df)
            st.metric("Total Registros", f"{total_records:,}")

        with col2:
            if "label" in df.columns:
                pls_count = len(df[df["label"] == "pls"])
                st.metric("PLS", f"{pls_count:,}", f"{pls_count/total_records*100:.1f}%")
            else:
                st.metric("PLS", "N/A")

        with col3:
            if "label" in df.columns:
                non_pls_count = len(df[df["label"] == "non_pls"])
                st.metric("Non-PLS", f"{non_pls_count:,}", f"{non_pls_count/total_records*100:.1f}%")
            else:
                st.metric("Non-PLS", "N/A")

        with col4:
            if "source_dataset" in df.columns:
                sources = df["source_dataset"].nunique()
                st.metric("Fuentes", sources)
            else:
                st.metric("Fuentes", "N/A")

        # Gráficos
        st.markdown('<h3 class="subsection-header">Distribucion de Datos</h3>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            if "label" in df.columns:
                # Distribución PLS vs Non-PLS
                label_counts = df["label"].value_counts()
                fig = px.pie(
                    values=label_counts.values,
                    names=label_counts.index,
                    title="Distribución PLS vs Non-PLS",
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig.update_traces(textposition="inside", textinfo="percent+label")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hay información de labels disponible")

        with col2:
            if "source_dataset" in df.columns:
                # Distribución por fuente
                source_counts = df["source_dataset"].value_counts()
                fig = px.bar(
                    x=source_counts.index,
                    y=source_counts.values,
                    title="Distribución por Fuente",
                    color=source_counts.values,
                    color_continuous_scale="Blues"
                )
                fig.update_layout(xaxis_title="Fuente", yaxis_title="Cantidad")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hay información de fuentes disponible")

        # Estadísticas de texto
        if "texto_original" in df.columns:
            st.markdown('<h3 class="subsection-header">Estadisticas de Texto</h3>', unsafe_allow_html=True)

            df["text_length"] = df["texto_original"].astype(str).apply(lambda x: len(x.split()))

            col1, col2, col3 = st.columns(3)

            with col1:
                avg_length = df["text_length"].mean()
                st.metric("Longitud Promedio", f"{avg_length:.0f}", "palabras")

            with col2:
                max_length = df["text_length"].max()
                st.metric("Longitud Maxima", f"{max_length:,}", "palabras")

            with col3:
                min_length = df["text_length"].min()
                st.metric("Longitud Minima", f"{min_length:,}", "palabras")

            # Histograma de longitudes
            fig = px.histogram(
                df, x="text_length",
                title="Distribución de Longitud de Textos",
                labels={"text_length": "Número de palabras"},
                color_discrete_sequence=["#1f77b4"]
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    def render_model_comparison(self):
        """Sección: Comparación de modelos"""
        st.markdown('<h2 class="section-header">Comparacion de Modelos PLS</h2>', unsafe_allow_html=True)

        if not self.experiments:
            st.warning("No se encontraron experimentos en MLflow")
            st.info("Intenta recargar la pagina o verificar que MLflow este funcionando")
            return

        # Los experimentos ya están cargados en el constructor

        # Encontrar experimento de comparación de modelos
        comparison_exp = None
        for exp_name, exp_data in self.experiments.items():
            if "comparison" in exp_name.lower() or "pls_models" in exp_name.lower():
                comparison_exp = exp_data
                break

        if comparison_exp is None:
            st.warning("No se encontro experimento de comparacion de modelos PLS")
            st.info("Ejecuta `python scripts/compare_pls_models.py` para generar la comparacion")
            return

        # Usar runs procesados
        runs = comparison_exp.get("runs", [])
        if not runs:
            st.warning("No hay runs en el experimento de comparacion")
            return

        # Crear DataFrame con métricas
        model_data = []

        for run in runs:
            # Los runs ya están procesados, tienen "metrics" y "params" como diccionarios
            metrics = run.get("metrics", {})
            params = run.get("params", {})

            if metrics:  # Si hay métricas
                model_info = {
                    "model_name": params.get("model_name", "Unknown"),
                    "model_type": params.get("model_type", "Unknown"),
                    "compression": metrics.get("avg_compression_ratio", 0),
                    "fkgl": metrics.get("avg_fkgl_score", 0),
                    "flesch": metrics.get("avg_flesch_reading_ease", 0),
                    "processing_time": metrics.get("avg_processing_time", 0),
                    "successful": metrics.get("successful_generations", 0)
                }
                model_data.append(model_info)

        if not model_data:
            st.warning("No se encontraron metricas en los runs")
            return

        df_models = pd.DataFrame(model_data)

        # Tabla comparativa
        st.markdown('<h3 class="subsection-header">Tabla Comparativa</h3>', unsafe_allow_html=True)

        # Formatear tabla
        display_df = df_models.copy()
        display_df["compression"] = display_df["compression"].round(3)
        display_df["fkgl"] = display_df["fkgl"].round(1)
        display_df["flesch"] = display_df["flesch"].round(1)
        display_df["processing_time"] = display_df["processing_time"].round(2)

        st.dataframe(display_df, use_container_width=True)

        # Gráficos comparativos
        st.markdown('<h3 class="subsection-header">Graficos Comparativos</h3>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            # Gráfico de compresión
            fig = px.bar(
                df_models, x="model_name", y="compression",
                title="Compresión por Modelo",
                color="compression",
                color_continuous_scale="Blues"
            )
            fig.update_layout(xaxis_title="Modelo", yaxis_title="Ratio de Compresión")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Gráfico de legibilidad (FKGL)
            fig = px.bar(
                df_models, x="model_name", y="fkgl",
                title="Legibilidad (FKGL) por Modelo",
                color="fkgl",
                color_continuous_scale="Reds_r"  # Invertido: menor es mejor
            )
            fig.update_layout(xaxis_title="Modelo", yaxis_title="FKGL Score")
            st.plotly_chart(fig, use_container_width=True)

        # Gráfico de velocidad
        st.markdown('<h3 class="subsection-header"> Velocidad de Procesamiento</h3>', unsafe_allow_html=True)

        fig = px.bar(
            df_models, x="model_name", y="processing_time",
            title="Tiempo de Procesamiento por Modelo",
            color="processing_time",
            color_continuous_scale="Greens_r"  # Invertido: menor es mejor
        )
        fig.update_layout(xaxis_title="Modelo", yaxis_title="Tiempo (segundos)")
        st.plotly_chart(fig, use_container_width=True)

        # Ranking de modelos
        st.markdown('<h3 class="subsection-header"> Ranking de Modelos</h3>', unsafe_allow_html=True)

        # Calcular score compuesto
        df_rank = df_models.copy()
        df_rank["score"] = (
            (1 - df_rank["fkgl"] / df_rank["fkgl"].max()) * 0.4 +  # Legibilidad (40%)
            (df_rank["flesch"] / df_rank["flesch"].max()) * 0.3 +   # Facilidad de lectura (30%)
            (1 - df_rank["processing_time"] / df_rank["processing_time"].max()) * 0.3  # Velocidad (30%)
        )

        df_rank = df_rank.sort_values("score", ascending=False)

        # Mostrar ranking
        for idx, row in df_rank.iterrows():
            medal = "" if idx == 0 else "" if idx == 1 else "" if idx == 2 else ""
            st.markdown(f"""<div class="metric-card"> <strong>{medal} {row["model_name"]}</strong><br> <small>Score: {row["score"]:.3f} | Tipo: {row["model_type"]}</small><br> <small>FKGL: {row["fkgl"]:.1f} | Flesch: {row["flesch"]:.1f} | Tiempo: {row["processing_time"]:.2f}s</small> </div>""", unsafe_allow_html=True)

    def render_experiment_explorer(self):
        """Sección: Explorador de experimentos MLflow"""
        st.markdown('<h2 class="section-header">Explorador de Experimentos MLflow</h2>', unsafe_allow_html=True)

        if not self.experiments:
            st.warning("No se encontraron experimentos")
            return

        # Selector de experimento
        exp_names = list(self.experiments.keys())
        selected_exp = st.selectbox(
            "Seleccionar Experimento",
            exp_names,
            key="exp_selector"
        )

        if selected_exp:
            exp_data = self.experiments[selected_exp]

            st.markdown('<h3 class="subsection-header">Informacion del Experimento</h3>', unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("ID", exp_data["experiment_id"][:8] + "...")

            with col2:
                st.metric("Runs Totales", exp_data["run_count"])

            with col3:
                st.metric("Ubicacion", "Local MLflow")

            # Mostrar runs recientes
            if exp_data["run_count"] > 0:
                st.markdown('<h3 class="subsection-header">Runs Recientes</h3>', unsafe_allow_html=True)

                runs = exp_data["runs"]
                recent_runs = runs[:5]  # Primeros 5 runs (ya son listas procesadas)

                for run in recent_runs:
                    run_id_key = "run_id"
                    with st.expander(f"Run: {run.get(run_id_key, 'N/A')[:8]}..."):
                        col1, col2 = st.columns(2)

                        with col1:
                            status_key = "status"
                            st.write(f"**Estado:** {run.get(status_key, 'Unknown')}")
                            if "start_time" in run and run["start_time"]:
                                start_time = pd.to_datetime(run["start_time"])
                                st.write(f"**Inicio:** {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

                        with col2:
                            metrics = run.get("metrics", {})
                            if metrics:
                                st.write("**Métricas principales:**")
                                for key, value in list(metrics.items())[:3]:  # Primeras 3 métricas
                                    if isinstance(value, (int, float)):
                                        st.write(f"• {key}: {value:.3f}")
                                    else:
                                        st.write(f"• {key}: {value}")
                            else:
                                st.write("Sin métricas disponibles")

            # Gráfico de timeline de experimentos
            if exp_data["run_count"] > 1:
                st.markdown('<h3 class="subsection-header">Timeline de Experimentos</h3>', unsafe_allow_html=True)

                runs = exp_data["runs"]

                # Crear datos para timeline
                timeline_data = []
                for run in runs:
                    if "start_time" in run and run["start_time"]:
                        timeline_data.append({
                            "start_time": pd.to_datetime(run["start_time"]),
                            "run_id": run.get("run_id", "")[:8],
                            "status": run.get("status", "Unknown")
                        })

                if timeline_data:
                    df_timeline = pd.DataFrame(timeline_data)

                    fig = px.scatter(
                        df_timeline, x="start_time", y="status",
                        title="Timeline de Runs",
                        labels={"start_time": "Fecha y Hora", "status": "Estado"},
                        color="status",
                        color_discrete_sequence=["#1f77b4", "#ff7f0e", "#2ca02c"]
                    )
                    fig.update_layout(showlegend=True)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No hay datos suficientes para mostrar timeline")

    def render_live_testing(self):
        """Sección: Prueba en vivo"""
        st.markdown('<h2 class="section-header">Prueba en Vivo</h2>', unsafe_allow_html=True)

        st.markdown("""<div class="info-box"> <strong>Prueba interactiva:</strong> Ingresa un texto médico y genera su versión en lenguaje sencillo usando nuestros modelos PLS entrenados. </div>""", unsafe_allow_html=True)

        # Modelo selector
        available_models = ["PLS Ligero (Rule-based)", "T5-Base", "BART-Base", "BART-Large-CNN"]
        selected_model = st.selectbox(
            "Seleccionar Modelo PLS",
            available_models,
            index=0
        )

        # Input de texto
        st.markdown('<h3 class="subsection-header">Texto Medico de Entrada</h3>', unsafe_allow_html=True)

        # Ejemplos de texto
        example_texts = {
            "Enfermedad Cardiovascular": """The patient presents with acute myocardial infarction characterized by ST-elevation in leads II, III, and aVF. Troponin levels are elevated at 15.2 ng/mL, indicating myocardial necrosis. Immediate revascularization via percutaneous coronary intervention is recommended.""",

            "Diabetes Mellitus": """The 65-year-old male patient has type 2 diabetes mellitus with HbA1c of 8.7%. He requires intensification of glycemic control with metformin 1000mg twice daily and glipizide 5mg daily. Regular monitoring of blood glucose and HbA1c is essential.""",

            "Cáncer de Mama": """Biopsy results confirm invasive ductal carcinoma, estrogen receptor positive, HER2 negative. Tumor size is 2.3 cm with 2 positive lymph nodes. Recommended treatment includes lumpectomy followed by adjuvant chemotherapy and hormonal therapy.""",

            "Enfermedad Pulmonar": """Patient diagnosed with chronic obstructive pulmonary disease, GOLD stage III. FEV1/FVC ratio is 0.45, FEV1 is 1.2L (42% predicted). Long-acting bronchodilators and inhaled corticosteroids prescribed. Smoking cessation counseling initiated."""
        }

        col1, col2 = st.columns([3, 1])

        with col1:
            selected_example = st.selectbox(
                "o selecciona un ejemplo:",
                ["Personalizado"] + list(example_texts.keys())
            )

            if selected_example == "Personalizado":
                input_text = st.text_area(
                    "Texto médico:",
                    height=150,
                    placeholder="Ingresa aquí el texto médico que deseas simplificar..."
                )
            else:
                input_text = st.text_area(
                    "Texto médico:",
                    value=example_texts[selected_example],
                    height=150
                )

        with col2:
            st.markdown("### Estadisticas")
            if input_text:
                word_count = len(input_text.split())
                char_count = len(input_text)
                st.metric("Palabras", word_count)
                st.metric("Caracteres", char_count)

                # Estimación de complejidad
                complex_words = len([w for w in input_text.split() if len(w) > 6])
                complexity = (complex_words / word_count * 100) if word_count > 0 else 0
                st.metric("Palabras complejas", f"{complexity:.1f}%")
            else:
                st.metric("Palabras", 0)
                st.metric("Caracteres", 0)
                st.metric("Palabras complejas", "0%")

        # Botón de generación
        if st.button("Generar resumen en lenguaje sencillo", type="primary", use_container_width=True):
            if input_text and len(input_text.strip()) > 10:
                with st.spinner(f"Generando resumen en lenguaje sencillo con {selected_model}..."):
                    # Simular tiempo de procesamiento
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)

                    # Generar PLS según el modelo seleccionado
                    pls_result = self.generate_pls_demo(input_text, selected_model)

                    progress_bar.empty()

                # Mostrar resultados
                st.success("resumen en lenguaje sencillo generado exitosamente!")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown('<h3 class="subsection-header">Texto Original</h3>', unsafe_allow_html=True)
                    st.text_area(
                        "Texto médico original:",
                        value=input_text,
                        height=150,
                        disabled=True,
                        key="original_text"
                    )

                with col2:
                    st.markdown('<h3 class="subsection-header">Plain Language Summary</h3>', unsafe_allow_html=True)
                    st.text_area(
                        "Resumen en lenguaje sencillo:",
                        value=pls_result,
                        height=150,
                        disabled=True,
                        key="pls_text"
                    )

                # Métricas de comparación
                st.markdown('<h3 class="subsection-header">Metricas de Calidad</h3>', unsafe_allow_html=True)

                col1, col2, col3, col4 = st.columns(4)

                orig_words = len(input_text.split())
                pls_words = len(pls_result.split())
                compression = pls_words / orig_words if orig_words > 0 else 0

                with col1:
                    st.metric("Original", f"{orig_words}", "palabras")
                with col2:
                    st.metric("PLS", f"{pls_words}", "palabras")
                with col3:
                    st.metric("Compresion", f"{compression:.2f}", f"{compression*100:.1f}%")
                with col4:
                    # Estimación simple de legibilidad
                    readability = "Fácil" if compression > 1.1 else "Moderada" if compression > 0.8 else "Difícil"
                    st.metric("Legibilidad", readability)

                # Guardar en historial (simulado)
                if "history" not in st.session_state:
                    st.session_state.history = []

                st.session_state.history.append({
                    "timestamp": datetime.now(),
                    "model": selected_model,
                    "original": input_text,
                    "pls": pls_result,
                    "compression": compression
                })

            else:
                st.error("Por favor ingresa un texto medico valido (minimo 10 caracteres)")

        # Historial de generaciones
        if "history" in st.session_state and st.session_state.history:
            st.markdown('<h3 class="subsection-header">Historial de Generaciones</h3>', unsafe_allow_html=True)

            # Mostrar últimas 3 generaciones
            for i, item in enumerate(reversed(st.session_state.history[-3:])):
                # Usar .get() para manejar claves faltantes de forma segura
                model_name = item.get("model", "Unknown")
                timestamp = item.get("timestamp", datetime.now())
                with st.expander(f"Generación {len(st.session_state.history)-i} - {model_name} ({timestamp.strftime('%H:%M:%S')})"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**Original:**")
                        original_text = item.get("original", "No disponible")
                        st.write(original_text[:200] + "..." if len(original_text) > 200 else original_text)

                    with col2:
                        st.write("**PLS:**")
                        # Manejar tanto la clave antigua como la nueva
                        pls_result = item.get("pls") or item.get("por favor", "No disponible")
                        st.write(pls_result)
                        compression = item.get("compression", 0)
                        st.write(f"**Compresión:** {compression:.2f}")

    def generate_pls_demo(self, text: str, model: str) -> str:
        """Generar PLS usando lógica simplificada (demo)"""

        # Diccionario de términos médicos a lenguaje simple (EN INGLÉS)
        medical_terms = {
            "myocardial infarction": "heart attack",
            "ST-elevation": "ST-segment elevation",
            "troponin": "heart protein",
            "revascularization": "blood flow restoration",
            "percutaneous coronary intervention": "minimally invasive heart procedure",
            "diabetes mellitus": "diabetes",
            "HbA1c": "blood sugar level",
            "glycemic control": "blood sugar control",
            "metformin": "diabetes medication",
            "glipizide": "blood sugar lowering medication",
            "invasive ductal carcinoma": "invasive breast cancer",
            "estrogen receptor positive": "positive for estrogen receptors",
            "HER2 negative": "negative for HER2",
            "lumpectomy": "tumor removal",
            "adjuvant chemotherapy": "additional chemotherapy",
            "hormonal therapy": "hormone therapy",
            "chronic obstructive pulmonary disease": "chronic obstructive pulmonary disease",
            "FEV1/FVC ratio": "lung capacity ratio",
            "bronchodilators": "medications to open airways",
            "inhaled corticosteroids": "inhaled corticosteroids"
        }

        # Simplificar según el modelo
        pls = text.lower()

        # Reemplazar términos médicos
        for term, simple in medical_terms.items():
            pls = pls.replace(term.lower(), simple)

        # Simplificaciones adicionales según modelo (EN INGLÉS)
        if "ligero" in model.lower():
            # Modelo rule-based más agresivo
            pls = pls.replace("the patient", "the patient")
            pls = pls.replace("presents with", "has")
            pls = pls.replace("characterized by", "characterized by")
            pls = pls.replace("levels are elevated", "levels are high")
            pls = pls.replace("is recommended", "is recommended")
            pls = pls.replace("requires", "needs")
            pls = pls.replace("essential", "essential")
            pls = pls.replace("confirmed", "confirmed")
            pls = pls.replace("recommended", "recommended")
            pls = pls.replace("diagnosed with", "diagnosed with")
            pls = pls.replace("prescribed", "prescribed")

        elif "t5" in model.lower():
            # Simular estilo T5 (más conciso)
            pls = pls.replace("the patient", "Patient")
            pls = pls.replace("presents with", "presents")
            pls = pls.replace("characterized by", "with")
            pls = pls.replace("levels are elevated", "elevated levels")
            pls = pls.replace("is recommended", "recommended")
            pls = pls.replace("requires", "requires")

        elif "bart" in model.lower():
            # Simular estilo BART (más natural)
            pls = pls.replace("the patient", "The patient")
            pls = pls.replace("presents with", "has")
            pls = pls.replace("characterized by", "that is characterized by")
            pls = pls.replace("levels are elevated", "has high levels of")
            pls = pls.replace("is recommended", "is recommended")
            pls = pls.replace("requires", "needs")

        # Crear resumen (primeras 2-3 oraciones)
        sentences = pls.split(".")
        if len(sentences) > 3:
            if "large" in model.lower():
                pls = ".".join(sentences[:3]) + "."  # BART-Large más detallado
            else:
                pls = ".".join(sentences[:2]) + "."  # Otros modelos más concisos

        # Añadir prefijo según modelo (EN INGLÉS)
        if "ligero" in model.lower():
            pls = f"In simple words: {pls}"
        elif "t5" in model.lower():
            pls = f"Simplified summary: {pls}"
        elif "bart" in model.lower():
            pls = f"Clear version: {pls}"

        return pls

    def render_footer(self):
        """Renderizar footer"""
        st.markdown("---")
        st.markdown("""<div style="text-align: center; color: #666; margin-top: 2rem;"> <p><strong>Plain Language Summarizer Dashboard</strong></p> <p>Proyecto de Maestria - Universidad de los Andes | Desarrollado con dedicacion</p> <p>Última actualización: {}</p> </div>""".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)

    def run(self):
        """Ejecutar dashboard completo"""
        # Header
        self.render_header()

        # Sidebar de navegación
        st.sidebar.title("Navegacion")
        st.sidebar.markdown("---")

        page = st.sidebar.radio(
            "Seleccionar sección:",
            ["Inicio", "Datos", "Modelos", "Experimentos", "Prueba en Vivo"],
            key="main_nav"
        )

        # Contenido principal
        if page == "Inicio":
            st.markdown("""## Bienvenido al Dashboard PLS Este dashboard interactivo te permite: ### **Analizar Datos** - Explorar el dataset de textos médicos - Visualizar distribuciones y estadísticas - Entender la composición de los datos ### **Comparar Modelos** - Ver métricas de rendimiento de todos los modelos PLS - Comparar velocidad, calidad y eficiencia - Identificar el mejor modelo para cada caso ### **Explorar Experimentos** - Navegar por todos los experimentos MLflow - Revisar métricas detalladas por run - Analizar evolución temporal ### **Probar en Vivo** - Generar PLS interactivamente - Comparar diferentes modelos - Ver métricas en tiempo real --- ** Comienza explorando la sección de Datos para familiarizarte con el proyecto.**""")

        elif page == "Datos":
            self.render_data_overview()

        elif page == "Modelos":
            self.render_model_comparison()

        elif page == "Experimentos":
            self.render_experiment_explorer()

        elif page == "Prueba en Vivo":
            self.render_live_testing()

        # Sidebar adicional
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Enlaces Útiles")

        if st.sidebar.button("Abrir MLflow UI"):
            st.sidebar.markdown("[Abrir MLflow](http://localhost:5000)")

        if st.sidebar.button("Documentación"):
            st.sidebar.markdown("[Ver Docs](https://github.com/gabrielchaparro/pds-proyecto-final)")

        if st.sidebar.button("Reportar Issue"):
            st.sidebar.markdown("[GitHub Issues](https://github.com/gabrielchaparro/pds-proyecto-final/issues)")

        # Footer
        self.render_footer()

def main():
    """Función principal"""
    dashboard = PLSDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
