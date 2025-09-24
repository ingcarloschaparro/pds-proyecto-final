#!/usr/bin/env python3
"""
Generador simple de Plain Language Summaries (PLS)
Versión elevada del generador simple con CLI completa, métricas y MLflow
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from src.config.mlflow_remote import apply_tracking_uri as _mlf_apply
    _mlf_apply(experiment="E2-Generator")
except (ImportError, Exception) as e:
    print(f"MLflow remote config failed: {e}")
    print("Using local MLflow tracking")
    import mlflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("E2-Generator")

import os
import sys
import yaml
import argparse
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import warnings

# Suprimir warnings de transformers
warnings.filterwarnings("ignore")

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    import torch
except ImportError:
    print("Error: transformers no está instalado. Instalar con: pip install transformers torch")
    sys.exit(1)

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    print("Warning: MLflow no está disponible. Instalar con: pip install mlflow")
    MLFLOW_AVAILABLE = False


class PLSGeneratorSimple:
    """Generador simple de Plain Language Summaries usando modelos pre-entrenados"""
    
    def __init__(self):
        self.pipeline = None
        self.model_name = None
        self.tokenizer = None
        self.model = None
        
    def cargar_modelo(self, model_name="facebook/bart-large-cnn", config=None):
        """
        Cargar modelo de generación de resúmenes
        
        Args:
            model_name: Nombre del modelo a cargar
            config: Configuración del generador
        """
        try:
            print(f"Cargando modelo: {model_name}")
            
            # Parámetros por defecto
            gen_params = {
                "model": model_name,
                "max_length": 150,
                "min_length": 50,
                "num_beams": 4,
                "temperature": 0.8,
                "do_sample": True,
                "truncation": True
            }
            
            # Usar configuración si está disponible
            if config and 'generator' in config:
                gen_config = config['generator']
                gen_params.update({
                    "model": gen_config.get('base_model', model_name),
                    "max_length": gen_config.get('max_length_tokens', 150),
                    "min_length": gen_config.get('min_length_tokens', 50),
                    "num_beams": gen_config.get('num_beams', 4),
                    "temperature": gen_config.get('temperature', 0.8),
                    "do_sample": gen_config.get('do_sample', True)
                })
            
            self.pipeline = pipeline("summarization", **gen_params)
            self.model_name = model_name
            print(f"Modelo {model_name} cargado exitosamente")
            
        except Exception as e:
            print(f"Error cargando modelo {model_name}: {e}")
            raise
    
    def generar_pls(self, texto: str, instrucciones: str = None, config: Dict = None) -> str:
        """
        Generar Plain Language Summary de un texto médico
        
        Args:
            texto: Texto médico original
            instrucciones: Instrucciones específicas para la generación
            config: Configuración del generador
            
        Returns:
            Resumen en lenguaje sencillo
        """
        if self.pipeline is None:
            raise ValueError("Modelo no cargado. Ejecutar cargar_modelo() primero")
        
        try:
            # Instrucciones por defecto - SIEMPRE usar inglés
            if instrucciones is None:
                if config and 'generator' in config and 'instructions' in config['generator']:
                    # Usar instrucciones del archivo de configuración (siempre en inglés)
                    instrucciones = config['generator']['instructions']
                else:
                    # Usar instrucciones por defecto en inglés
                    instrucciones = "Summarize this medical text in simple, clear language that a patient can understand: "
            
            # Combinar instrucciones con texto
            texto_completo = f"{instrucciones}{texto}"
            
            # Generar resumen
            resultados = self.pipeline(texto_completo)
            
            # Extraer texto del resumen
            summary = resultados[0]['summary_text'] if resultados else ""
            
            # Remover la instrucción si aparece en el resumen
            if instrucciones.lower() in summary.lower():
                summary = summary.replace(instrucciones, "").strip()
                # También remover variaciones comunes
                summary = summary.replace("summarize this medical text in simple, clear language that a patient can understand:", "").strip()
            
            return summary
            
        except Exception as e:
            return f"ERROR generando PLS: {str(e)}"


def main():
    """Función principal para probar el generador"""
    parser = argparse.ArgumentParser(description="Generador de Plain Language Summaries")
    parser.add_argument("--modelo", default="facebook/bart-large-cnn", help="Modelo a usar")
    parser.add_argument("--texto", help="Texto médico a procesar")
    parser.add_argument("--archivo", help="Archivo con textos médicos")
    parser.add_argument("--input", help="Archivo de entrada (para pipeline DVC)")
    parser.add_argument("--output", help="Directorio de salida (para pipeline DVC)")
    parser.add_argument("--zero-shot", action="store_true", help="Modo zero-shot con textos de ejemplo")
    parser.add_argument("--config", default="params.generator.yaml", help="Archivo de configuración")
    
    args = parser.parse_args()
    
    # Cargar configuración
    config = None
    if os.path.exists(args.config):
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print("Configuración cargada desde params.generator.yaml")
        except Exception as e:
            print(f"Error cargando configuración: {e}")
    
    # Crear generador
    generador = PLSGeneratorSimple()
    generador.cargar_modelo(args.modelo, config)
    
    if args.zero_shot:
        # Modo zero-shot con textos de ejemplo
        textos_prueba = [
            "The patient presents with acute myocardial infarction characterized by ST-elevation in leads II, III, and aVF. Troponin levels are elevated at 15.2 ng/mL, indicating myocardial necrosis. Immediate revascularization via percutaneous coronary intervention is recommended.",
            
            "A 65-year-old male with a history of hypertension and diabetes mellitus presents with chest pain radiating to the left arm. Electrocardiogram shows ST-segment elevation in leads V1-V4. Cardiac catheterization reveals 90% stenosis of the left anterior descending artery. The patient underwent successful percutaneous coronary intervention with drug-eluting stent placement.",
            
            "Clinical trial investigating the efficacy of atorvastatin 20mg daily versus placebo in reducing cardiovascular events in patients with hypercholesterolemia. The study enrolled 500 participants and followed them for two years, measuring LDL cholesterol levels and incidence of myocardial infarction."
        ]
        
        # Generar PLS
        print("GENERANDO PLS...")
        resultados = []
        
        for i, texto in enumerate(textos_prueba, 1):
            print(f"--- Texto {i} ---")
            print(f"Original: {texto[:100]}...")
            
            pls = generador.generar_pls(texto, config=config)
            print(f"PLS generado: {pls}")
            
            resultados.append({
                "id": i,
                "texto_original": texto,
                "pls_generado": pls,
                "longitud_original": len(texto),
                "longitud_pls": len(pls),
                "compression_ratio": len(pls) / len(texto) if len(texto) > 0 else 0
            })
        
        # Guardar resultados
        os.makedirs("data/outputs", exist_ok=True)
        df_resultados = pd.DataFrame(resultados)
        output_file = f"data/outputs/pls_test_{args.modelo.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_resultados.to_csv(output_file, index=False)
        
        print(f"Resultados guardados en: {output_file}")
        
        # Estadísticas
        pls_validos = [r for r in resultados if not r["pls_generado"].startswith("ERROR")]
        print(f"Estadísticas: {len(pls_validos)}/{len(resultados)} PLS generados exitosamente")
        
        if pls_validos:
            avg_compression = sum(r["compression_ratio"] for r in pls_validos) / len(pls_validos)
            print(f"Compresión promedio: {avg_compression:.2f}")
    
    elif args.texto:
        # Modo texto individual
        pls = generador.generar_pls(args.texto, config=config)
        print(f"Texto original: {args.texto}")
        print(f"PLS generado: {pls}")
    
    elif args.input and args.output:
        # Modo pipeline DVC
        if not os.path.exists(args.input):
            print(f"Error: Archivo {args.input} no encontrado")
            return
        
        # Crear directorio de salida si no existe
        os.makedirs(args.output, exist_ok=True)
        
        df = pd.read_csv(args.input)
        if 'texto_original' not in df.columns:
            print("Error: El archivo debe tener una columna 'texto_original'")
            return
        
        # Limitar a una muestra pequeña para el pipeline
        sample_size = min(100, len(df))
        df_sample = df.sample(n=sample_size, random_state=42)
        
        print(f"Procesando muestra de {sample_size} textos desde {args.input} (de {len(df)} total)")
        resultados = []
        
        for idx, (_, row) in enumerate(df_sample.iterrows()):
            if idx % 20 == 0:
                print(f"Procesando texto {idx+1}/{sample_size}")
            
            texto = row['texto_original']
            pls = generador.generar_pls(texto, config=config)
            resultados.append({
                "id": idx,
                "texto_original": texto,
                "pls_generado": pls
            })
        
        # Guardar resultados
        df_resultados = pd.DataFrame(resultados)
        output_file = os.path.join(args.output, f"pls_generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        df_resultados.to_csv(output_file, index=False)
        print(f"Resultados guardados en: {output_file}")
        
        # Estadísticas
        pls_validos = [r for r in resultados if not r["pls_generado"].startswith("ERROR")]
        print(f"Estadísticas: {len(pls_validos)}/{len(resultados)} PLS generados exitosamente")
    
    elif args.archivo:
        # Modo archivo
        if not os.path.exists(args.archivo):
            print(f"Error: Archivo {args.archivo} no encontrado")
            return
        
        df = pd.read_csv(args.archivo)
        if 'texto' not in df.columns:
            print("Error: El archivo debe tener una columna 'texto'")
            return
        
        resultados = []
        for idx, row in df.iterrows():
            texto = row['texto_original']
            pls = generador.generar_pls(texto, config=config)
            resultados.append({
                "id": idx,
                "texto_original": texto,
                "pls_generado": pls
            })
        
        # Guardar resultados
        df_resultados = pd.DataFrame(resultados)
        output_file = f"data/outputs/pls_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_resultados.to_csv(output_file, index=False)
        print(f"Resultados guardados en: {output_file}")
    
    else:
        print("Error: Debe especificar --texto, --archivo, --input/--output o --zero-shot")


def cargar_configuracion(config_path: str = "params.generator.yaml") -> Dict:
    """Cargar configuración desde YAML con fallbacks sensatos"""
    config_default = {
        "generator": {
            "base_model": "facebook/bart-large-cnn",
            "instructions": "Summarize this medical text in simple, clear language that a patient can understand: ",
            "max_length_tokens": 200,
            "min_length_tokens": 80,
            "num_beams": 4,
            "temperature": 0.8,
            "do_sample": True,
            "language": "en"
        },
        "mlflow": {
            "experiment": "generator_pls",
            "run_name_prefix": "pls_gen"
        }
    }
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            # Merge con defaults
            for key, value in config_default.items():
                if key not in config:
                    config[key] = value
                elif isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if subkey not in config[key]:
                            config[key][subkey] = subvalue
            return config
        except Exception as e:
            print(f"Error cargando configuración: {e}")
            return config_default
    
    return config_default


def configurar_mlflow_experimento(experiment_name: str = "generator_pls"):
    """Configurar experimento MLflow"""
    if not MLFLOW_AVAILABLE:
        raise ImportError("MLflow no está disponible. Instalar con: pip install mlflow")
    
    try:
        # Crear o obtener experimento
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"Experimento '{experiment_name}' creado")
        else:
            experiment_id = experiment.experiment_id
            print(f"Experimento '{experiment_name}' ya existe")
        
        mlflow.set_experiment(experiment_name)
        print(f"Experimento activo: {experiment_name}")
        return experiment_id
        
    except Exception as e:
        print(f"Error configurando experimento: {e}")
        raise


def registrar_experimento_mlflow(config: Dict, resultados: List[Dict], modelo: str):
    """Registrar experimento en MLflow"""
    if not MLFLOW_AVAILABLE:
        print("MLflow no disponible, saltando registro")
        return
    
    try:
        with mlflow.start_run():
            # Loggear configuración
            mlflow.log_params({
                "model": modelo,
                "max_length": config['generator']['max_length_tokens'],
                "min_length": config['generator']['min_length_tokens'],
                "num_beams": config['generator']['num_beams'],
                "temperature": config['generator']['temperature'],
                "do_sample": config['generator']['do_sample']
            })
            
            # Calcular métricas
            pls_validos = [r for r in resultados if not r["pls_generado"].startswith("ERROR")]
            total_textos = len(resultados)
            exitosos = len(pls_validos)
            
            if pls_validos:
                avg_compression = sum(r["compression_ratio"] for r in pls_validos) / len(pls_validos)
                avg_length_original = sum(r["longitud_original"] for r in pls_validos) / len(pls_validos)
                avg_length_pls = sum(r["longitud_pls"] for r in pls_validos) / len(pls_validos)
            else:
                avg_compression = 0
                avg_length_original = 0
                avg_length_pls = 0
            
            # Loggear métricas
            mlflow.log_metrics({
                "success_rate": exitosos / total_textos if total_textos > 0 else 0,
                "avg_compression_ratio": avg_compression,
                "avg_original_length": avg_length_original,
                "avg_pls_length": avg_length_pls,
                "total_texts": total_textos,
                "successful_generations": exitosos
            })
            
            print("Experimento registrado en MLflow")
            
    except Exception as e:
        print(f"Error registrando en MLflow: {e}")


if __name__ == "__main__":
    main()