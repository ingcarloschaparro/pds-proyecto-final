"""
Generador PLS Elevado - Zero-shot y LoRA Fine-tuning
Versión elevada del generador simple con CLI completa, métricas y MLflow
"""
import pandas as pd
import numpy as np
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import os
import json
import yaml
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import warnings

# Nuevas dependencias (añadir a requirements.txt)
typer_available = False
peft_available = False
mlflow_available = False
evaluate_available = False
bert_score_available = False
textstat_available = False
rouge_score_available = False

try:
    import typer
    typer_available = True
except ImportError:
    pass

try:
    from peft import LoraConfig, get_peft_model, PeftModel
    peft_available = True
except ImportError:
    pass

try:
    from accelerate import Accelerator
except ImportError:
    pass

try:
    import evaluate
    evaluate_available = True
except ImportError:
    pass

try:
    import mlflow
    import mlflow.pytorch
    mlflow_available = True
except ImportError:
    pass

try:
    from bert_score import BERTScorer
    bert_score_available = True
except ImportError:
    pass

try:
    from nltk.translate import meteor_score
    import nltk
except ImportError:
    pass

try:
    import textstat
    textstat_available = True
except ImportError:
    pass

try:
    from rouge_score import rouge_scorer
    rouge_score_available = True
except ImportError:
    pass

# Verificar dependencias mínimas para CLI
if not typer_available:
    raise ImportError("typer es requerido para la CLI. Instalar con: pip install typer")

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PLSGeneratorSimple:
    """Generador simplificado de PLS usando BART"""

    def __init__(self):
        self.pipeline = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def cargar_modelo(self):
        """Cargar modelo BART-base"""
        print("Cargando modelo BART-base...")
        try:
            self.pipeline = pipeline(
                "summarization",
                model="facebook/bart-base",
                device=self.device,
                max_length=150,
                min_length=50,
                num_beams=4,
                temperature=0.8,
                do_sample=True,
                early_stopping=True
            )
            print("Modelo cargado exitosamente")
            return True
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            return False

    def generar_pls(self, texto: str, instrucciones: str = None) -> str:
        """Generar PLS para un texto individual"""
        if not texto or pd.isna(texto):
            return ""

        # Instrucciones por defecto
        if instrucciones is None:
            instrucciones = "Explica con lenguaje cotidiano y claro: "

        # Preparar input
        input_text = f"{instrucciones}\n\n{str(texto)[:1000]}"  # Limitar longitud

        try:
            resultados = self.pipeline(input_text)
            return resultados[0]['summary_text'] if resultados else ""
        except Exception as e:
            print(f"Error generando PLS: {e}")
            return ""

def main():
    """Función principal para probar el generador"""
    print("PROBANDO GENERADOR PLS CON BART-BASE")
    print("=" * 50)

    # Inicializar generador
    generador = PLSGeneratorSimple()

    if not generador.cargar_modelo():
        return

    # Textos de prueba
    textos_prueba = [
        "The study evaluated the effects of metformin on glycemic control in patients with type 2 diabetes mellitus. Participants received either metformin 500mg twice daily or placebo for 12 weeks. The primary outcome was change in HbA1c levels from baseline.",

        "Randomized controlled trial comparing laparoscopic vs open cholecystectomy for symptomatic cholelithiasis. The intervention group underwent laparoscopic procedure while control group received open surgery. Primary endpoints included operative time, postoperative complications, and length of hospital stay.",

        "Clinical trial investigating the efficacy of atorvastatin 20mg daily versus placebo in reducing cardiovascular events in patients with hypercholesterolemia. The study enrolled 500 participants and followed them for 2 years, measuring LDL cholesterol levels and incidence of myocardial infarction."
    ]

    # Generar PLS
    print("\n GENERANDO PLS...")
    resultados = []

    for i, texto in enumerate(textos_prueba, 1):
        print(f"\n--- Texto {i} ---")
        print(f"Original: {texto[:100]}...")

        pls = generador.generar_pls(texto)
        print(f"PLS generado: {pls}")

        resultados.append({
            'id': i,
            'texto_original': texto,
            'pls_generado': pls,
            'longitud_original': len(texto),
            'longitud_pls': len(pls)
        })

    # Guardar resultados
    os.makedirs("data/outputs", exist_ok=True)
    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv("data/outputs/pls_prueba_bart.csv", index=False)

    print("\nPrueba completada!")
    print(f"Resultados guardados en: data/outputs/pls_prueba_bart.csv")
    # Estadísticas
    pls_validos = [r for r in resultados if r['pls_generado']]
    print(f"Estadísticas: {len(pls_validos)}/{len(resultados)} PLS generados exitosamente")

# ===== FUNCIONES UTILITARIAS NUEVAS =====

def load_config(config_path: str) -> Dict[str, Any]:
    """Cargar configuración desde YAML con fallbacks sensatos"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Fallbacks para campos faltantes
    defaults = {
        'data': {
            'path': 'data/processed/dataset_clean_v1.csv',
            'text_source_col': 'texto_original',
            'text_pls_col': 'resumen',
            'source_col': 'source',
            'doc_id_col': 'doc_id',
            'val_size': 0.2,
            'test_size': 0.1,
            'random_state': 42
        },
        'generator': {
            'base_model': 'facebook/bart-large-cnn',
            'strategy': 'zero-shot',
            'max_length_tokens': 200,
            'min_length_tokens': 80,
            'num_beams': 4,
            'temperature': 0.8,
            'max_input_length': 1024,
            'truncation_side': 'right',
            'eval_subset_size': 500
        },
        'mlflow': {
            'experiment': 'generator_pls',
            'run_name_prefix': 'pls_gen'
        },
        'artifacts': {
            'base_dir': 'artifacts/generator'
        }
    }

    # Merge configs
    for section, values in defaults.items():
        if section not in config:
            config[section] = values
        else:
            for key, value in values.items():
                if key not in config[section]:
                    config[section][key] = value

    return config

def setup_mlflow(config: Dict[str, Any], strategy: str, base_model: str) -> str:
    """Configurar MLflow y retornar run_id"""
    if not mlflow_available:
        raise ImportError("MLflow no está disponible. Instalar con: pip install mlflow")
    
    mlflow.set_experiment(config['mlflow']['experiment'])

    # Crear run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    base_short = base_model.split('/')[-1]
    run_name = f"{base_short}_{strategy}_global_{timestamp}"

    mlflow.start_run(run_name=run_name)

    # Tags
    tags = config['mlflow'].get('tags', {})
    tags.update({
        'strategy': strategy,
        'base_model': base_model,
        'lora': 'true' if strategy == 'lora' else 'false'
    })

    mlflow.set_tags(tags)

    # Loggear configuración
    mlflow.log_params({
        'data_path': config['data']['path'],
        'text_source_col': config['data']['text_source_col'],
        'text_pls_col': config['data']['text_pls_col'],
        'source_col': config['data']['source_col'],
        'base_model': base_model,
        'strategy': strategy,
        'max_length_tokens': config['generator']['max_length_tokens'],
        'min_length_tokens': config['generator']['min_length_tokens'],
        'num_beams': config['generator']['num_beams'],
        'eval_subset_size': config['generator']['eval_subset_size']
    })

    run_id = mlflow.active_run().info.run_id
    logger.info(f"MLflow run iniciado: {run_name} (ID: {run_id})")
    return run_id

def load_dataset(config: Dict[str, Any]) -> pd.DataFrame:
    """Cargar dataset con validaciones y manejo de pares texto-resumen"""
    path = config['data']['path']
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset no encontrado: {path}")

    df = pd.read_csv(path, low_memory=False)

    # Validar columnas requeridas
    required_cols = [config['data']['text_source_col'], config['data']['text_pls_col']]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columnas faltantes en dataset: {missing_cols}")

    # Crear pares texto-resumen
    df_pares = create_text_summary_pairs(df, config)
    
    # Filtrar solo anotados si está configurado
    if config.get('evaluation', {}).get('eval_only_annotated', False):
        mask = df_pares[config['data']['text_pls_col']].notna()
        df_pares = df_pares[mask].copy()
        logger.info(f"Filtrando solo filas con ground truth: {len(df_pares)} filas")

    # Subset para evaluación rápida
    eval_size = config['generator']['eval_subset_size']
    if eval_size and len(df_pares) > eval_size:
        df_pares = df_pares.sample(n=eval_size, random_state=config['data']['random_state'])
        logger.info(f"Usando subset de {eval_size} filas para evaluación rápida")

    logger.info(f"Dataset cargado: {len(df_pares)} filas válidas")
    return df_pares

def create_text_summary_pairs(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Crear pares texto-resumen a partir de filas separadas"""
    text_col = config['data']['text_source_col']
    summary_col = config['data']['text_pls_col']
    doc_id_col = config['data']['doc_id_col']
    
    # Extraer doc_id base (sin #L número)
    df['doc_id_base'] = df[doc_id_col].str.replace(r'\.txt#L\d+', '', regex=True)
    
    # Separar textos y resúmenes
    df_textos = df[df[text_col].notna()].copy()
    df_resumenes = df[df[summary_col].notna()].copy()
    
    # Encontrar doc_ids que tienen tanto texto como resumen
    doc_ids_texto = set(df_textos['doc_id_base'])
    doc_ids_resumen = set(df_resumenes['doc_id_base'])
    doc_ids_comunes = doc_ids_texto & doc_ids_resumen
    
    logger.info(f"Doc_ids con texto: {len(doc_ids_texto)}")
    logger.info(f"Doc_ids con resumen: {len(doc_ids_resumen)}")
    logger.info(f"Doc_ids con ambos (pares válidos): {len(doc_ids_comunes)}")
    
    # Crear pares
    pares = []
    for doc_id in doc_ids_comunes:
        # Tomar el primer texto y el primer resumen para este doc_id
        texto_row = df_textos[df_textos['doc_id_base'] == doc_id].iloc[0]
        resumen_row = df_resumenes[df_resumenes['doc_id_base'] == doc_id].iloc[0]
        
        # Crear fila combinada
        fila_combinada = {
            doc_id_col: doc_id,
            text_col: texto_row[text_col],
            summary_col: resumen_row[summary_col],
            'source': texto_row.get('source', 'unknown'),
            'doc_id_original': doc_id
        }
        pares.append(fila_combinada)
    
    df_pares = pd.DataFrame(pares)
    logger.info(f"Pares creados: {len(df_pares)} filas")
    
    return df_pares

def calculate_metrics(references: List[str], predictions: List[str], sources: List[str]) -> Dict[str, Any]:
    """Calcular todas las métricas requeridas"""
    metrics = {}

    # Filtrar pares válidos (con ground truth)
    valid_pairs = [(ref, pred) for ref, pred in zip(references, predictions) if ref and pred]
    valid_refs = [ref for ref, pred in valid_pairs]
    valid_preds = [pred for ref, pred in valid_pairs]
    
    logger.info(f"Calculando métricas para {len(valid_pairs)} pares válidos de {len(references)} total")

    # ROUGE-L
    if rouge_score_available and valid_pairs:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_scores = []
        for ref, pred in valid_pairs:
            try:
                score = scorer.score(ref, pred)['rougeL'].fmeasure
                rouge_scores.append(score)
            except:
                continue

        metrics['rouge_l'] = np.mean(rouge_scores) if rouge_scores else np.nan
    else:
        metrics['rouge_l'] = np.nan if not valid_pairs else 0.0

    # METEOR
    if valid_pairs:
        try:
            meteor_scores = []
            for ref, pred in valid_pairs:
                try:
                    score = meteor_score.meteor_score([ref.split()], pred.split())
                    meteor_scores.append(score)
                except:
                    continue

            metrics['meteor'] = np.mean(meteor_scores) if meteor_scores else np.nan
        except:
            metrics['meteor'] = np.nan
    else:
        metrics['meteor'] = np.nan

    # BERTScore
    if bert_score_available and valid_pairs:
        try:
            scorer = BERTScorer(model_type="microsoft/DistilBERT-base-uncased", batch_size=16)
            P, R, F1 = scorer.score(valid_preds, valid_refs)
            metrics['bertscore_f1'] = F1.mean().item()
        except Exception as e:
            logger.warning(f"Error calculando BERTScore: {e}")
            metrics['bertscore_f1'] = np.nan
    else:
        metrics['bertscore_f1'] = np.nan if not valid_pairs else 0.0

    # FKGL - siempre calcular (no requiere ground truth)
    if textstat_available:
        fkgl_src = []
        fkgl_pred = []
        for ref, pred in zip(references, predictions):
            if ref:
                try:
                    fkgl_src.append(textstat.flesch_kincaid_grade(ref))
                except:
                    continue
            if pred:
                try:
                    fkgl_pred.append(textstat.flesch_kincaid_grade(pred))
                except:
                    continue

        metrics['fkgl_src'] = np.mean(fkgl_src) if fkgl_src else np.nan
        metrics['fkgl_pred'] = np.mean(fkgl_pred) if fkgl_pred else np.nan
    else:
        metrics['fkgl_src'] = np.nan
        metrics['fkgl_pred'] = np.nan

    # Compression ratio - siempre calcular
    ratios = []
    for ref, pred in zip(references, predictions):
        if ref and pred:
            src_words = len(ref.split())
            pred_words = len(pred.split())
            if src_words > 0:
                ratios.append(pred_words / src_words)

    metrics['compression_ratio'] = np.mean(ratios) if ratios else np.nan

    # Análisis de longitud - siempre calcular
    pred_lengths = [len(pred.split()) for pred in predictions if pred]
    metrics['length_mean'] = np.mean(pred_lengths) if pred_lengths else 0.0
    metrics['length_std'] = np.std(pred_lengths) if pred_lengths else 0.0

    # Porcentaje en rango objetivo (120-250 palabras)
    target_range = [p for p in pred_lengths if 120 <= p <= 250]
    metrics['pct_length_in_range'] = len(target_range) / len(pred_lengths) if pred_lengths else 0.0

    # Métricas por fuente
    metrics_by_source = {}
    for src in set(sources):
        if not src:
            continue
        src_mask = [s == src for s in sources]
        src_refs = [r for r, m in zip(references, src_mask) if m]
        src_preds = [p for p, m in zip(predictions, src_mask) if m]

        if src_refs and src_preds:
            src_metrics = calculate_metrics(src_refs, src_preds, [src] * len(src_refs))
            metrics_by_source[src] = {
                'count': len(src_refs),
                'rouge_l': src_metrics['rouge_l'],
                'meteor': src_metrics['meteor'],
                'bertscore_f1': src_metrics['bertscore_f1'],
                'fkgl_pred': src_metrics['fkgl_pred'],
                'compression_ratio': src_metrics['compression_ratio']
            }

    return metrics, metrics_by_source

def calculate_individual_metrics(references: List[str], predictions: List[str]) -> Dict[str, List[float]]:
    """Calcular métricas individuales para cada par texto-resumen"""
    individual_metrics = {
        'rouge_l': [],
        'meteor': [],
        'bertscore_f1': [],
        'fkgl_src': [],
        'fkgl_pred': [],
        'compression_ratio': []
    }
    
    for ref, pred in zip(references, predictions):
        # ROUGE-L
        if rouge_score_available and ref and pred:
            try:
                scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
                score = scorer.score(ref, pred)['rougeL'].fmeasure
                individual_metrics['rouge_l'].append(score)
            except:
                individual_metrics['rouge_l'].append(np.nan)
        else:
            individual_metrics['rouge_l'].append(np.nan)
        
        # METEOR
        if ref and pred:
            try:
                score = meteor_score.meteor_score([ref.split()], pred.split())
                individual_metrics['meteor'].append(score)
            except:
                individual_metrics['meteor'].append(np.nan)
        else:
            individual_metrics['meteor'].append(np.nan)
        
        # BERTScore (placeholder - se calcula por batch)
        individual_metrics['bertscore_f1'].append(np.nan)
        
        # FKGL
        if textstat_available:
            if ref:
                try:
                    fkgl_src = textstat.flesch_kincaid_grade(ref)
                    individual_metrics['fkgl_src'].append(fkgl_src)
                except:
                    individual_metrics['fkgl_src'].append(np.nan)
            else:
                individual_metrics['fkgl_src'].append(np.nan)
            
            if pred:
                try:
                    fkgl_pred = textstat.flesch_kincaid_grade(pred)
                    individual_metrics['fkgl_pred'].append(fkgl_pred)
                except:
                    individual_metrics['fkgl_pred'].append(np.nan)
            else:
                individual_metrics['fkgl_pred'].append(np.nan)
        else:
            individual_metrics['fkgl_src'].append(np.nan)
            individual_metrics['fkgl_pred'].append(np.nan)
        
        # Compression ratio
        if ref and pred:
            src_words = len(ref.split())
            pred_words = len(pred.split())
            if src_words > 0:
                ratio = pred_words / src_words
                individual_metrics['compression_ratio'].append(ratio)
            else:
                individual_metrics['compression_ratio'].append(np.nan)
        else:
            individual_metrics['compression_ratio'].append(np.nan)
    
    return individual_metrics

def save_artifacts(config: Dict[str, Any], df_eval: pd.DataFrame, metrics_global: Dict,
                  metrics_by_source: Dict, run_id: str):
    """Guardar todos los artifacts organizadamente"""
    base_dir = Path(config['artifacts']['base_dir']) / run_id
    base_dir.mkdir(parents=True, exist_ok=True)

    # Guardar configuración usada
    with open(base_dir / "used_config.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Guardar métricas globales
    with open(base_dir / "metrics_global.json", 'w') as f:
        json.dump(metrics_global, f, indent=2)

    # Guardar métricas por fuente
    if metrics_by_source:
        pd.DataFrame.from_dict(metrics_by_source, orient='index').to_csv(
            base_dir / "metrics_by_source.csv", index_label='source')

    # Guardar samples de evaluación
    df_eval.to_csv(base_dir / "samples_eval.csv", index=False)

    # Loggear en MLflow si está disponible
    if mlflow_available:
        mlflow.log_artifact(str(base_dir / "used_config.yaml"))
        mlflow.log_artifact(str(base_dir / "metrics_global.json"))
        mlflow.log_artifact(str(base_dir / "samples_eval.csv"))
        if metrics_by_source:
            mlflow.log_artifact(str(base_dir / "metrics_by_source.csv"))

        # Loggear métricas en MLflow
        for key, value in metrics_global.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)

    logger.info(f"Artifacts guardados en: {base_dir}")
    return str(base_dir)

# ===== CLI CON TYPER =====

app = typer.Typer(help="Generador PLS - Zero-shot y LoRA Fine-tuning")

@app.command()
def zero_shot(
    config_path: str = typer.Option("params.generator.yaml", help="Ruta al archivo de configuración"),
    output_dir: Optional[str] = typer.Option(None, help="Directorio de salida (opcional)")
):
    """Generar PLS usando zero-shot con BART-large-cnn"""
    logger.info("=== MODO ZERO-SHOT ===")

    # Cargar configuración
    config = load_config(config_path)
    if output_dir:
        config['artifacts']['base_dir'] = output_dir

    base_model = config['generator']['base_model']
    logger.info(f"Usando modelo: {base_model}")

    # Setup MLflow
    run_id = setup_mlflow(config, 'zero-shot', base_model)

    try:
        # Cargar dataset
        df = load_dataset(config)

        # Inicializar generador (usando clase existente)
        generador = PLSGeneratorSimple()

        # Modificar para usar modelo configurable
        generador.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Cargar modelo configurable
        logger.info(f"Cargando modelo {base_model}...")
        
        # Configurar parámetros de generación
        gen_params = {
            "model": base_model,
            "device": generador.device,
            "num_beams": config['generator']['num_beams'],
            "early_stopping": True
        }
        
        # Usar min_new_tokens/max_new_tokens si están disponibles, sino fallback
        if 'min_new_tokens' in config['generator'] and 'max_new_tokens' in config['generator']:
            gen_params.update({
                "min_new_tokens": config['generator']['min_new_tokens'],
                "max_new_tokens": config['generator']['max_new_tokens']
            })
            logger.info(f"Usando min_new_tokens={config['generator']['min_new_tokens']}, max_new_tokens={config['generator']['max_new_tokens']}")
        else:
            gen_params.update({
                "min_length": config['generator']['min_length_tokens'],
                "max_length": config['generator']['max_length_tokens']
            })
            logger.info(f"Usando min_length={config['generator']['min_length_tokens']}, max_length={config['generator']['max_length_tokens']}")
        
        # Configurar decodificación
        if config['generator'].get('do_sample', False):
            gen_params.update({
                "do_sample": True,
                "temperature": config['generator']['temperature']
            })
        else:
            gen_params.update({
                "do_sample": False
            })
        
        generador.pipeline = pipeline("summarization", **gen_params)

        # Generar PLS
        logger.info("Generando PLS...")
        predictions = []
        sources = []
        doc_ids = []

        for idx, row in df.iterrows():
            texto = str(row[config['data']['text_source_col']])
            source = str(row.get(config['data']['source_col'], 'unknown'))
            doc_id = str(row.get(config['data']['doc_id_col'], f'row_{idx}'))

            # Control de longitud de input
            max_input = config['generator']['max_input_length']
            if len(texto) > max_input:
                texto = texto[:max_input]
                logger.warning(f"Texto truncado para {doc_id}: {len(texto)} chars")

            pls = generador.generar_pls(texto)
            predictions.append(pls)
            sources.append(source)
            doc_ids.append(doc_id)

        # Calcular métricas
        logger.info("Calculando métricas...")
        references = df[config['data']['text_pls_col']].tolist()
        metrics_global, metrics_by_source = calculate_metrics(references, predictions, sources)

        # Preparar dataframe de evaluación con métricas individuales
        df_eval = df.copy()
        df_eval['pls_predicted'] = predictions
        
        # Calcular métricas individuales
        individual_metrics = calculate_individual_metrics(references, predictions)
        df_eval['rouge_l'] = individual_metrics['rouge_l']
        df_eval['meteor'] = individual_metrics['meteor']
        df_eval['bertscore_f1'] = individual_metrics['bertscore_f1']
        df_eval['fkgl_src'] = individual_metrics['fkgl_src']
        df_eval['fkgl_pred'] = individual_metrics['fkgl_pred']
        df_eval['compression_ratio'] = individual_metrics['compression_ratio']

        # Guardar artifacts
        artifacts_dir = save_artifacts(config, df_eval, metrics_global, metrics_by_source, run_id)

        # Resultado final
        print("\n" + "="*50)
        print("ZERO-SHOT COMPLETADO")
        print("="*50)
        print(f"Run ID: {run_id}")
        print(f"Artifacts: {artifacts_dir}")
        print(f"Filas generadas: {len(predictions)}")
        print(f"Filas evaluadas con GT: {len([r for r in references if r])}")
        print(f"ROUGE-L: {metrics_global['rouge_l']:.3f}")
        print(f"METEOR: {metrics_global['meteor']:.3f}")
        print(f"BERTScore F1: {metrics_global['bertscore_f1']:.3f}")
        print(f"FKGL Pred: {metrics_global['fkgl_pred']:.2f}")
        print(f"Compression Ratio: {metrics_global['compression_ratio']:.3f}")
        print(f"Length in Range: {metrics_global['pct_length_in_range']:.1%}")
        
        # Mensaje final de validación
        print(f"\nOK: artifacts escritos en {artifacts_dir}")
        print(f"OK: MLflow run_id={run_id}")

    except Exception as e:
        logger.error(f"Error en zero-shot: {e}")
        if mlflow_available:
            mlflow.end_run(status="FAILED")
        raise
    finally:
        if mlflow_available:
            mlflow.end_run()

@app.command("finetune-lora")
def finetune_lora(
    config_path: str = typer.Option("params.generator.yaml", help="Ruta al archivo de configuración"),
    base_model: str = typer.Option("bart-base", help="Modelo base: bart-base o t5-base"),
    output_dir: Optional[str] = typer.Option(None, help="Directorio de salida (opcional)")
):
    """Fine-tuning LoRA del generador PLS"""
    logger.info("=== MODO FINETUNE LoRA ===")

    # Cargar configuración
    config = load_config(config_path)
    if output_dir:
        config['artifacts']['base_dir'] = output_dir

    # Configurar modelo base
    if base_model == "bart-base":
        model_name = "facebook/bart-base"
    elif base_model == "t5-base":
        model_name = "t5-base"
    else:
        raise ValueError(f"Modelo base no soportado: {base_model}")

    logger.info(f"Fine-tuning LoRA en {model_name}")

    # Setup MLflow
    run_id = setup_mlflow(config, 'lora', model_name)

    try:
        # Placeholder - implementación completa próximamente
        logger.info("Implementación LoRA próximamente...")

        # Por ahora, solo marcar como completado
        print("\n" + "="*50)
        print("FINETUNE LORA (PLACEHOLDER)")
        print("="*50)
        print(f"Run ID: {run_id}")
        print("Implementación completa próximamente")

    except Exception as e:
        logger.error(f"Error en finetune-lora: {e}")
        if mlflow_available:
            mlflow.end_run(status="FAILED")
        raise
    finally:
        if mlflow_available:
            mlflow.end_run()

@app.command()
def evaluate(
    config_path: str = typer.Option("params.generator.yaml", help="Ruta al archivo de configuración"),
    model_path: str = typer.Option(..., help="Ruta al modelo guardado"),
    output_dir: Optional[str] = typer.Option(None, help="Directorio de salida (opcional)")
):
    """Evaluar modelo guardado"""
    logger.info("=== MODO EVALUATE ===")

    # Cargar configuración
    config = load_config(config_path)
    if output_dir:
        config['artifacts']['base_dir'] = output_dir

    logger.info(f"Evaluando modelo: {model_path}")

    # Setup MLflow
    run_id = setup_mlflow(config, 'evaluate', os.path.basename(model_path))

    try:
        # Placeholder - implementación completa próximamente
        logger.info("Implementación evaluate próximamente...")

        print("\n" + "="*50)
        print("EVALUATE (PLACEHOLDER)")
        print("="*50)
        print(f"Run ID: {run_id}")
        print("Implementación completa próximamente")

    except Exception as e:
        logger.error(f"Error en evaluate: {e}")
        if mlflow_available:
            mlflow.end_run(status="FAILED")
        raise
    finally:
        if mlflow_available:
            mlflow.end_run()

# ===== FUNCIÓN MAIN PARA COMPATIBILIDAD =====

def show_validation_checklist():
    """Mostrar checklist de validación final"""
    print("\n" + "="*60)
    print("VALIDACIÓN FINAL - GENERADOR PLS ELEVADO")
    print("="*60)

    # Verificar archivos
    print("\n📁 ARCHIVOS:")
    files_status = {
        "params.generator.yaml": os.path.exists("params.generator.yaml"),
        "src/models/generate_pls_simple.py": os.path.exists("src/models/generate_pls_simple.py"),
        "docs/analysis/generate_pls_simple_audit.md": os.path.exists("docs/analysis/generate_pls_simple_audit.md"),
        "docs/analysis/generate_pls_dependencies.md": os.path.exists("docs/analysis/generate_pls_dependencies.md"),
        "Makefile": os.path.exists("Makefile"),
        "data/processed/dataset_clean_v1.csv": os.path.exists("data/processed/dataset_clean_v1.csv")
    }

    for file, exists in files_status.items():
        status = "✅" if exists else "❌"
        print(f"  {status} {file}")

    # Verificar dependencias
    print("\n🐍 DEPENDENCIAS:")
    deps_status = {}
    try:
        import pandas, numpy, torch, transformers
        deps_status["Core (pandas, numpy, torch, transformers)"] = True
    except ImportError:
        deps_status["Core (pandas, numpy, torch, transformers)"] = False

    try:
        import typer, peft, accelerate, evaluate, bert_score, mlflow, textstat, nltk
        deps_status["Generator (typer, peft, accelerate, evaluate, bert-score, mlflow, textstat, nltk)"] = True
    except ImportError as e:
        deps_status["Generator (typer, peft, accelerate, evaluate, bert-score, mlflow, textstat, nltk)"] = False
        print(f"    ⚠️  Faltantes: {e}")

    for dep, available in deps_status.items():
        status = "✅" if available else "❌"
        print(f"  {status} {dep}")

    # Verificar configuración
    print("\n⚙️  CONFIGURACIÓN:")
    if os.path.exists("params.generator.yaml"):
        try:
            config = load_config("params.generator.yaml")
            print("  ✅ params.generator.yaml - OK")
            print(f"     📊 Dataset: {config['data']['path']}")
            print(f"     🤖 Modelo: {config['generator']['base_model']}")
            print(f"     📈 Eval subset: {config['generator']['eval_subset_size']}")
            print(f"     🔬 Experimento MLflow: {config['mlflow']['experiment']}")
        except Exception as e:
            print(f"  ❌ params.generator.yaml - Error: {e}")
    else:
        print("  ❌ params.generator.yaml - No encontrado")

    # Verificar dataset
    print("\n📊 DATASET:")
    if os.path.exists("data/processed/dataset_clean_v1.csv"):
        try:
            df = pd.read_csv("data/processed/dataset_clean_v1.csv")
            print("  ✅ Dataset cargado")
            print(f"     📏 Filas: {len(df)}")
            print(f"     📝 Columnas: {list(df.columns)}")

            # Verificar columnas requeridas
            config = load_config("params.generator.yaml")
            required_cols = [config['data']['text_source_col'], config['data']['text_pls_col']]
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                print(f"  ⚠️  Columnas faltantes: {missing}")
            else:
                print("  ✅ Columnas requeridas presentes")

        except Exception as e:
            print(f"  ❌ Error cargando dataset: {e}")
    else:
        print("  ❌ Dataset no encontrado")

    # Comandos disponibles
    print("\n🚀 COMANDOS DISPONIBLES:")
    commands = [
        "python src/models/generate_pls_simple.py zero-shot",
        "python src/models/generate_pls_simple.py finetune-lora --base bart-base",
        "python src/models/generate_pls_simple.py evaluate --model-path <path>",
        "make gen-zero-shot",
        "make gen-lora-train BASE_MODEL=bart-base",
        "make gen-eval MODEL_PATH=<path>",
        "make help"
    ]

    for cmd in commands:
        print(f"  💻 {cmd}")

    # Información de salida
    print("\n📂 DIRECTORIOS DE SALIDA:")
    print("  📊 artifacts/generator/<run_id>/")
    print("     ├── metrics_global.json")
    print("     ├── metrics_by_source.csv")
    print("     ├── samples_eval.csv")
    print("     ├── used_config.yaml")
    print("     └── [modelo guardado]")
    print("  🔬 mlruns/generator_pls/")
    print("     └── [experimentos MLflow]")
    print()

    # Próximos pasos
    print("🎯 PRÓXIMOS PASOS:")
    steps = [
        "1. Instalar dependencias: pip install -r requirements.txt",
        "2. Verificar instalación: make check-install",
        "3. Probar zero-shot: make gen-zero-shot",
        "4. Revisar artifacts generados",
        "5. Revisar experimento en MLflow UI"
    ]

    for step in steps:
        print(f"   {step}")

    print("\n" + "="*60)
    print("✅ VALIDACIÓN COMPLETADA")
    print("="*60)

@app.command()
def validate():
    """Mostrar checklist de validación del sistema"""
    show_validation_checklist()

if __name__ == "__main__":
    # Verificar si hay argumentos CLI
    import sys
    if len(sys.argv) > 1:
        # Usar CLI de Typer
        app()
    else:
        # Usar función original para compatibilidad
        main()
