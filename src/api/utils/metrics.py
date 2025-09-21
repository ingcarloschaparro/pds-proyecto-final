import time
import textstat
import numpy as np
from typing import Dict, List, Any, Optional
from loguru import logger

class PLSMetricsCalculator:
    """Calculadora de métricas específicas para modelos PLS"""
    
    def __init__(self):
        """Inicializar calculadora de métricas"""
        self.metrics_history = []
        self.baseline_metrics = {
            "compression_ratio": 0.292,
            "fkgl_score": 12.2,
            "flesch_score": 39.0,
            "inference_time": 3.64,
            "rouge_1": 0.68,
            "rouge_2": 0.52,
            "rouge_l": 0.61,
            "bleu_score": 0.58,
            "readability_score": 8.5,
            "quality_score": 9.2
        }
    
    def calculate_readability_metrics(self, text: str) -> Dict[str, float]:
        """
        Calcular métricas de legibilidad de un texto
        """
        try:
            if not text or len(text.strip()) == 0:
                return {
                    "flesch_score": 0.0,
                    "fkgl_score": 0.0,
                    "word_count": 0,
                    "sentence_count": 0,
                    "avg_word_length": 0.0,
                    "complex_words": 0
                }
            
            # Métricas básicas
            words = len(text.split())
            sentences = len([s for s in text.split('.') if s.strip()])
            
            # Flesch Reading Ease Score (0-100, mayor es mejor)
            flesch_score = textstat.flesch_reading_ease(text)
            
            # Flesch-Kincaid Grade Level (menor es mejor)
            fkgl_score = textstat.flesch_kincaid_grade(text)
            
            # Longitud promedio de palabras
            avg_word_length = sum(len(word) for word in text.split()) / words if words > 0 else 0
            
            # Palabras complejas (más de 6 caracteres)
            complex_words = len([w for w in text.split() if len(w) > 6])
            
            return {
                "flesch_score": round(flesch_score, 2),
                "fkgl_score": round(fkgl_score, 2),
                "word_count": words,
                "sentence_count": sentences,
                "avg_word_length": round(avg_word_length, 2),
                "complex_words": complex_words,
                "complexity_ratio": round(complex_words / words * 100, 2) if words > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculando métricas de legibilidad: {e}")
            return {
                "flesch_score": 0.0,
                "fkgl_score": 0.0,
                "word_count": 0,
                "sentence_count": 0,
                "avg_word_length": 0.0,
                "complex_words": 0,
                "complexity_ratio": 0.0
            }
    
    def calculate_compression_metrics(self, original_text: str, summary_text: str) -> Dict[str, float]:
        """
        Calcular métricas de compresión entre texto original y resumen
        """
        try:
            if not original_text or not summary_text:
                return {
                    "compression_ratio": 0.0,
                    "length_reduction": 0.0,
                    "word_reduction": 0.0,
                    "sentence_reduction": 0.0
                }
            
            # Longitudes
            orig_length = len(original_text)
            summ_length = len(summary_text)
            
            # Conteos de palabras
            orig_words = len(original_text.split())
            summ_words = len(summary_text.split())
            
            # Conteos de oraciones
            orig_sentences = len([s for s in original_text.split('.') if s.strip()])
            summ_sentences = len([s for s in summary_text.split('.') if s.strip()])
            
            # Ratios de compresión
            compression_ratio = summ_length / orig_length if orig_length > 0 else 0
            length_reduction = (orig_length - summ_length) / orig_length if orig_length > 0 else 0
            word_reduction = (orig_words - summ_words) / orig_words if orig_words > 0 else 0
            sentence_reduction = (orig_sentences - summ_sentences) / orig_sentences if orig_sentences > 0 else 0
            
            return {
                "compression_ratio": round(compression_ratio, 3),
                "length_reduction": round(length_reduction, 3),
                "word_reduction": round(word_reduction, 3),
                "sentence_reduction": round(sentence_reduction, 3),
                "original_length": orig_length,
                "summary_length": summ_length,
                "original_words": orig_words,
                "summary_words": summ_words,
                "original_sentences": orig_sentences,
                "summary_sentences": summ_sentences
            }
            
        except Exception as e:
            logger.error(f"Error calculando métricas de compresión: {e}")
            return {
                "compression_ratio": 0.0,
                "length_reduction": 0.0,
                "word_reduction": 0.0,
                "sentence_reduction": 0.0
            }
    
    def calculate_quality_score(self, original_text: str, summary_text: str) -> Dict[str, float]:
        """
        Calcular score de calidad compuesto del resumen
        """
        try:
            # Métricas de legibilidad del resumen
            summary_readability = self.calculate_readability_metrics(summary_text)
            
            # Métricas de compresión
            compression_metrics = self.calculate_compression_metrics(original_text, summary_text)
            
            # Score de legibilidad (0-10)
            readability_score = min(10, max(0, summary_readability["flesch_score"] / 10))
            
            # Score de compresión (0-10, penalizar si es muy corto o muy largo)
            compression_ratio = compression_metrics["compression_ratio"]
            if compression_ratio < 0.1:  # Muy corto
                compression_score = compression_ratio * 5
            elif compression_ratio > 2.0:  # Muy largo
                compression_score = max(0, 10 - (compression_ratio - 1) * 5)
            else:  # Rango ideal
                compression_score = 10 - abs(compression_ratio - 0.3) * 10
            
            # Score de complejidad (0-10, menor complejidad es mejor)
            complexity_score = max(0, 10 - summary_readability["fkgl_score"] / 2)
            
            # Score compuesto (promedio ponderado)
            quality_score = (
                readability_score * 0.4 +
                compression_score * 0.3 +
                complexity_score * 0.3
            )
            
            return {
                "quality_score": round(quality_score, 2),
                "readability_score": round(readability_score, 2),
                "compression_score": round(compression_score, 2),
                "complexity_score": round(complexity_score, 2),
                "overall_grade": self._get_quality_grade(quality_score)
            }
            
        except Exception as e:
            logger.error(f"Error calculando score de calidad: {e}")
            return {
                "quality_score": 0.0,
                "readability_score": 0.0,
                "compression_score": 0.0,
                "complexity_score": 0.0,
                "overall_grade": "F"
            }
    
    def _get_quality_grade(self, score: float) -> str:
        """Convertir score numérico a calificación"""
        if score >= 9.0:
            return "A+"
        elif score >= 8.0:
            return "A"
        elif score >= 7.0:
            return "B+"
        elif score >= 6.0:
            return "B"
        elif score >= 5.0:
            return "C+"
        elif score >= 4.0:
            return "C"
        elif score >= 3.0:
            return "D"
        else:
            return "F"
    
    def calculate_comprehensive_metrics(self, original_text: str, summary_text: str, 
                                      processing_time: float = 0.0) -> Dict[str, Any]:
        """
        Calcular métricas completas del resumen
        """
        try:
            # Métricas de legibilidad
            readability_metrics = self.calculate_readability_metrics(summary_text)
            
            # Métricas de compresión
            compression_metrics = self.calculate_compression_metrics(original_text, summary_text)
            
            # Score de calidad
            quality_metrics = self.calculate_quality_score(original_text, summary_text)
            
            # Métricas de rendimiento
            performance_metrics = {
                "processing_time": round(processing_time, 3),
                "words_per_second": round(len(summary_text.split()) / processing_time, 2) if processing_time > 0 else 0,
                "characters_per_second": round(len(summary_text) / processing_time, 2) if processing_time > 0 else 0
            }
            
            # Métricas comparativas con baseline
            baseline_comparison = self._compare_with_baseline({
                "compression_ratio": compression_metrics["compression_ratio"],
                "fkgl_score": readability_metrics["fkgl_score"],
                "flesch_score": readability_metrics["flesch_score"],
                "processing_time": processing_time
            })
            
            # Compilar todas las métricas
            comprehensive_metrics = {
                "readability": readability_metrics,
                "compression": compression_metrics,
                "quality": quality_metrics,
                "performance": performance_metrics,
                "baseline_comparison": baseline_comparison,
                "timestamp": time.time(),
                "model_info": {
                    "model_name": "t5-base",
                    "version": "1.0.0"
                }
            }
            
            # Guardar en historial
            self.metrics_history.append(comprehensive_metrics)
            
            return comprehensive_metrics
            
        except Exception as e:
            logger.error(f"Error calculando métricas completas: {e}")
            return {
                "error": str(e),
                "timestamp": time.time()
            }
    
    def _compare_with_baseline(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Comparar métricas actuales con baseline del modelo
        """
        comparison = {}
        
        for metric, current_value in metrics.items():
            if metric in self.baseline_metrics:
                baseline_value = self.baseline_metrics[metric]
                
                # Calcular diferencia porcentual
                if baseline_value != 0:
                    diff_percent = ((current_value - baseline_value) / baseline_value) * 100
                else:
                    diff_percent = 0
                
                # Determinar si es mejor, peor o igual
                if metric in ["fkgl_score", "processing_time"]:  # Menor es mejor
                    if current_value < baseline_value:
                        status = "better"
                    elif current_value > baseline_value:
                        status = "worse"
                    else:
                        status = "equal"
                else:  # Mayor es mejor
                    if current_value > baseline_value:
                        status = "better"
                    elif current_value < baseline_value:
                        status = "worse"
                    else:
                        status = "equal"
                
                comparison[metric] = {
                    "current": current_value,
                    "baseline": baseline_value,
                    "difference_percent": round(diff_percent, 2),
                    "status": status
                }
        
        return comparison
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Obtener resumen de métricas históricas
        """
        if not self.metrics_history:
            return {"message": "No hay métricas históricas disponibles"}
        
        # Calcular promedios
        total_metrics = len(self.metrics_history)
        
        avg_compression = np.mean([m["compression"]["compression_ratio"] for m in self.metrics_history])
        avg_fkgl = np.mean([m["readability"]["fkgl_score"] for m in self.metrics_history])
        avg_flesch = np.mean([m["readability"]["flesch_score"] for m in self.metrics_history])
        avg_processing_time = np.mean([m["performance"]["processing_time"] for m in self.metrics_history])
        avg_quality = np.mean([m["quality"]["quality_score"] for m in self.metrics_history])
        
        return {
            "total_evaluations": total_metrics,
            "average_metrics": {
                "compression_ratio": round(avg_compression, 3),
                "fkgl_score": round(avg_fkgl, 2),
                "flesch_score": round(avg_flesch, 2),
                "processing_time": round(avg_processing_time, 3),
                "quality_score": round(avg_quality, 2)
            },
            "baseline_metrics": self.baseline_metrics,
            "last_evaluation": self.metrics_history[-1] if self.metrics_history else None
        }


# Instancia global del calculador de métricas
_metrics_calculator = None

def get_metrics_calculator() -> PLSMetricsCalculator:
    """
    Obtener instancia global del calculador de métricas
    """
    global _metrics_calculator
    
    if _metrics_calculator is None:
        _metrics_calculator = PLSMetricsCalculator()
    
    return _metrics_calculator
