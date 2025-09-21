from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from loguru import logger
import time

from ..models.t5_base_api import get_t5_model
from ..utils.metrics import get_metrics_calculator

# Router para este endpoint
router = APIRouter()

class MetricsResponse(BaseModel):
    """Modelo de respuesta para métricas de rendimiento"""
    timestamp: float = Field(..., description="Timestamp de las métricas")
    model_performance: Dict[str, Any] = Field(..., description="Métricas de rendimiento del modelo")
    quality_metrics: Dict[str, Any] = Field(..., description="Métricas de calidad históricas")
    baseline_comparison: Dict[str, Any] = Field(..., description="Comparación con métricas baseline")

@router.get("/metrics", response_model=MetricsResponse, status_code=200)
async def get_performance_metrics() -> MetricsResponse:
    """
    Obtener métricas de rendimiento del modelo T5-Base y del sistema
    """
    try:
        logger.info("Obteniendo métricas de rendimiento...")
        
        current_time = time.time()
        
        # Obtener métricas del modelo
        t5_model = get_t5_model()
        model_performance = t5_model.get_performance_metrics()
        
        # Obtener métricas de calidad históricas
        metrics_calculator = get_metrics_calculator()
        quality_summary = metrics_calculator.get_metrics_summary()
        
        
        # Comparación con baseline
        baseline_comparison = _get_baseline_comparison(quality_summary)
        
        # Preparar respuesta
        response_data = {
            "timestamp": current_time,
            "model_performance": model_performance,
            "quality_metrics": quality_summary,
            "baseline_comparison": baseline_comparison
        }
        
        logger.info("Métricas de rendimiento obtenidas exitosamente")
        return MetricsResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Error obteniendo métricas: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error interno obteniendo métricas de rendimiento"
        )

@router.get("/metrics/detailed")
async def get_detailed_metrics() -> Dict[str, Any]:
    """
    Obtener métricas detalladas con información adicional
    """
    try:
        logger.info("Obteniendo métricas detalladas...")
        
        # Métricas básicas
        basic_metrics = await get_performance_metrics()
        
        # Análisis de tendencias
        trends_analysis = _analyze_trends()
        
        # Métricas de calidad por categoría
        quality_breakdown = _get_quality_breakdown()
        
        # Recomendaciones de optimización
        optimization_recommendations = _get_optimization_recommendations(basic_metrics)
        
        detailed_metrics = {
            **basic_metrics.dict(),
            "trends_analysis": trends_analysis,
            "quality_breakdown": quality_breakdown,
            "optimization_recommendations": optimization_recommendations,
            "detailed_analysis": True
        }
        
        logger.info("Métricas detalladas obtenidas exitosamente")
        return detailed_metrics
        
    except Exception as e:
        logger.error(f"Error obteniendo métricas detalladas: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error interno obteniendo métricas detalladas"
        )

@router.get("/metrics/health")
async def get_metrics_health() -> Dict[str, Any]:
    """
    Obtener estado de salud basado en métricas
    """
    try:
        logger.info("Analizando salud basada en métricas...")
        
        # Obtener métricas actuales
        metrics = await get_performance_metrics()
        
        # Analizar salud
        health_analysis = _analyze_health(metrics)
        
        # Generar alertas
        alerts = _generate_alerts(metrics)
        
        # Calcular score de salud
        health_score = _calculate_health_score(metrics)
        
        health_status = {
            "health_score": health_score,
            "status": "healthy" if health_score > 0.8 else "degraded" if health_score > 0.6 else "unhealthy",
            "analysis": health_analysis,
            "alerts": alerts,
            "recommendations": _get_health_recommendations(health_score, alerts),
            "timestamp": time.time()
        }
        
        logger.info(f"Análisis de salud completado - Score: {health_score:.2f}")
        return health_status
        
    except Exception as e:
        logger.error(f"Error analizando salud: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error interno analizando salud de métricas"
        )


def _get_baseline_comparison(quality_summary: Dict[str, Any]) -> Dict[str, Any]:
    """Obtener comparación con métricas baseline"""
    try:
        baseline_metrics = {
            "compression_ratio": 0.292,
            "fkgl_score": 12.2,
            "flesch_score": 39.0,
            "processing_time": 3.64,
            "quality_score": 9.2
        }
        
        if "average_metrics" not in quality_summary:
            return {"message": "No hay métricas históricas para comparar"}
        
        current_metrics = quality_summary["average_metrics"]
        comparison = {}
        
        for metric, baseline_value in baseline_metrics.items():
            if metric in current_metrics:
                current_value = current_metrics[metric]
                
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
                
                comparison[metric] {
                    "current": current_value,
                    "baseline": baseline_value,
                    "difference_percent": round(diff_percent, 2),
                    "status": status
                }
        
        return comparison
        
    except Exception as e:
        logger.error(f"Error calculando comparación con baseline: {e}")
        return {"error": str(e)}

def _analyze_trends() -> Dict[str, Any]:
    """Analizar tendencias en las métricas"""
    try:
        # Aquí se implementaría análisis de tendencias temporal
        # Por ahora retornamos un placeholder
        return {
            "trend_analysis": "Not implemented yet",
            "time_series_data": [],
            "trend_direction": "stable",
            "confidence": 0.0
        }
    except Exception as e:
        logger.error(f"Error analizando tendencias: {e}")
        return {"error": str(e)}

def _get_quality_breakdown() -> Dict[str, Any]:
    """Obtener desglose de métricas de calidad por categoría"""
    try:
        return {
            "readability": {
                "flesch_score": {"current": 0, "target": 60, "status": "needs_improvement"},
                "fkgl_score": {"current": 0, "target": 8, "status": "needs_improvement"},
                "gunning_fog": {"current": 0, "target": 12, "status": "needs_improvement"}
            },
            "compression": {
                "compression_ratio": {"current": 0, "target": 0.3, "status": "needs_improvement"},
                "length_reduction": {"current": 0, "target": 0.7, "status": "needs_improvement"}
            },
            "performance": {
                "processing_time": {"current": 0, "target": 3.0, "status": "needs_improvement"},
                "success_rate": {"current": 0, "target": 0.95, "status": "needs_improvement"}
            }
        }
    except Exception as e:
        logger.error(f"Error obteniendo desglose de calidad: {e}")
        return {"error": str(e)}

def _get_optimization_recommendations(metrics: MetricsResponse) -> List[str]:
    """Obtener recomendaciones de optimización basadas en métricas"""
    try:
        recommendations = []
        
        # Analizar métricas de rendimiento
        if metrics.model_performance.get("average_processing_time", 0) > 5.0:
            recommendations.append("Considerar optimización del modelo para reducir tiempo de procesamiento")
        
        if metrics.model_performance.get("success_rate", 0) < 0.9:
            recommendations.append("Investigar causas de fallos en la generación de resúmenes")
        
        # Analizar métricas de calidad
        if "average_metrics" in metrics.quality_metrics:
            avg_metrics = metrics.quality_metrics["average_metrics"]
            
            if avg_metrics.get("fkgl_score", 0) > 15:
                recommendations.append("Mejorar legibilidad del texto generado")
            
            if avg_metrics.get("compression_ratio", 0) < 0.2:
                recommendations.append("Ajustar parámetros para mejor compresión")
        
        
        if not recommendations:
            recommendations.append("Sistema funcionando dentro de parámetros óptimos")
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Error generando recomendaciones: {e}")
        return ["Error generando recomendaciones de optimización"]

def _analyze_health(metrics: MetricsResponse) -> Dict[str, Any]:
    """Analizar salud basada en métricas"""
    try:
        health_indicators = {
            "model_loaded": True,
            "processing_time_ok": metrics.model_performance.get("average_processing_time", 0) < 10.0,
            "success_rate_ok": metrics.model_performance.get("success_rate", 0) > 0.8
        }
        
        healthy_indicators = sum(health_indicators.values())
        total_indicators = len(health_indicators)
        
        return {
            "indicators": health_indicators,
            "healthy_count": healthy_indicators,
            "total_count": total_indicators,
            "health_percentage": (healthy_indicators / total_indicators) * 100
        }
        
    except Exception as e:
        logger.error(f"Error analizando salud: {e}")
        return {"error": str(e)}

def _generate_alerts(metrics: MetricsResponse) -> List[Dict[str, Any]]:
    """Generar alertas basadas en métricas"""
    try:
        alerts = []
        
        # Alertas de rendimiento
        if metrics.model_performance.get("average_processing_time", 0) > 10.0:
            alerts.append({
                "type": "performance",
                "severity": "warning",
                "message": "Tiempo de procesamiento alto",
                "value": metrics.model_performance.get("average_processing_time", 0),
                "threshold": 10.0
            })
        
        if metrics.model_performance.get("success_rate", 0) < 0.8:
            alerts.append({
                "type": "reliability",
                "severity": "error",
                "message": "Tasa de éxito baja",
                "value": metrics.model_performance.get("success_rate", 0),
                "threshold": 0.8
            })
        
        
        return alerts
        
    except Exception as e:
        logger.error(f"Error generando alertas: {e}")
        return [{"type": "system", "severity": "error", "message": f"Error generando alertas: {str(e)}"}]

def _calculate_health_score(metrics: MetricsResponse) -> float:
    """Calcular score de salud general"""
    try:
        score = 0.0
        total_weight = 0.0
        
        # Peso de métricas de rendimiento (40%)
        performance_weight = 0.4
        success_rate = metrics.model_performance.get("success_rate", 0)
        processing_time = metrics.model_performance.get("average_processing_time", 0)
        
        # Normalizar success rate (0-1)
        success_score = success_rate
        
        # Normalizar processing time (0-1, menor es mejor)
        processing_score = max(0, 1 - (processing_time / 20.0))  # 20s es el máximo esperado
        
        performance_score = (success_score + processing_score) / 2
        score += performance_score * performance_weight
        total_weight += performance_weight
        
        
        # Peso de métricas de calidad (30%)
        quality_weight = 0.3
        if "average_metrics" in metrics.quality_metrics:
            avg_metrics = metrics.quality_metrics["average_metrics"]
            
            # Normalizar FKGL (menor es mejor)
            fkgl_score = avg_metrics.get("fkgl_score", 20)
            fkgl_normalized = max(0, 1 - (fkgl_score / 20.0))
            
            # Normalizar Flesch (mayor es mejor)
            flesch_score = avg_metrics.get("flesch_score", 0)
            flesch_normalized = min(1, flesch_score / 100.0)
            
            quality_score = (fkgl_normalized + flesch_normalized) / 2
        else:
            quality_score = 0.5  # Valor neutral si no hay métricas de calidad
        
        score += quality_score * quality_weight
        total_weight += quality_weight
        
        # Normalizar score final
        if total_weight > 0:
            score = score / total_weight
        
        return round(score, 3)
        
    except Exception as e:
        logger.error(f"Error calculando score de salud: {e}")
        return 0.0

def _get_health_recommendations(health_score: float, alerts: List[Dict[str, Any]]) -> List[str]:
    """Obtener recomendaciones de salud basadas en score y alertas"""
    try:
        recommendations = []
        
        if health_score < 0.6:
            recommendations.append("Sistema en estado crítico - Revisar configuración y recursos")
        elif health_score < 0.8:
            recommendations.append("Sistema con problemas - Monitorear métricas de cerca")
        else:
            recommendations.append("Sistema funcionando correctamente")
        
        # Recomendaciones basadas en alertas
        for alert in alerts:
            if alert["type"] == "performance":
                recommendations.append("Optimizar procesamiento del modelo")
            elif alert["type"] == "reliability":
                recommendations.append("Investigar causas de fallos")
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Error generando recomendaciones de salud: {e}")
        return ["Error generando recomendaciones de salud"]
