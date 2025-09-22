from typing import Any, Dict
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from loguru import logger

from ..models.t5_base_api import get_t5_model
from ..utils.metrics import get_metrics_calculator

# Router para este endpoint
router = APIRouter()

class GeneratePLSRequest(BaseModel):
    """Modelo de solicitud para generación de resúmenes PLS"""
    text: str = Field(..., description="Texto médico a resumir en lenguaje sencillo", min_length=10, max_length=5000)
    max_length: int = Field(100, description="Longitud máxima del resumen", ge=20, le=200)
    min_length: int = Field(30, description="Longitud mínima del resumen", ge=10, le=100)
    temperature: float = Field(0.8, description="Temperatura para la generación", ge=0.1, le=2.0)
    num_beams: int = Field(4, description="Número de beams para búsqueda", ge=1, le=10)
    include_metrics: bool = Field(True, description="Incluir métricas detalladas en la respuesta")

class GeneratePLSResponse(BaseModel):
    """Modelo de respuesta para generación de resúmenes PLS"""
    summary: str = Field(..., description="Resumen generado en lenguaje sencillo")
    original_length: int = Field(..., description="Longitud del texto original")
    summary_length: int = Field(..., description="Longitud del resumen generado")
    compression_ratio: float = Field(..., description="Ratio de compresión (resumen/original)")
    processing_time: float = Field(..., description="Tiempo de procesamiento en segundos")
    model_info: Dict[str, Any] = Field(..., description="Información del modelo utilizado")
    success: bool = Field(..., description="Indica si la generación fue exitosa")
    metrics: Dict[str, Any] = Field(None, description="Métricas detalladas (si se solicitaron)")

@router.post("/generate-pls", response_model=GeneratePLSResponse, status_code=200)
async def generate_pls(
    request: GeneratePLSRequest,
    background_tasks: BackgroundTasks
) -> GeneratePLSResponse:
    """
    Generar resumen en lenguaje sencillo (PLS) de un texto médico
    """
    try:
        logger.info(f"Recibida solicitud de generación PLS para texto de {len(request.text)} caracteres")
        
        # Obtener instancia del modelo T5-Base
        t5_model = get_t5_model()
        
        # Verificar que el modelo esté cargado
        if not t5_model.is_loaded:
            logger.error("Modelo T5-Base no está cargado")
            raise HTTPException(
                status_code=503, 
                detail="Modelo no disponible. Intente nuevamente en unos momentos."
            )
        
        # Preparar parámetros de generación
        generation_params = {
            "max_length": request.max_length,
            "min_length": request.min_length,
            "temperature": request.temperature,
            "num_beams": request.num_beams
        }
        
        # Generar resumen
        logger.info("Iniciando generación de resumen...")
        result = t5_model.generate_summary(request.text, **generation_params)
        
        if not result["success"]:
            logger.error(f"Error en generación: {result.get('error', 'Error desconocido')}")
            raise HTTPException(
                status_code=500,
                detail=f"Error generando resumen: {result.get('error', 'Error interno')}"
            )
        
        # Preparar respuesta base
        response_data = {
            "summary": result["summary"],
            "original_length": result["original_length"],
            "summary_length": result["summary_length"],
            "compression_ratio": result["compression_ratio"],
            "processing_time": result["processing_time"],
            "model_info": result["model_info"],
            "success": result["success"]
        }
        
        # Calcular métricas detalladas si se solicitan
        if request.include_metrics:
            logger.info("Calculando métricas detalladas...")
            metrics_calculator = get_metrics_calculator()
            
            # Calcular métricas completas
            detailed_metrics = metrics_calculator.calculate_comprehensive_metrics(
                original_text=request.text,
                summary_text=result["summary"],
                processing_time=result["processing_time"]
            )
            
            response_data["metrics"] = detailed_metrics
            
            # Log de métricas principales
            logger.info(f"Métricas calculadas - Compresión: {result['compression_ratio']:.3f}, "
                       f"FKGL: {detailed_metrics.get('readability', {}).get('fkgl_score', 0):.1f}, "
                       f"Flesch: {detailed_metrics.get('readability', {}).get('flesch_score', 0):.1f}")
        
        # Tarea en segundo plano: actualizar estadísticas
        background_tasks.add_task(
            _update_usage_statistics,
            result["processing_time"],
            result["success"]
        )
        
        logger.info(f"Resumen generado exitosamente en {result['processing_time']:.2f}s")
        
        return GeneratePLSResponse(**response_data)
        
    except HTTPException:
        # Re-lanzar excepciones HTTP
        raise
    except Exception as e:
        logger.error(f"Error inesperado en generación PLS: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error interno del servidor. Contacte al administrador."
        )

def _update_usage_statistics(processing_time: float, success: bool):
    """
    Actualizar estadísticas de uso en segundo plano
    """
    try:
        # Aquí se podrían actualizar estadísticas en base de datos
        # Por ahora solo loggeamos
        logger.info(f"Estadísticas actualizadas - Tiempo: {processing_time:.2f}s, Éxito: {success}")
    except Exception as e:
        logger.error(f"Error actualizando estadísticas: {e}")