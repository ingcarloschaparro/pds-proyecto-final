from typing import Any, Dict
from fastapi import APIRouter, HTTPException
from loguru import logger
import time

from pydantic import BaseModel, Field

from ..models.t5_base_api import get_t5_model

# Router para este endpoint
router = APIRouter()

class HealthResponse(BaseModel):
    """Modelo de respuesta para el endpoint de salud"""
    status: str = Field(..., description="Estado general de la API")
    timestamp: float = Field(..., description="Timestamp de la verificación")
    api_version: str = Field(..., description="Versión de la API")
    model_status: Dict[str, Any] = Field(..., description="Estado del modelo T5-Base")
    uptime: float = Field(..., description="Tiempo de funcionamiento en segundos")

# Variable para tracking de uptime
_start_time = time.time()

@router.get("/health", response_model=HealthResponse, status_code=200)
async def health_check() -> HealthResponse:
    """
    Verificar el estado de salud de la API y el modelo T5-Base
    """
    try:
        logger.info("Realizando verificación de salud de la API...")
        
        # Obtener timestamp actual
        current_time = time.time()
        uptime = current_time - _start_time
        
        # Obtener información del modelo
        t5_model = get_t5_model()
        model_health = t5_model.health_check()
        
        
        # Determinar estado general
        overall_status = "healthy"
        status_code = 200
        
        # Verificar estado del modelo
        if not model_health["model_loaded"]:
            overall_status = "unhealthy"
            status_code = 503
            logger.warning("Modelo T5-Base no está cargado")
        
        
        # Preparar respuesta
        response_data = {
            "status": overall_status,
            "timestamp": current_time,
            "api_version": "1.0.0",
            "model_status": model_health,
            "uptime": uptime
        }
        
        logger.info(f"Verificación de salud completada - Estado: {overall_status}")
        
        # Si hay problemas, lanzar excepción HTTP
        if status_code != 200:
            raise HTTPException(
                status_code=status_code,
                detail=f"API en estado {overall_status}. Verifique los detalles en la respuesta."
            )
        
        return HealthResponse(**response_data)
        
    except HTTPException:
        # Re-lanzar excepciones HTTP
        raise
    except Exception as e:
        logger.error(f"Error en verificación de salud: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error interno durante la verificación de salud"
        )


@router.get("/health/detailed")
async def detailed_health_check() -> Dict[str, Any]:
    """
    Verificación de salud detallada con información adicional
    """
    try:
        logger.info("Realizando verificación de salud detallada...")
        
        # Verificación básica
        basic_health = await health_check()
        
        # Información adicional del modelo
        t5_model = get_t5_model()
        model_info = t5_model.get_model_info()
        performance_metrics = t5_model.get_performance_metrics()
        
        
        detailed_info = {
            **basic_health.dict(),
            "model_detailed_info": model_info,
            "performance_metrics": performance_metrics,
            "detailed_check": True
        }
        
        logger.info("Verificación de salud detallada completada")
        return detailed_info
        
    except Exception as e:
        logger.error(f"Error en verificación de salud detallada: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error en verificación detallada: {str(e)}"
        )

@router.get("/health/ready")
async def readiness_check() -> Dict[str, Any]:
    """
    Verificación de preparación (readiness) de la API
    """
    try:
        logger.info("Realizando verificación de preparación...")
        
        # Verificar que el modelo esté cargado
        t5_model = get_t5_model()
        if not t5_model.is_loaded:
            raise HTTPException(
                status_code=503,
                detail="Modelo T5-Base no está cargado"
            )
        
        
        # Verificar que el modelo responda
        try:
            test_result = t5_model.generate_summary("Test de funcionamiento del modelo.")
            if not test_result["success"]:
                raise HTTPException(
                    status_code=503,
                    detail="Modelo no responde correctamente"
                )
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Error en prueba del modelo: {str(e)}"
            )
        
        return {
            "status": "ready",
            "timestamp": time.time(),
            "message": "API lista para recibir tráfico",
            "model_loaded": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en verificación de preparación: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error interno en verificación de preparación: {str(e)}"
        )
