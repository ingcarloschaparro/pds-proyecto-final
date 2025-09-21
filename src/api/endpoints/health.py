from typing import Any, Dict
from fastapi import APIRouter, HTTPException
from loguru import logger
import time
import psutil
import platform

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
    system_info: Dict[str, Any] = Field(..., description="Información del sistema")
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
        
        # Obtener información del sistema
        system_info = _get_system_info()
        
        # Determinar estado general
        overall_status = "healthy"
        status_code = 200
        
        # Verificar estado del modelo
        if not model_health["model_loaded"]:
            overall_status = "unhealthy"
            status_code = 503
            logger.warning("Modelo T5-Base no está cargado")
        
        # Verificar recursos del sistema
        if system_info["memory_usage_percent"] > 90:
            overall_status = "degraded"
            logger.warning(f"Uso de memoria alto: {system_info['memory_usage_percent']:.1f}%")
        
        if system_info["cpu_usage_percent"] > 95:
            overall_status = "degraded"
            logger.warning(f"Uso de CPU alto: {system_info['cpu_usage_percent']:.1f}%")
        
        # Preparar respuesta
        response_data = {
            "status": overall_status,
            "timestamp": current_time,
            "api_version": "1.0.0",
            "model_status": model_health,
            "system_info": system_info,
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

def _get_system_info() -> Dict[str, Any]:
    """
    Obtener información del sistema
    """
    try:
        # Información de CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        # Información de memoria
        memory = psutil.virtual_memory()
        memory_total_gb = memory.total / (1024**3)
        memory_used_gb = memory.used / (1024**3)
        memory_percent = memory.percent
        
        # Información de disco
        disk = psutil.disk_usage('/')
        disk_total_gb = disk.total / (1024**3)
        disk_used_gb = disk.used / (1024**3)
        disk_percent = (disk.used / disk.total) * 100
        
        # Información de la plataforma
        platform_info = {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version()
        }
        
        return {
            "cpu": {
                "usage_percent": round(cpu_percent, 2),
                "count": cpu_count,
                "frequency_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else None
            },
            "memory": {
                "total_gb": round(memory_total_gb, 2),
                "used_gb": round(memory_used_gb, 2),
                "usage_percent": round(memory_percent, 2),
                "available_gb": round(memory.available / (1024**3), 2)
            },
            "disk": {
                "total_gb": round(disk_total_gb, 2),
                "used_gb": round(disk_used_gb, 2),
                "usage_percent": round(disk_percent, 2),
                "free_gb": round(disk.free / (1024**3), 2)
            },
            "platform": platform_info,
            "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
        }
        
    except Exception as e:
        logger.error(f"Error obteniendo información del sistema: {e}")
        return {
            "error": str(e),
            "cpu": {"usage_percent": 0, "count": 0},
            "memory": {"total_gb": 0, "used_gb": 0, "usage_percent": 0},
            "disk": {"total_gb": 0, "used_gb": 0, "usage_percent": 0},
            "platform": {"system": "unknown"}
        }

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
        
        # Información de procesos
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                if 'python' in proc.info['name'].lower():
                    processes.append({
                        "pid": proc.info['pid'],
                        "name": proc.info['name'],
                        "cpu_percent": proc.info['cpu_percent'],
                        "memory_percent": proc.info['memory_percent']
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # Información de red
        network_info = {}
        try:
            net_io = psutil.net_io_counters()
            network_info = {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv
            }
        except Exception as e:
            network_info = {"error": str(e)}
        
        detailed_info = {
            **basic_health.dict(),
            "model_detailed_info": model_info,
            "performance_metrics": performance_metrics,
            "processes": processes[:10],  # Top 10 procesos Python
            "network": network_info,
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
        
        # Verificar recursos mínimos
        system_info = _get_system_info()
        
        if system_info["memory"]["usage_percent"] > 95:
            raise HTTPException(
                status_code=503,
                detail="Memoria insuficiente"
            )
        
        if system_info["cpu"]["usage_percent"] > 98:
            raise HTTPException(
                status_code=503,
                detail="CPU sobrecargado"
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
            "model_loaded": True,
            "system_resources": "adequate"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en verificación de preparación: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error interno en verificación de preparación: {str(e)}"
        )
