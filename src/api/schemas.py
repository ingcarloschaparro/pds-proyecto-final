from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

# ==============================
# Esquemas de Request
# ==============================

class GeneratePLSRequest(BaseModel):
    """Solicitud para generación de resúmenes PLS"""
    text: str = Field(
        ..., 
        description="Texto médico a resumir en lenguaje sencillo",
        min_length=10,
        max_length=5000,
        example="El paciente presenta un infarto agudo de miocardio con elevación del segmento ST..."
    )
    max_length: int = Field(
        100, 
        description="Longitud máxima del resumen en palabras",
        ge=20,
        le=200,
        example=80
    )
    min_length: int = Field(
        30, 
        description="Longitud mínima del resumen en palabras",
        ge=10,
        le=100,
        example=40
    )
    temperature: float = Field(
        0.8, 
        description="Temperatura para controlar la creatividad de la generación",
        ge=0.1,
        le=2.0,
        example=0.8
    )
    num_beams: int = Field(
        4, 
        description="Número de beams para búsqueda durante la generación",
        ge=1,
        le=10,
        example=4
    )
    include_metrics: bool = Field(
        True, 
        description="Incluir métricas detalladas en la respuesta",
        example=True
    )

# ==============================
# Esquemas de Response
# ==============================

class GeneratePLSResponse(BaseModel):
    """Respuesta para generación de resúmenes PLS"""
    summary: str = Field(
        ..., 
        description="Resumen generado en lenguaje sencillo",
        example="El paciente tuvo un ataque al corazón. Los análisis muestran que el músculo del corazón está dañado."
    )
    original_length: int = Field(
        ..., 
        description="Longitud del texto original en caracteres",
        example=245
    )
    summary_length: int = Field(
        ..., 
        description="Longitud del resumen generado en caracteres",
        example=89
    )
    compression_ratio: float = Field(
        ..., 
        description="Ratio de compresión (resumen/original)",
        example=0.363
    )
    processing_time: float = Field(
        ..., 
        description="Tiempo de procesamiento en segundos",
        example=3.24
    )
    model_info: Dict[str, Any] = Field(
        ..., 
        description="Información del modelo utilizado"
    )
    success: bool = Field(
        ..., 
        description="Indica si la generación fue exitosa",
        example=True
    )
    metrics: Optional[Dict[str, Any]] = Field(
        None, 
        description="Métricas detalladas (si se solicitaron)"
    )

class HealthResponse(BaseModel):
    """Respuesta para verificación de salud"""
    status: str = Field(
        ..., 
        description="Estado general de la API",
        example="healthy"
    )
    timestamp: float = Field(
        ..., 
        description="Timestamp de la verificación",
        example=1695123456.789
    )
    api_version: str = Field(
        ..., 
        description="Versión de la API",
        example="1.0.0"
    )
    model_status: Dict[str, Any] = Field(
        ..., 
        description="Estado del modelo T5-Base"
    )
    system_info: Dict[str, Any] = Field(
        ..., 
        description="Información del sistema"
    )
    uptime: float = Field(
        ..., 
        description="Tiempo de funcionamiento en segundos",
        example=3600.5
    )

class ModelInfoResponse(BaseModel):
    """Respuesta para información del modelo"""
    model_name: str = Field(
        ..., 
        description="Nombre del modelo",
        example="t5-base"
    )
    model_type: str = Field(
        ..., 
        description="Tipo de modelo",
        example="T5-Base"
    )
    architecture: str = Field(
        ..., 
        description="Arquitectura del modelo",
        example="transformer"
    )
    task: str = Field(
        ..., 
        description="Tarea para la que fue entrenado",
        example="summarization"
    )
    device: str = Field(
        ..., 
        description="Dispositivo donde está ejecutándose",
        example="cuda"
    )
    parameters: int = Field(
        ..., 
        description="Número de parámetros del modelo",
        example=220000000
    )
    load_time: float = Field(
        ..., 
        description="Tiempo de carga en segundos",
        example=15.3
    )
    config: Dict[str, Any] = Field(
        ..., 
        description="Configuración del modelo"
    )
    capabilities: List[str] = Field(
        ..., 
        description="Capacidades del modelo"
    )
    performance_baseline: Dict[str, float] = Field(
        ..., 
        description="Métricas de rendimiento baseline"
    )
    status: str = Field(
        ..., 
        description="Estado actual del modelo",
        example="loaded"
    )

class MetricsResponse(BaseModel):
    """Respuesta para métricas de rendimiento"""
    timestamp: float = Field(
        ..., 
        description="Timestamp de las métricas",
        example=1695123456.789
    )
    model_performance: Dict[str, Any] = Field(
        ..., 
        description="Métricas de rendimiento del modelo"
    )
    system_metrics: Dict[str, Any] = Field(
        ..., 
        description="Métricas del sistema"
    )
    quality_metrics: Dict[str, Any] = Field(
        ..., 
        description="Métricas de calidad históricas"
    )
    baseline_comparison: Dict[str, Any] = Field(
        ..., 
        description="Comparación con métricas baseline"
    )

# ==============================
# Esquemas de Métricas
# ==============================

class ReadabilityMetrics(BaseModel):
    """Métricas de legibilidad"""
    flesch_score: float = Field(..., description="Puntuación Flesch Reading Ease")
    fkgl_score: float = Field(..., description="Puntuación Flesch-Kincaid Grade Level")
    word_count: int = Field(..., description="Número de palabras")
    sentence_count: int = Field(..., description="Número de oraciones")
    avg_word_length: float = Field(..., description="Longitud promedio de palabras")
    complex_words: int = Field(..., description="Número de palabras complejas")
    complexity_ratio: float = Field(..., description="Ratio de complejidad")

class CompressionMetrics(BaseModel):
    """Métricas de compresión"""
    compression_ratio: float = Field(..., description="Ratio de compresión")
    length_reduction: float = Field(..., description="Reducción de longitud")
    word_reduction: float = Field(..., description="Reducción de palabras")
    sentence_reduction: float = Field(..., description="Reducción de oraciones")
    original_length: int = Field(..., description="Longitud original")
    summary_length: int = Field(..., description="Longitud del resumen")
    original_words: int = Field(..., description="Palabras originales")
    summary_words: int = Field(..., description="Palabras del resumen")

class QualityMetrics(BaseModel):
    """Métricas de calidad"""
    quality_score: float = Field(..., description="Score de calidad compuesto")
    readability_score: float = Field(..., description="Score de legibilidad")
    compression_score: float = Field(..., description="Score de compresión")
    complexity_score: float = Field(..., description="Score de complejidad")
    overall_grade: str = Field(..., description="Calificación general")

class PerformanceMetrics(BaseModel):
    """Métricas de rendimiento"""
    processing_time: float = Field(..., description="Tiempo de procesamiento")
    words_per_second: float = Field(..., description="Palabras por segundo")
    characters_per_second: float = Field(..., description="Caracteres por segundo")

# ==============================
# Esquemas de Error
# ==============================

class ErrorResponse(BaseModel):
    """Respuesta de error"""
    error: str = Field(..., description="Mensaje de error")
    detail: str = Field(..., description="Detalle del error")
    timestamp: float = Field(..., description="Timestamp del error")
    request_id: Optional[str] = Field(None, description="ID de la solicitud")

class ValidationErrorResponse(BaseModel):
    """Respuesta de error de validación"""
    error: str = Field(..., description="Error de validación")
    details: List[Dict[str, Any]] = Field(..., description="Detalles de los errores")
    timestamp: float = Field(..., description="Timestamp del error")

# ==============================
# Esquemas de Ejemplos
# ==============================

class ExampleRequest(BaseModel):
    """Ejemplo de solicitud"""
    name: str = Field(..., description="Nombre del ejemplo")
    description: str = Field(..., description="Descripción del ejemplo")
    request: Dict[str, Any] = Field(..., description="Solicitud de ejemplo")
    expected_response: Dict[str, Any] = Field(..., description="Respuesta esperada")

class APIExamples(BaseModel):
    """Ejemplos de uso de la API"""
    endpoint: str = Field(..., description="Endpoint")
    description: str = Field(..., description="Descripción del endpoint")
    examples: List[ExampleRequest] = Field(..., description="Lista de ejemplos")
    curl_example: str = Field(..., description="Ejemplo con cURL")
    python_example: str = Field(..., description="Ejemplo en Python")

# ==============================
# Esquemas de Configuración
# ==============================

class ModelConfiguration(BaseModel):
    """Configuración del modelo"""
    max_length: int = Field(..., description="Longitud máxima")
    min_length: int = Field(..., description="Longitud mínima")
    temperature: float = Field(..., description="Temperatura")
    num_beams: int = Field(..., description="Número de beams")
    do_sample: bool = Field(..., description="Muestreo")
    early_stopping: bool = Field(..., description="Parada temprana")
    repetition_penalty: float = Field(..., description="Penalización de repetición")
    length_penalty: float = Field(..., description="Penalización de longitud")

class SystemConfiguration(BaseModel):
    """Configuración del sistema"""
    max_concurrent_requests: int = Field(..., description="Máximo de solicitudes concurrentes")
    request_timeout: int = Field(..., description="Timeout de solicitudes")
    enable_detailed_metrics: bool = Field(..., description="Habilitar métricas detalladas")
    metrics_history_size: int = Field(..., description="Tamaño del historial de métricas")
