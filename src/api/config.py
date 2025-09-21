import logging
import sys
from types import FrameType
from typing import List, cast

from loguru import logger
from pydantic import AnyHttpUrl, BaseSettings

# ==============================
# Configuración de Logs
# ==============================
class LoggingSettings(BaseSettings):
    LOGGING_LEVEL: int = logging.INFO  # Niveles: DEBUG, INFO, WARNING, ERROR

# ==============================
# Configuración General
# ==============================
class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    
    # Configuración de logs
    logging: LoggingSettings = LoggingSettings()
    
    # Orígenes permitidos para consumir la API
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = [
        "http://localhost:3000",   # Dashboard local (React/Streamlit)
        "http://localhost:8000",   # Pruebas API local
        "http://localhost:8001"   # API local
    ]
    
    PROJECT_NAME: str = "T5-Base PLS API"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "API REST para generación de resúmenes médicos en lenguaje sencillo usando modelo T5-Base"
    
    # Configuración del modelo T5-Base
    MODEL_NAME: str = "t5-base"
    MODEL_DEVICE: str = "auto"  # auto, cpu, cuda
    MAX_INPUT_LENGTH: int = 5000
    DEFAULT_MAX_LENGTH: int = 100
    DEFAULT_MIN_LENGTH: int = 30
    DEFAULT_TEMPERATURE: float = 0.8
    DEFAULT_NUM_BEAMS: int = 4
    
    # Configuración de métricas
    ENABLE_DETAILED_METRICS: bool = True
    METRICS_HISTORY_SIZE: int = 1000
    
    # Configuración de rendimiento
    MAX_CONCURRENT_REQUESTS: int = 10
    REQUEST_TIMEOUT: int = 30  # segundos
    
    class Config:
        case_sensitive = True

# ==============================
# Interceptor de Logs
# ==============================
class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = str(record.levelno)

        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = cast(FrameType, frame.f_back)
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level,
            record.getMessage(),
        )

# ==============================
# Inicialización de Logging
# ==============================
def setup_app_logging(config: Settings) -> None:
    """Configura el sistema de logging con Loguru y Uvicorn."""

    LOGGERS = ("uvicorn.asgi", "uvicorn.access")
    logging.getLogger().handlers = [InterceptHandler()]
    for logger_name in LOGGERS:
        logging_logger = logging.getLogger(logger_name)
        logging_logger.handlers = [InterceptHandler(level=config.logging.LOGGING_LEVEL)]

    logger.configure(
        handlers=[{"sink": sys.stderr, "level": config.logging.LOGGING_LEVEL}]
    )

# ==============================
# Instancia Global de Configuración
# ==============================
settings = Settings()
