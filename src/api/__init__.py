"""
API T5-Base para generación de resúmenes PLS
Módulo principal de la API REST
"""

__version__ = "1.0.0"
__author__ = "Equipo PLS - Universidad de los Andes"
__description__ = "API REST para generación de resúmenes médicos en lenguaje sencillo usando modelo T5-Base"

from .app import app
from .config import settings

__all__ = ["app", "settings", "__version__"]
