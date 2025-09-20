import time
import logging
from typing import Dict, Any, Optional
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
import torch
from loguru import logger

class T5BaseAPI:
    """Wrapper para el modelo T5-Base optimizado para la API"""
    
    def __init__(self, model_name: str = "t5-base", device: str = "auto"):
        """
        Inicializar el modelo T5-Base
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.is_loaded = False
        self.load_time = None
        self.model_info = {}
        
        # Configuración del modelo
        self.config = {
            "max_length": 100,
            "min_length": 30,
            "num_beams": 4,
            "temperature": 0.8,
            "do_sample": True,
            "early_stopping": True,
            "repetition_penalty": 1.1,
            "length_penalty": 1.0
        }
        
        # Métricas de rendimiento
        self.performance_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0,
            "last_request_time": None
        }
        
        logger.info(f"Inicializando T5BaseAPI con modelo: {model_name}")
    
    def load_model(self) -> bool:
        """
        Cargar el modelo T5-Base y el tokenizer
        """
        try:
            start_time = time.time()
            logger.info("Cargando modelo T5-Base...")
            
            # Determinar dispositivo
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Usando dispositivo: {self.device}")
            
            # Cargar tokenizer
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            
            # Cargar modelo
            self.model = T5ForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            # Mover modelo al dispositivo
            self.model.to(self.device)
            
            # Crear pipeline de summarization
            self.pipeline = pipeline(
                "summarization",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                framework="pt"
            )
            
            self.load_time = time.time() - start_time
            self.is_loaded = True
            
            # Información del modelo
            self.model_info = {
                "model_name": self.model_name,
                "model_type": "T5-Base",
                "architecture": "transformer",
                "task": "summarization",
                "device": self.device,
                "load_time": self.load_time,
                "parameters": self.model.num_parameters(),
                "config": self.config
            }
            
            logger.info(f"Modelo T5-Base cargado exitosamente en {self.load_time:.2f}s")
            logger.info(f"Parámetros del modelo: {self.model.num_parameters():,}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error cargando modelo T5-Base: {e}")
            self.is_loaded = False
            return False
    
    def generate_summary(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Generar resumen PLS del texto de entrada
        """
        if not self.is_loaded:
            raise RuntimeError("Modelo no cargado. Llame a load_model() primero.")
        
        start_time = time.time()
        self.performance_metrics["total_requests"] += 1
        
        try:
            # Validar entrada
            if not text or len(text.strip()) == 0:
                raise ValueError("El texto de entrada no puede estar vacío")
            
            # Truncar texto si es muy largo (T5 tiene límite de 512 tokens)
            if len(text) > 2000:  # Aproximadamente 512 tokens
                text = text[:2000]
                logger.warning("Texto truncado a 2000 caracteres")
            
            # Preparar texto para T5 (agregar prefijo de tarea)
            input_text = f"summarize: {text}"
            
            # Configuración de generación
            generation_config = {**self.config, **kwargs}
            
            # Generar resumen
            result = self.pipeline(
                input_text,
                max_length=generation_config["max_length"],
                min_length=generation_config["min_length"],
                num_beams=generation_config["num_beams"],
                temperature=generation_config["temperature"],
                do_sample=generation_config["do_sample"],
                early_stopping=generation_config["early_stopping"],
                repetition_penalty=generation_config["repetition_penalty"],
                length_penalty=generation_config["length_penalty"]
            )
            
            # Extraer resumen
            summary = result[0]["summary_text"] if result else ""
            
            # Calcular métricas
            processing_time = time.time() - start_time
            
            # Actualizar métricas de rendimiento
            self.performance_metrics["successful_requests"] += 1
            self.performance_metrics["total_processing_time"] += processing_time
            self.performance_metrics["average_processing_time"] = (
                self.performance_metrics["total_processing_time"] / 
                self.performance_metrics["successful_requests"]
            )
            self.performance_metrics["last_request_time"] = time.time()
            
            # Calcular métricas de compresión
            compression_ratio = len(summary) / len(text) if len(text) > 0 else 0
            
            result_data = {
                "summary": summary,
                "original_length": len(text),
                "summary_length": len(summary),
                "compression_ratio": compression_ratio,
                "processing_time": processing_time,
                "model_info": {
                    "model_name": self.model_name,
                    "device": self.device,
                    "config_used": generation_config
                },
                "success": True
            }
            
            logger.info(f"Resumen generado exitosamente en {processing_time:.2f}s")
            logger.info(f"Compresión: {compression_ratio:.3f}")
            
            return result_data
            
        except Exception as e:
            self.performance_metrics["failed_requests"] += 1
            processing_time = time.time() - start_time
            
            logger.error(f"Error generando resumen: {e}")
            
            return {
                "summary": "",
                "error": str(e),
                "processing_time": processing_time,
                "success": False
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Obtener información detallada del modelo
        """
        if not self.is_loaded:
            return {"error": "Modelo no cargado"}
        
        return {
            **self.model_info,
            "performance_metrics": self.performance_metrics,
            "status": "loaded" if self.is_loaded else "not_loaded"
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Obtener métricas de rendimiento del modelo
        """
        return {
            **self.performance_metrics,
            "success_rate": (
                self.performance_metrics["successful_requests"] / 
                self.performance_metrics["total_requests"]
                if self.performance_metrics["total_requests"] > 0 else 0
            ),
            "is_loaded": self.is_loaded
        }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Verificar el estado de salud del modelo
        """
        return {
            "model_loaded": self.is_loaded,
            "model_name": self.model_name,
            "device": self.device,
            "total_requests": self.performance_metrics["total_requests"],
            "success_rate": (
                self.performance_metrics["successful_requests"] / 
                self.performance_metrics["total_requests"]
                if self.performance_metrics["total_requests"] > 0 else 0
            ),
            "average_processing_time": self.performance_metrics["average_processing_time"],
            "status": "healthy" if self.is_loaded else "unhealthy"
        }
    
    def unload_model(self):
        """Descargar el modelo de memoria"""
        if self.is_loaded:
            del self.model
            del self.tokenizer
            del self.pipeline
            self.model = None
            self.tokenizer = None
            self.pipeline = None
            self.is_loaded = False
            logger.info("Modelo T5-Base descargado de memoria")


# Instancia global del modelo (singleton)
_t5_model_instance = None

def get_t5_model() -> T5BaseAPI:
    """
    Obtener instancia global del modelo T5-Base (singleton) para la API
    """
    global _t5_model_instance
    
    if _t5_model_instance is None:
        _t5_model_instance = T5BaseAPI()
        _t5_model_instance.load_model()
    
    return _t5_model_instance

def reload_t5_model():
    """Recargar el modelo T5-Base"""
    global _t5_model_instance
    
    if _t5_model_instance is not None:
        _t5_model_instance.unload_model()
    
    _t5_model_instance = T5BaseAPI()
    _t5_model_instance.load_model()
    
    return _t5_model_instance
