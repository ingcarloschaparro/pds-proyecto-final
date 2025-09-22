from typing import Any
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from loguru import logger

import uvicorn
from dotenv import load_dotenv

from .config import settings, setup_app_logging
from .endpoints import generate, health, info, metrics

# Configurar logging
setup_app_logging(config=settings)

# Crear aplicación FastAPI
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="API REST para generación de resúmenes médicos en lenguaje sencillo (PLS) usando modelo T5-Base",
    version="1.0.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "Equipo PLS - Universidad de los Andes",
        "email": "pls-team@uniandes.edu.co",
        "url": "https://github.com/pls-project/api"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    }
)

# Configurar CORS
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Página de inicio
@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> Any:
    """
    Página de inicio de la API
    """
    body = (
        "<html>"
        "<head>"
        "<title>API T5-Base - Resúmenes PLS</title>"
        "<style>"
        "body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }"
        ".container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }"
        "h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }"
        ".endpoint { background: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #3498db; }"
        ".method { font-weight: bold; color: #27ae60; }"
        ".description { color: #7f8c8d; margin-top: 5px; }"
        "a { color: #3498db; text-decoration: none; }"
        "a:hover { text-decoration: underline; }"
        ".footer { margin-top: 30px; padding-top: 20px; border-top: 1px solid #bdc3c7; color: #7f8c8d; text-align: center; }"
        "</style>"
        "</head>"
        "<body>"
        "<div class='container'>"
        "<h1>API T5-Base - Resúmenes en Lenguaje Sencillo</h1>"
        "<p>API REST para convertir texto médico complejo en resúmenes accesibles y comprensibles para pacientes y público general.</p>"
        
        "<h2>Endpoints Disponibles</h2>"
        
        "<div class='endpoint'>"
        "<div class='method'>POST /api/v1/generate-pls</div>"
        "<div class='description'>Generar resumen en lenguaje sencillo de texto médico</div>"
        "</div>"
        
        "<div class='endpoint'>"
        "<div class='method'>GET /api/v1/health</div>"
        "<div class='description'>Verificar estado de salud de la API y el modelo</div>"
        "</div>"
        
        "<div class='endpoint'>"
        "<div class='method'>GET /api/v1/model-info</div>"
        "<div class='description'>Obtener información detallada del modelo T5-Base</div>"
        "</div>"
        
        "<div class='endpoint'>"
        "<div class='method'>GET /api/v1/metrics</div>"
        "<div class='description'>Obtener métricas de rendimiento y calidad</div>"
        "</div>"
        
        "<h2>Documentación</h2>"
        "<p>"
        "<a href='/docs' target='_blank'>Documentación Swagger/OpenAPI</a> | "
        "<a href='/redoc' target='_blank'>Documentación ReDoc</a>"
        "</p>"
        
        "<h2>Inicio Rápido</h2>"
        "<p>Para comenzar a usar la API, envía una solicitud POST a <code>/api/v1/generate-pls</code> con el texto médico que deseas resumir.</p>"
        
        "<h3>Ejemplo con cURL:</h3>"
        "<pre style='background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px; overflow-x: auto;'>"
        "curl -X POST 'http://localhost:8001/api/v1/generate-pls' \\\n"
        "     -H 'Content-Type: application/json' \\\n"
        "     -d '{\n"
        "       \"text\": \"El paciente presenta un infarto agudo de miocardio...\",\n"
        "       \"max_length\": 80,\n"
        "       \"min_length\": 40,\n"
        "       \"include_metrics\": true\n"
        "     }'"
        "</pre>"
        
        "<h2>Características</h2>"
        "<ul>"
        "<li><strong>Modelo T5-Base:</strong> Transformer especializado en tareas de texto-a-texto</li>"
        "<li><strong>Métricas de Calidad:</strong> FKGL, Flesch, ROUGE, BLEU, compresión</li>"
        "<li><strong>API REST:</strong> Interfaz estándar con documentación OpenAPI</li>"
        "<li><strong>Monitoreo:</strong> Métricas de rendimiento y salud del sistema</li>"
        "<li><strong>Escalabilidad:</strong> Diseñado para manejar múltiples solicitudes</li>"
        "</ul>"
        
        "<div class='footer'>"
        "<p>Desarrollado por el Equipo PLS - Universidad de los Andes</p>"
        "<p>Versión 1.0.0 | <a href='https://github.com/pls-project' target='_blank'>GitHub</a></p>"
        "</div>"
        "</div>"
        "</body>"
        "</html>"
    )
    return HTMLResponse(content=body)

# Montar routers de endpoints
app.include_router(generate.router, prefix=settings.API_V1_STR, tags=["Generación"])
app.include_router(health.router, prefix=settings.API_V1_STR, tags=["Salud"])
app.include_router(info.router, prefix=settings.API_V1_STR, tags=["Información"])
app.include_router(metrics.router, prefix=settings.API_V1_STR, tags=["Métricas"])

# Eventos de inicio y cierre
@app.on_event("startup")
async def startup_event():
    """Evento de inicio de la aplicación"""
    logger.info("Iniciando API T5-Base...")
    logger.info(f"Proyecto: {settings.PROJECT_NAME}")
    logger.info(f"CORS Origins: {settings.BACKEND_CORS_ORIGINS}")
    logger.info("API iniciada correctamente")

@app.on_event("shutdown")
async def shutdown_event():
    """Evento de cierre de la aplicación"""
    logger.info("Cerrando API T5-Base...")
    logger.info("API cerrada correctamente")

# Middleware personalizado para logging de requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware para logging de requests HTTP"""
    start_time = time.time()
    
    # Log de request
    logger.info(f"Request: {request.method} {request.url.path} - Cliente: {request.client.host if request.client else 'unknown'}")
    
    # Procesar request
    response = await call_next(request)
    
    # Calcular tiempo de procesamiento
    process_time = time.time() - start_time
    
    # Log de response
    logger.info(f"Response: {request.method} {request.url.path} - Status: {response.status_code} - Tiempo: {process_time:.3f}s")
    
    return response

# Importar time para el middleware
import time

# Función para ejecutar la aplicación
def run_app():
    """Función para ejecutar la aplicación en modo desarrollo"""
    load_dotenv()

    logger.warning("Ejecutando en modo desarrollo. No usar en producción.")
    
    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=8001,
        log_level="info",
        reload=True
    )

if __name__ == "__main__":
    run_app()
