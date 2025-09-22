import requests
import json
import time
from typing import Dict, Any

# Configuración
API_BASE_URL = "http://localhost:8001"
API_V1_URL = f"{API_BASE_URL}/api/v1"

def test_health_endpoint():
    """Probar endpoint de salud"""
    print("Probando endpoint de salud...")
    
    try:
        response = requests.get(f"{API_V1_URL}/health")
        response.raise_for_status()
        
        data = response.json()
        print(f"Estado: {data['status']}")
        print(f"Versión API: {data['api_version']}")
        print(f"Uptime: {data['uptime']:.2f}s")
        print(f"Modelo cargado: {data['model_status']['model_loaded']}")
        
        return True
        
    except Exception as e:
        print(f"Error en health check: {e}")
        return False

def test_model_info_endpoint():
    """Probar endpoint de información del modelo"""
    print("\nProbando endpoint de información del modelo...")
    
    try:
        response = requests.get(f"{API_V1_URL}/model-info")
        response.raise_for_status()
        
        data = response.json()
        print(f"Modelo: {data['model_name']}")
        print(f"Arquitectura: {data['architecture']}")
        print(f"Parámetros: {data['parameters']:,}")
        print(f"Dispositivo: {data['device']}")
        print(f"Tiempo de carga: {data['load_time']:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"Error en model info: {e}")
        return False

def test_metrics_endpoint():
    """Probar endpoint de métricas"""
    print("\nProbando endpoint de métricas...")
    
    try:
        response = requests.get(f"{API_V1_URL}/metrics")
        response.raise_for_status()
        
        data = response.json()
        print(f"Métricas obtenidas")
        print(f"Solicitudes totales: {data['model_performance']['total_requests']}")
        print(f"Tasa de éxito: {data['model_performance']['success_rate']:.2%}")
        print(f"Tiempo promedio: {data['model_performance']['average_processing_time']:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"Error en metrics: {e}")
        return False

def test_generate_endpoint():
    """Probar endpoint de generación"""
    print("\nProbando endpoint de generación...")
    
    # Texto de ejemplo
    test_text = """
    El paciente presenta un infarto agudo de miocardio con elevación del segmento ST 
    en las derivaciones II, III y aVF. Los niveles de troponina están elevados en 
    15.2 ng/mL, indicando necrosis miocárdica. Se recomienda revascularización 
    inmediata mediante intervención coronaria percutánea.
    """
    
    payload = {
        "text": test_text.strip(),
        "max_length": 80,
        "min_length": 40,
        "temperature": 0.8,
        "num_beams": 4,
        "include_metrics": True
    }
    
    try:
        print(f"Texto original ({len(test_text)} caracteres):")
        print(f"   {test_text.strip()[:100]}...")
        
        start_time = time.time()
        response = requests.post(f"{API_V1_URL}/generate-pls", json=payload)
        processing_time = time.time() - start_time
        
        response.raise_for_status()
        
        data = response.json()
        print(f"\nResumen generado:")
        print(f"   {data['summary']}")
        print(f"\nMétricas:")
        print(f"   Longitud original: {data['original_length']} caracteres")
        print(f"   Longitud resumen: {data['summary_length']} caracteres")
        print(f"   Ratio compresión: {data['compression_ratio']:.3f}")
        print(f"   Tiempo procesamiento: {data['processing_time']:.2f}s")
        print(f"   Tiempo total: {processing_time:.2f}s")
        
        # Mostrar métricas detalladas si están disponibles
        if data.get('metrics'):
            metrics = data['metrics']
            if 'readability' in metrics:
                readability = metrics['readability']
                print(f"\n📖 Legibilidad:")
                print(f"   Flesch Score: {readability['flesch_score']:.1f}")
                print(f"   FKGL Score: {readability['fkgl_score']:.1f}")
            
            if 'quality' in metrics:
                quality = metrics['quality']
                print(f"   Score de calidad: {quality['quality_score']:.1f}")
                print(f"   Calificación: {quality['overall_grade']}")
        
        return True
        
    except Exception as e:
        print(f"Error en generación: {e}")
        if hasattr(e, 'response'):
            print(f"   Respuesta: {e.response.text}")
        return False

def test_multiple_requests():
    """Probar múltiples solicitudes para métricas de rendimiento"""
    print("\nProbando múltiples solicitudes...")
    
    test_texts = [
        "El paciente presenta diabetes mellitus tipo 2 con HbA1c de 8.7%.",
        "Biopsia confirma carcinoma ductal invasivo, receptor de estrógeno positivo.",
        "Paciente diagnosticado con enfermedad pulmonar obstructiva crónica, estadio GOLD III."
    ]
    
    success_count = 0
    total_time = 0
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nSolicitud {i}/{len(test_texts)}")
        
        payload = {
            "text": text,
            "max_length": 60,
            "min_length": 30,
            "include_metrics": False
        }
        
        try:
            start_time = time.time()
            response = requests.post(f"{API_V1_URL}/generate-pls", json=payload)
            processing_time = time.time() - start_time
            
            response.raise_for_status()
            data = response.json()
            
            print(f"Resumen: {data['summary'][:50]}...")
            print(f"Tiempo: {processing_time:.2f}s")
            
            success_count += 1
            total_time += processing_time
            
        except Exception as e:
            print(f"Error en solicitud {i}: {e}")
    
    print(f"\nResumen de rendimiento:")
    print(f"   Solicitudes exitosas: {success_count}/{len(test_texts)}")
    print(f"   Tiempo total: {total_time:.2f}s")
    print(f"   Tiempo promedio: {total_time/len(test_texts):.2f}s")

def test_error_handling():
    """Probar manejo de errores"""
    print("\nProbando manejo de errores...")
    
    # Texto muy corto
    try:
        response = requests.post(f"{API_V1_URL}/generate-pls", json={
            "text": "Corto",
            "max_length": 80,
            "min_length": 40
        })
        print(f"Debería fallar con texto corto: {response.status_code}")
    except Exception as e:
        print(f"Error manejado correctamente: {e}")
    
    # Texto muy largo
    try:
        long_text = "A" * 6000  # Más del límite
        response = requests.post(f"{API_V1_URL}/generate-pls", json={
            "text": long_text,
            "max_length": 80,
            "min_length": 40
        })
        print(f"Debería fallar con texto largo: {response.status_code}")
    except Exception as e:
        print(f"Error manejado correctamente: {e}")

def main():
    """Función principal de prueba"""
    print("Iniciando pruebas de la API T5-Base")
    print("=" * 50)
    
    # Verificar que la API esté funcionando
    try:
        response = requests.get(f"{API_BASE_URL}/")
        if response.status_code != 200:
            print("La API no está funcionando. Asegúrate de que esté ejecutándose en el puerto 8001.")
            return
    except Exception as e:
        print(f"No se puede conectar a la API: {e}")
        print("Ejecuta: python -m src.api.app")
        return
    
    print("API está funcionando")
    
    # Ejecutar pruebas
    tests = [
        ("Health Check", test_health_endpoint),
        ("Model Info", test_model_info_endpoint),
        ("Metrics", test_metrics_endpoint),
        ("Generate PLS", test_generate_endpoint),
        ("Multiple Requests", test_multiple_requests),
        ("Error Handling", test_error_handling)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"{test_name} - PASÓ")
            else:
                print(f"{test_name} - FALLÓ")
        except Exception as e:
            print(f"{test_name} - ERROR: {e}")
    
    # Resumen final
    print(f"\n{'='*50}")
    print(f"Resumen de pruebas: {passed}/{total} pasaron")
    
    if passed == total:
        print("¡Todas las pruebas pasaron!")
    else:
        print("Algunas pruebas fallaron. Revisa los logs.")
    
    print("\nPara más información, visita:")
    print(f"   Documentación: {API_BASE_URL}/docs")
    print(f"   ReDoc: {API_BASE_URL}/redoc")

if __name__ == "__main__":
    main()
