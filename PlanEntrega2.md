
# �� PLAN DE TRABAJO TÉCNICO - PROYECTO PLS

## �� **RESUMEN EJECUTIVO**

El proyecto actual tiene una **arquitectura sólida** con pipeline DVC funcional y datos procesados, pero los **scripts de modelos están vacíos** y falta implementación de MLflow, dashboard y documentación completa. El dataset tiene **182,753 registros** con solo **27% de datos anotados** (PLS), lo que requiere estrategias específicas para datos limitados.

---

## 🚀 **FASE 1: MODELOS INICIALES Y BASELINES**

### **1.1 Análisis del Estado Actual**
- **Entregable**: Documento de análisis técnico
- **Criterios**: Identificación clara de gaps y oportunidades
- **Tareas**:
  - [ ] Analizar distribución de datos por fuente (Cochrane: 69%, Trial Summaries: 28%, Pfizer: 1.5%, ClinicalTrials: 0.6%)
  - [ ] Evaluar calidad de anotaciones PLS vs non-PLS
  - [ ] Identificar patrones de longitud y complejidad por fuente

### **1.2 Modelos Base a Implementar**

#### **Clasificador PLS vs non-PLS**
- **Baseline 1**: TF-IDF + Logistic Regression (rápido, interpretable)
- **Baseline 2**: DistilBERT fine-tuning (contextual, mejor rendimiento)
- **Estrategia**: Semi-supervisado con datos no etiquetados
- **Entregable**: Scripts funcionales en `src/models/train_classifier.py`
- **Criterios**: F1 macro ≥ 0.80, matriz de confusión por fuente

#### **Generador de PLS**
- **Modelo 1**: BART-base (robusto, recursos moderados)
- **Modelo 2**: T5-base (bueno para tareas de resumen)
- **Modelo 3**: BioBART (si está disponible, dominio específico)
- **Estrategia**: Fine-tuning con LoRA para recursos limitados
- **Entregable**: Scripts funcionales en `src/models/generate_pls.py`
- **Criterios**: Generación de resúmenes 120-250 palabras, legibilidad mejorada

### **1.3 Métricas de Evaluación**
- **ROUGE-L**: Similitud léxica con ground truth
- **METEOR**: Alineación semántica y flexibilidad
- **FKGL**: Flesch-Kincaid Grade Level (legibilidad)
- **BERTScore**: Similitud semántica contextual
- **Compresión**: Ratio palabras PLS/original (0.30-0.50 objetivo)

---

## 🔬 **FASE 2: EXPERIMENTOS Y VERSIONADO CON MLFLOW**

### **2.1 Integración MLflow + DVC**
- **Arquitectura**: MLflow para experimentos, DVC para datos y modelos
- **Entregable**: Configuración `mlflow.yaml` y scripts de logging
- **Tareas**:
  - [ ] Configurar MLflow tracking server local
  - [ ] Integrar logging en scripts de entrenamiento
  - [ ] Definir parámetros experimentales en `params.yaml`

### **2.2 Parámetros a Registrar**
```yaml
# Ejemplo de estructura en params.yaml
experiments:
  classifier:
    - model_type: ["tfidf_logreg", "distilbert", "biobert"]
    - learning_rate: [0.001, 0.01, 0.1]
    - batch_size: [16, 32, 64]
  generator:
    - model_name: ["bart-base", "t5-base", "biobart"]
    - max_length: [150, 200, 250]
    - temperature: [0.7, 0.8, 0.9]
```

### **2.3 Convención de Nombres**
- **Experimentos**: `{fase}_{modelo}_{fecha}`
  - Ejemplo: `classifier_tfidf_20250115`
- **Runs**: `{modelo}_{parametros}_{timestamp}`
  - Ejemplo: `distilbert_lr0.001_bs32_20250115_143022`
- **Modelos**: `{tipo}_{fuente}_{version}`
  - Ejemplo: `classifier_cochrane_v1.0`

### **2.4 Artefactos a Registrar**
- [ ] Modelos entrenados (formato HuggingFace)
- [ ] Métricas de evaluación (JSON)
- [ ] Gráficos de entrenamiento (PNG)
- [ ] Ejemplos de predicciones (CSV)
- [ ] Configuración de parámetros (YAML)

---

## 📊 **FASE 3: DASHBOARD INTERACTIVO**

### **3.1 Herramienta Seleccionada: Streamlit**
- **Justificación**: Fácil integración con MLflow, visualizaciones ricas, despliegue simple
- **Alternativa**: Gradio (si se requiere más simplicidad)

### **3.2 Secciones del Dashboard**

#### **Panel Principal**
- **Métricas Globales**: Resumen de rendimiento por modelo
- **Comparación de Modelos**: Gráficos de barras y líneas
- **Estado del Pipeline**: Indicadores DVC y MLflow

#### **Análisis de Datos**
- **Distribución por Fuente**: Gráficos de torta y barras
- **Calidad de Anotaciones**: Estadísticas de PLS vs non-PLS
- **Exploración de Textos**: Búsqueda y visualización de ejemplos

#### **Evaluación de Modelos**
- **Métricas por Fuente**: ROUGE, METEOR, FKGL estratificados
- **Matrices de Confusión**: Visualización interactiva
- **Ejemplos de Predicciones**: Comparación lado a lado

#### **Gestión de Experimentos**
- **Historial MLflow**: Búsqueda y comparación de runs
- **Parámetros Óptimos**: Identificación automática de mejores configuraciones
- **Exportación de Resultados**: Descarga de métricas y modelos

### **3.3 Entregables**
- [ ] Script principal `dashboard.py` en `src/`
- [ ] Componentes modulares en `src/dashboard/`
- [ ] Configuración de estilos y temas
- [ ] Documentación de uso del dashboard

---

## �� **FASE 4: DOCUMENTACIÓN COMPLETA**

### **4.1 README Ampliado**
- [ ] **Instalación**: Dependencias, configuración DVC/MLflow
- [ ] **Uso Rápido**: Comandos para ejecutar pipeline completo
- [ ] **Visualización**: Instrucciones para dashboard
- [ ] **Troubleshooting**: Problemas comunes y soluciones
- [ ] **Contribución**: Guía para desarrolladores

### **4.2 Documentación de Scripts**
- [ ] **Docstrings en Español**: Todos los módulos y funciones
- [ ] **Ejemplos de Uso**: Casos de uso comunes
- [ ] **Parámetros**: Descripción detallada de configuración
- [ ] **Salidas**: Formato y ubicación de archivos generados

### **4.3 Lineamientos de Estilo**
```python
# Ejemplo de docstring estándar
def entrenar_clasificador(
    datos_entrenamiento: pd.DataFrame,
    parametros: Dict[str, Any]
) -> Tuple[Any, Dict[str, float]]:
    """
    Entrena un clasificador para distinguir entre textos PLS y no-PLS.
    
    Args:
        datos_entrenamiento: DataFrame con columnas 'texto' y 'label'
        parametros: Diccionario con hiperparámetros del modelo
        
    Returns:
        Tuple con modelo entrenado y métricas de rendimiento
        
    Raises:
        ValueError: Si los datos no tienen el formato esperado
    """
```

---

## ✅ **FASE 5: REVISIÓN Y APROBACIÓN**

### **5.1 Criterios de Aceptación por Fase**

#### **Fase 1 - Modelos**
- [ ] Scripts de clasificación ejecutándose sin errores
- [ ] Scripts de generación produciendo resúmenes coherentes
- [ ] Métricas de evaluación calculándose correctamente
- [ ] Pipeline DVC ejecutándose end-to-end

#### **Fase 2 - MLflow**
- [ ] Experimentos registrándose en MLflow
- [ ] Parámetros y métricas siendo trackeados
- [ ] Modelos guardándose y versionándose
- [ ] Integración funcionando con DVC

#### **Fase 3 - Dashboard**
- [ ] Dashboard ejecutándose localmente
- [ ] Todas las secciones funcionando correctamente
- [ ] Visualizaciones mostrando datos reales
- [ ] Integración con MLflow operativa

#### **Fase 4 - Documentación**
- [ ] README completo y actualizado
- [ ] Docstrings en todos los módulos
- [ ] Guías de uso claras y completas
- [ ] Ejemplos ejecutables

### **5.2 Entregables Finales**
- [ ] **Repositorio funcional**: Pipeline completo ejecutándose
- [ ] **Dashboard operativo**: Visualización de resultados
- [ ] **Documentación completa**: Guías de uso y desarrollo
- [ ] **Experimentos versionados**: Historial MLflow con resultados
- [ ] **Modelos entrenados**: Clasificador y generador funcionales

---

## 🚨 **RIESGOS IDENTIFICADOS Y MITIGACIONES**

### **Riesgo 1: Datos Limitados (27% anotado)**
- **Mitigación**: Enfoques semi-supervisados, data augmentation, transfer learning

### **Riesgo 2: Recursos Computacionales**
- **Mitigación**: Modelos pequeños, LoRA, fine-tuning parcial

### **Riesgo 3: Calidad de Anotaciones**
- **Mitigación**: Validación manual de muestras, filtros de calidad

### **Riesgo 4: Integración MLflow-DVC**
- **Mitigación**: Pruebas incrementales, documentación de configuración

---

## 🎯 **PRÓXIMOS PASOS INMEDIATOS**

1. **Validar este plan** con el equipo
2. **Priorizar fases** según recursos disponibles
3. **Definir milestones** específicos por semana
4. **Asignar responsabilidades** por componente
5. **Iniciar implementación** de Fase 1

---
