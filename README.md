# 🏥 Plain Language Summarizer (PLS)

[![CI/CD Pipeline](https://github.com/gabrielchaparro/pds-proyecto-final/actions/workflows/ci.yml/badge.svg)](https://github.com/gabrielchaparro/pds-proyecto-final/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![MLflow](https://img.shields.io/badge/MLflow-2.8.0-orange.svg)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-2.50.0-green.svg)](https://dvc.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red.svg)](https://streamlit.io/)

**Sistema de resumización médica en lenguaje sencillo para pacientes no expertos**

Proyecto final de la materia **Proyecto de Desarrollo de Soluciones** - Universidad de los Andes - 2025

---

## 🎯 Objetivo

Transformar textos médicos complejos en **resúmenes accesibles** para pacientes sin conocimientos técnicos especializados, utilizando modelos de inteligencia artificial avanzados y un dashboard interactivo para análisis.

## 🏗️ Arquitectura del Proyecto

```
pds-proyecto-final/
├── 🗂️ data/                     # Datos versionados con DVC
│   ├── raw/                    # Datos crudos médicos
│   ├── processed/              # Datasets limpios y splits
│   └── outputs/                # Resultados y predicciones
├── 🧠 src/                     # Código fuente modular
│   ├── data/                   # Procesamiento de datos
│   ├── models/                 # Modelos ML y PLS
│   ├── dashboard/              # Dashboard interactivo
│   └── utils/                  # Utilidades compartidas
├── 🤖 models/                  # Modelos entrenados
├── 📊 mlruns/                  # Experimentos MLflow
├── 🖥️ dashboard/               # Dashboard estático
├── 📚 docs/                    # Documentación completa
├── ⚙️ scripts/                 # Scripts de automatización
└── 🔧 Configuración (DVC, params, CI/CD)
```

## 🚀 Inicio Rápido

### Opción 1: Todo en uno (Recomendado)
```bash
# 1. Clonar y configurar
git clone https://github.com/gabrielchaparro/pds-proyecto-final.git
cd pds-proyecto-final

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Probar el sistema
python scripts/test_dashboard.py

# 4. Ejecutar dashboard completo
python scripts/run_dashboard.py
```

### Opción 2: Configuración manual
```bash
# Instalar dependencias
pip install -r requirements.txt

# Configurar DVC
dvc init

# Ejecutar pipeline completo
dvc repro

# Iniciar dashboard
streamlit run src/dashboard/app.py
```

### Opción 3: Solo modelos (sin dashboard)
```bash
# Ejecutar comparación de modelos
python scripts/compare_pls_models.py

# Ver resultados en MLflow
mlflow ui --port 5000
```

---

## 🤖 Modelos PLS Implementados

### 🎯 Modelos Disponibles

| Modelo | Tipo | Compresión | FKGL | Flesch | Tiempo | Estado |
|--------|------|------------|------|--------|--------|---------|
| **T5-Base** | Transformer | 0.292 | **12.2** | **39.0** | 3.64s | 🏆 **MEJOR** |
| **BART-Base** | Transformer | 0.306 | 14.6 | 21.6 | 2.37s | ✅ Bueno |
| **BART-Large-CNN** | Transformer | 0.277 | 14.5 | 19.6 | 5.90s | ✅ Bueno |
| **PLS Ligero** | Rule-based | 1.154 | 16.0 | 20.0 | 0.00s | ⚠️ Expande |

### 🏆 Modelo Recomendado: **T5-Base**
- ✅ **Mejor legibilidad**: FKGL más bajo (12.2)
- ✅ **Mayor facilidad de lectura**: Flesch más alto (39.0)
- ✅ **Compresión equilibrada**: 29.2%
- ✅ **Velocidad aceptable**: 3.64s

---

## 📊 Dashboard Interactivo

### 🎛️ Funcionalidades

#### 🏠 **Página de Inicio**
- Resumen ejecutivo del proyecto
- Navegación intuitiva por secciones
- Información general del sistema

#### 📊 **Sección de Datos**
- **Estadísticas del dataset**: 97,994 registros
- **Distribución PLS vs Non-PLS**: 50% cada clase
- **Análisis por fuente**: ClinicalTrials, Cochrane, Pfizer
- **Histogramas de longitud**: Distribución de textos
- **Métricas en tiempo real**

#### 🤖 **Sección de Modelos**
- **Tabla comparativa** de 4 modelos PLS
- **Gráficos de rendimiento** por métrica
- **Ranking automático** basado en score compuesto
- **Análisis de velocidad** de procesamiento
- **Métricas detalladas**: Compresión, FKGL, Flesch Reading Ease

#### 🔬 **Sección de Experimentos**
- **Navegador MLflow** integrado
- **Información detallada** por experimento
- **Lista de runs** con métricas
- **Timeline de experimentos**
- **Exploración interactiva**

#### 🧪 **Sección de Prueba en Vivo**
- **Generación interactiva** de PLS
- **Selección de modelo** dinámico
- **Ejemplos médicos** predefinidos
- **Métricas en tiempo real** de calidad
- **Historial de generaciones**
- **Comparación lado a lado**

### 🚀 Cómo usar el Dashboard

```bash
# Prueba rápida del sistema
python scripts/test_dashboard.py

# Ejecución completa
python scripts/run_dashboard.py

# Puerto personalizado
python scripts/run_dashboard.py --port 8080

# Sin navegador automático
python scripts/run_dashboard.py --no-browser
```

### 📈 Métricas de Evaluación

#### **Compresión**
- **Fórmula**: `palabras_PLS / palabras_originales`
- **Ideal**: 0.7 - 0.9 (reducción moderada)
- **Interpretación**: < 1.0 = comprime, > 1.0 = expande

#### **FKGL (Flesch-Kincaid Grade Level)**
- **Ideal**: < 8.0 (nivel elemental)
- **Interpretación**: Menor valor = más fácil de leer

#### **Flesch Reading Ease**
- **Ideal**: > 60 (fácil de leer)
- **Interpretación**: Mayor valor = más fácil de leer

---

## 🔬 Experimentos y Evaluación

### 📊 Experimentos en MLflow

```bash
# Ver experimentos disponibles
mlflow experiments search

# Experimento principal
pls_models_comparison (ID: 580102703641195907)

# Runs registrados:
# - pls_bart_base_20250906_205143
# - pls_bart_large_cnn_20250906_205200
# - pls_t5_base_20250906_205230
# - pls_pls_lightweight_20250906_205249
```

### 🎯 Resultados de Evaluación

| Aspecto | T5-Base | BART-Base | BART-Large | PLS Ligero |
|---------|---------|-----------|------------|------------|
| **Calidad PLS** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Velocidad** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Legibilidad** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Compresión** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |

---

## 🔧 Configuración y Desarrollo

### 📋 Requisitos del Sistema
- **Python**: 3.11+
- **RAM**: 8GB mínimo, 16GB recomendado
- **GPU**: Opcional (acelera modelos transformer)
- **Espacio**: 5GB para modelos y datos

### 📦 Dependencias Principales
```txt
streamlit>=1.28.0      # Dashboard
transformers>=4.20.0   # Modelos PLS
mlflow>=2.8.0         # Experiment tracking
dvc>=2.50.0           # Versionado de datos
plotly>=5.15.0        # Visualizaciones
pandas>=1.5.0         # Procesamiento de datos
```

### ⚙️ Configuración del Proyecto

#### **Archivo `params.yaml`**
```yaml
# Configuración centralizada
models:
  classifier:
    type: "tfidf_logistic_regression"
    tfidf:
      max_features: 10000
      ngram_range: [1, 2]

  generator:
    type: "t5_base"  # Modelo recomendado
    max_length: 100
    temperature: 0.8

evaluation:
  metrics:
    classification: ["accuracy", "f1_macro"]
    generation: ["rouge_l", "compression_ratio"]
```

#### **Pipeline DVC**
```yaml
# dvc.yaml - Pipeline reproducible
stages:
  make_dataset:
    cmd: python src/data/make_dataset.py
  split_dataset:
    cmd: python src/data/split_dataset.py
  train_classifier:
    cmd: python src/models/train_classifier.py
  evaluate_models:
    cmd: python src/models/evaluate.py
```

---

## 📚 Documentación

### 📖 Guías Disponibles
- `docs/diagnostico-inicial-pls.md` - Diagnóstico técnico completo
- `dashboard/README.md` - Guía específica del dashboard
- `docs/prototype.md` - Prototipo y arquitectura

### 🎯 Scripts de Automatización
- `scripts/run_dashboard.py` - Inicio rápido del dashboard
- `scripts/test_dashboard.py` - Verificación del sistema
- `scripts/compare_pls_models.py` - Comparación de modelos

---

## 👥 Equipo y Contribución


## Miembros del Equipo
- Erika Cárdenas: Documentación y testing
- Carlos Chaparro: Documentación y testing
- Jean Munevar: Desarrollo de modelos PLS, implementación del dashboard, evaluación y métricas
- Gabriela Munevar: Desarrollo de modelos PLS, implementación del dashboard, evaluación y métricas


## Contribuciones Individuales
- **Erika Cárdenas**: Encargada de la documentación técnica y pruebas de funcionalidad.  
- **Carlos Chaparro**: Encargado de la documentación técnica y pruebas de funcionalidad.  
- **Jean Munevar**: Responsable de la construcción de modelos PLS, diseño e implementación del dashboard, y análisis de métricas.  
- **Gabriela Munevar**: Responsable de la construcción de modelos PLS, diseño e implementación del dashboard, y análisis de métricas.  


### 📝 Cómo Contribuir
1. Fork el repositorio
2. Crear branch para feature: `git checkout -b feature/nueva-funcionalidad`
3. Commit cambios: `git commit -m 'feat: añadir nueva funcionalidad'`
4. Push y crear Pull Request

### 🐛 Reportar Problemas
- [GitHub Issues](https://github.com/gabrielchaparro/pds-proyecto-final/issues)
- Incluir: descripción, pasos para reproducir, logs de error

---

<div align="center">

**🏥 Desarrollado con ❤️ para mejorar la comprensión médica de los pacientes**

[![Made with Streamlit](https://img.shields.io/badge/Made%20with-Streamlit-red.svg)](https://streamlit.io/)
[![Powered by MLflow](https://img.shields.io/badge/Powered%20by-MLflow-orange.svg)](https://mlflow.org/)
[![Data Versioning with DVC](https://img.shields.io/badge/Data%20Versioning-DVC-green.svg)](https://dvc.org/)

**🌟 Dashboard PLS - Transformando la medicina compleja en lenguaje sencillo**

</div>
