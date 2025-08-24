# pds-proyecto-final

Proyecto final de la materia Proyecto de desarrollo de soluciones - Uniandes - 2025

## 📁 Estructura del Proyecto

```
├─ data/                    # Datos del proyecto
│  ├─ raw/                 # TXT originales sin modificar
│  ├─ processed/           # CSV/JSON limpios y normalizados
│  ├─ outputs/             # Resúmenes generados por modelos
│  └─ evaluation/          # Métricas, tablas y figuras de evaluación
├─ notebooks/              # Jupyter notebooks para análisis
│  └─ 01_eda.ipynb        # Análisis exploratorio de datos
├─ src/                    # Código fuente del proyecto
│  ├─ data/               # Scripts de procesamiento de datos
│  │  └─ make_dataset.py  # Creación y limpieza de datasets
│  ├─ models/             # Scripts de modelos y ML
│  │  ├─ train_classifier.py  # Entrenamiento del clasificador
│  │  ├─ generate_pls.py      # Generación de resúmenes PLS
│  │  └─ evaluate.py          # Evaluación de modelos
│  └─ utils/              # Funciones auxiliares
│     └─ io.py            # Utilidades de entrada/salida
├─ docs/                   # Documentación del proyecto
│  ├─ prototype.md         # Maqueta y diagrama del prototipo
│  └─ report_assets/       # Imágenes para el reporte
├─ dvc.yaml               # Pipeline de DVC para reproducibilidad
├─ params.yaml            # Parámetros configurables del proyecto
├─ requirements.txt        # Dependencias de Python
└─ README.md              # Este archivo
```

## 🚀 Inicio Rápido

1. **Instalar dependencias**: `pip install -r requirements.txt`
2. **Configurar DVC**: `dvc init` (si no está configurado)
3. **Ejecutar pipeline**: `dvc repro` para procesar datos y entrenar modelos

## 📋 Descripción

Este proyecto implementa un sistema de generación automática de resúmenes (PLS) utilizando técnicas de machine learning y procesamiento de lenguaje natural. La estructura está organizada siguiendo las mejores prácticas para proyectos de ML reproducibles y mantenibles.

## 🔄 Pipeline de Datos

El flujo de trabajo está automatizado con DVC:
- **prepare**: Preprocesamiento de datos
- **train**: Entrenamiento del clasificador
- **generate**: Generación de resúmenes PLS
- **evaluate**: Evaluación del rendimiento

Para más detalles, consulta `docs/prototype.md`.
