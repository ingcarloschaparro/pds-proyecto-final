# EVIDENCIA DEL PIPELINE Y DATOS VERSIONADOS - PLS PROJECT

## 1. PIPELINE DE DATOS CON DVC

**Configuración del Pipeline (`dvc.yaml`):**
```yaml
stages:
  manifest:     # Construcción del manifiesto de datos raw
  preprocess:   # Preprocesamiento de 182K+ registros médicos
  split:        # División train/val/test (70/15/15)
  train:        # Entrenamiento de 4 modelos PLS
  generate:     # Generación de resúmenes
  evaluate:     # Evaluación con métricas completas
```

**Datos Versionados:**
- **Datos Raw:** 207.6 MB, 65,175 archivos médicos (Hash: 10850e4de407fe3bae5619e5c8dccbc6)
- **Dataset Procesado:** 73.9 MB, 182,753 registros (Hash: a87cb05bc4be0b004009baa9c0bc03d9)
- **Splits:** Train (70%), Test (15%), Validación (15%) con hashes MD5 únicos

## 2. PIPELINE DE ENTRENAMIENTO

**Scripts Principales:**
- `scripts/compare_pls_models.py` - Entrenamiento de 4 modelos PLS
- `src/models/generate_pls_simple.py` - Generación de resúmenes
- `src/models/evaluate_simple.py` - Evaluación automática

**Comandos Makefile:**
```bash
make gen-zero-shot       # Generación con zero-shot
make gen-lora-train      # Fine-tuning LoRA
make gen-eval            # Evaluación de modelo
make validate            # Validación completa
```

## 3. ARTEFACTOS DE DESPLIEGUE

**Docker:**
- `Dockerfile` - Imagen principal
- `Dockerfile.base` - Imagen base optimizada
- `start-app.sh` - Script de inicio (API puerto 9000, Dashboard puerto 8501)

**Terraform:**
- ECR para imágenes Docker
- ECS para despliegue de contenedores
- ALB para balanceo de carga
- S3 para almacenamiento

**API y Dashboard:**
- `src/api/` - API REST con FastAPI
- `src/dashboard/` - Dashboard Streamlit con 5 secciones
- Colección Postman para pruebas

## 4. FLUJO COMPLETO

```
Datos Raw (65K archivos) → DVC → Preprocesamiento (182K registros) → DVC → 
División Train/Val/Test → DVC → Entrenamiento 4 Modelos → MLflow → 
Evaluación → Docker → AWS ECS → API + Dashboard
```

**Comandos de Ejecución:**
```bash
dvc repro                                    # Pipeline de datos
make gen-zero-shot                          # Entrenar modelos
terraform -chdir=terraform/stacks/ecs apply # Desplegar en AWS
```

## 5. EVIDENCIA DE VERSIONADO

**Archivos DVC:**
- `dvc.yaml` - Configuración del pipeline
- `dvc.lock` - Estado actual versionado
- `data/processed/*.dvc` - Archivos de control de datos

**Verificación:**
```bash
dvc status    # Estado del pipeline
dvc dag       # Visualizar dependencias
```

## 6. INTEGRACIÓN MLFLOW

**Experimentos:**
- 8 experimentos locales en `mlruns/`
- 6 experimentos remotos en AWS EC2 (52.0.127.25:5001)
- Métricas: ROUGE, FKGL, Flesch, Compresión, Tiempo

**Configuración:**
- Local: `file:./mlruns`
- Remoto: `http://52.0.127.25:5001`
- Experimento: `pls_models_comparison`

## 7. CONCLUSIÓN

El proyecto implementa un pipeline completo de datos versionados con DVC que garantiza reproducibilidad, trazabilidad y automatización desde datos raw hasta producción en AWS.
