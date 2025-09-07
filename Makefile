# Makefile para el Proyecto PDS - Generador PLS
# Targets para facilitar el uso del generador PLS

.PHONY: help gen-zero-shot gen-lora-train gen-eval install-deps test-generator

# Variables
GENERATOR_SCRIPT = src/models/generate_pls_simple.py
CONFIG_FILE = params.generator.yaml
PYTHON = python

# Target por defecto
help:
	@echo "=== GENERADOR PLS - TARGETS DISPONIBLES ==="
	@echo ""
	@echo "INSTALACIÓN:"
	@echo "  install-deps        Instalar dependencias del generador"
	@echo ""
	@echo "GENERACIÓN PLS:"
	@echo "  gen-zero-shot       Generar PLS con zero-shot (BART-large-cnn)"
	@echo "  gen-lora-train      Fine-tuning LoRA (BART-base o T5-base)"
	@echo "  gen-eval            Evaluar modelo guardado"
	@echo ""
	@echo "TESTING:"
	@echo "  test-generator      Ejecutar generador con modo legacy"
	@echo ""
	@echo "CONFIGURACIÓN:"
	@echo "  - Archivo config: $(CONFIG_FILE)"
	@echo "  - Script: $(GENERATOR_SCRIPT)"
	@echo ""
	@echo "EJEMPLOS:"
	@echo "  make gen-zero-shot"
	@echo "  make gen-lora-train BASE_MODEL=bart-base"
	@echo "  make gen-eval MODEL_PATH=path/to/model"
	@echo ""

# Instalar dependencias del generador
install-deps:
	@echo "Instalando dependencias del generador PLS..."
	pip install peft>=0.4.0 accelerate>=0.20.0 evaluate>=0.4.0 bert-score>=0.3.13 mlflow>=2.8.0 typer>=0.9.0
	@echo "Dependencias instaladas. Verificar instalación:"
	@echo "  python -c 'import typer, peft, accelerate, evaluate, bert_score, mlflow; print(\"OK\")'"

# Generar PLS con zero-shot
gen-zero-shot:
	@echo "=== GENERANDO PLS CON ZERO-SHOT ==="
	@if [ ! -f $(CONFIG_FILE) ]; then \
		echo "ERROR: Archivo de configuración $(CONFIG_FILE) no encontrado"; \
		exit 1; \
	fi
	$(PYTHON) $(GENERATOR_SCRIPT) zero-shot --config $(CONFIG_FILE)
	@echo ""
	@echo "Comando ejecutado: $(PYTHON) $(GENERATOR_SCRIPT) zero-shot --config $(CONFIG_FILE)"

# Fine-tuning LoRA
gen-lora-train:
	@echo "=== FINE-TUNING LORA ==="
	@if [ ! -f $(CONFIG_FILE) ]; then \
		echo "ERROR: Archivo de configuración $(CONFIG_FILE) no encontrado"; \
		exit 1; \
	fi
	@if [ -z "$(BASE_MODEL)" ]; then \
		echo "ERROR: Especificar BASE_MODEL (bart-base o t5-base)"; \
		echo "Ejemplo: make gen-lora-train BASE_MODEL=bart-base"; \
		exit 1; \
	fi
	$(PYTHON) $(GENERATOR_SCRIPT) finetune-lora --config $(CONFIG_FILE) --base $(BASE_MODEL)
	@echo ""
	@echo "Comando ejecutado: $(PYTHON) $(GENERATOR_SCRIPT) finetune-lora --config $(CONFIG_FILE) --base $(BASE_MODEL)"

# Evaluar modelo guardado
gen-eval:
	@echo "=== EVALUANDO MODELO ==="
	@if [ ! -f $(CONFIG_FILE) ]; then \
		echo "ERROR: Archivo de configuración $(CONFIG_FILE) no encontrado"; \
		exit 1; \
	fi
	@if [ -z "$(MODEL_PATH)" ]; then \
		echo "ERROR: Especificar MODEL_PATH"; \
		echo "Ejemplo: make gen-eval MODEL_PATH=artifacts/generator/run_id/model"; \
		exit 1; \
	fi
	$(PYTHON) $(GENERATOR_SCRIPT) evaluate --config $(CONFIG_FILE) --model-path $(MODEL_PATH)
	@echo ""
	@echo "Comando ejecutado: $(PYTHON) $(GENERATOR_SCRIPT) evaluate --config $(CONFIG_FILE) --model-path $(MODEL_PATH)"

# Modo de testing legacy (sin argumentos CLI)
test-generator:
	@echo "=== MODO TEST LEGACY ==="
	@echo "Ejecutando generador con función main() original..."
	$(PYTHON) $(GENERATOR_SCRIPT)
	@echo ""
	@echo "Comando ejecutado: $(PYTHON) $(GENERATOR_SCRIPT)"

# ===== TARGETS ADICIONALES =====

# Crear directorios necesarios
setup-dirs:
	@echo "Creando directorios necesarios..."
	mkdir -p artifacts/generator
	mkdir -p logs
	mkdir -p data/outputs
	@echo "Directorios creados"

# Limpiar artifacts generados
clean-artifacts:
	@echo "Limpiando artifacts generados..."
	rm -rf artifacts/generator/*
	@echo "Artifacts limpiados"

# Mostrar configuración actual
show-config:
	@echo "=== CONFIGURACIÓN ACTUAL ==="
	@if [ -f $(CONFIG_FILE) ]; then \
		head -20 $(CONFIG_FILE); \
		echo "..."; \
	else \
		echo "Archivo $(CONFIG_FILE) no encontrado"; \
	fi

# Verificar instalación
check-install:
	@echo "=== VERIFICANDO INSTALACIÓN ==="
	@echo "Python version: $$(python --version)"
	@echo "Verificando dependencias críticas..."
	@python -c "import sys; print('Python path:', sys.executable)"
	@python -c "import pandas, numpy, torch, transformers; print('Core deps: OK')" 2>/dev/null || echo "ERROR: Core dependencies faltantes"
	@python -c "import typer, peft, accelerate, evaluate, bert_score, mlflow; print('Generator deps: OK')" 2>/dev/null || echo "ADVERTENCIA: Generator dependencies faltantes (ejecutar: make install-deps)"
	@echo ""
	@echo "Si hay errores, ejecutar: make install-deps"

# ===== TARGETS DE DESARROLLO =====

# Ejecutar con subset pequeño para testing rápido
test-zero-shot-fast:
	@echo "=== TEST RÁPIDO ZERO-SHOT (SUBSET: 50 EJEMPLOS) ==="
	@if [ ! -f $(CONFIG_FILE) ]; then \
		echo "ERROR: Archivo de configuración $(CONFIG_FILE) no encontrado"; \
		exit 1; \
	fi
	# Crear config temporal con subset pequeño
	cp $(CONFIG_FILE) $(CONFIG_FILE).backup
	sed 's/eval_subset_size: [0-9]*/eval_subset_size: 50/' $(CONFIG_FILE) > $(CONFIG_FILE).tmp && mv $(CONFIG_FILE).tmp $(CONFIG_FILE)
	$(PYTHON) $(GENERATOR_SCRIPT) zero-shot --config $(CONFIG_FILE)
	# Restaurar config original
	mv $(CONFIG_FILE).backup $(CONFIG_FILE)
	@echo ""
	@echo "✅ Test completado con subset de 50 ejemplos"
	@echo "📊 Revisar artifacts/generator/<run_id>/ para resultados"
	@echo "🔬 Revisar MLflow UI para métricas detalladas"

# Ejecutar validación completa del sistema
validate:
	@echo "=== VALIDACIÓN COMPLETA DEL SISTEMA ==="
	$(PYTHON) $(GENERATOR_SCRIPT) validate

# Setup completo para desarrollo
setup-dev: setup-dirs install-deps check-install validate
	@echo ""
	@echo "🎉 SETUP COMPLETADO"
	@echo "Próximos pasos:"
	@echo "  make test-zero-shot-fast  # Prueba rápida"
	@echo "  make gen-zero-shot        # Generación completa"

# Ejecutar linting básico
lint:
	@echo "=== LINTING BÁSICO ==="
	python -m py_compile $(GENERATOR_SCRIPT)
	@echo "Sintaxis OK"
	python -c "import yaml; yaml.safe_load(open('$(CONFIG_FILE)')); print('YAML config OK')"

# ===== DOCUMENTACIÓN =====

# Mostrar ayuda detallada
help-detailed:
	@echo "=== GENERADOR PLS - GUÍA DETALLADA ==="
	@echo ""
	@echo "ARCHIVOS IMPORTANTES:"
	@echo "  $(CONFIG_FILE)          Configuración del generador"
	@echo "  $(GENERATOR_SCRIPT)     Script principal"
	@echo "  requirements.txt        Dependencias del proyecto"
	@echo ""
	@echo "DIRECTORIOS:"
	@echo "  artifacts/generator/    Outputs del generador"
	@echo "  data/processed/         Datasets procesados"
	@echo "  mlruns/                 Experimentos MLflow"
	@echo ""
	@echo "FLUJO TÍPICO:"
	@echo "  1. make check-install    # Verificar instalación"
	@echo "  2. make gen-zero-shot    # Probar zero-shot"
	@echo "  3. make gen-lora-train   # Fine-tuning (opcional)"
	@echo "  4. make gen-eval         # Evaluar modelo"
	@echo ""
	@echo "CONFIGURACIÓN:"
	@echo "  - Editar $(CONFIG_FILE) para cambiar parámetros"
	@echo "  - Modelo por defecto: BART-large-cnn (zero-shot)"
	@echo "  - Subset por defecto: 500 ejemplos"
	@echo ""
	@echo "MLFLOW:"
	@echo "  - Experimento: generator_pls"
	@echo "  - Tags: phase=F1, task=generator"
	@echo "  - Artifacts: métricas, samples, config"
	@echo ""
	@echo "SOLUCIÓN DE PROBLEMAS:"
	@echo "  - 'Module not found': make install-deps"
	@echo "  - 'Config not found': verificar $(CONFIG_FILE)"
	@echo "  - 'CUDA error': verificar GPU disponible"
	@echo "  - 'Memory error': reducir eval_subset_size"
	@echo ""

# Target alias para compatibilidad
zero-shot: gen-zero-shot
lora-train: gen-lora-train
evaluate: gen-eval
