# Makefile — Pipeline CI/CD Local
# Proyecto: Renovación de Crédito MLOps
# Uso: make <target>
# Uso completo: make all

.PHONY: all install lint test ingest validate-data build-features train evaluate validate-model register pipeline docker clean help

# ── Target por defecto: ejecutar todo en secuencia ────────────────────────────
all: lint test pipeline docker
	@echo ""
	@echo "✓ Pipeline CI/CD local completado. Listo para git push."

# ── Instalar dependencias ─────────────────────────────────────────────────────
install:
	@echo "=== Instalando dependencias ==="
	python3 -m pip install --upgrade pip
	python3 -m pip install -r requirements.txt
	@echo "Dependencias instaladas correctamente."

# ── Linting con flake8 ────────────────────────────────────────────────────────
lint:
	@echo "=== Linting con flake8 ==="
	flake8 src/ tests/ --config=setup.cfg
	@echo "Linting OK"

# ── Tests unitarios con pytest ────────────────────────────────────────────────
test:
	@echo "=== Tests con pytest ==="
	pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

# ── Ingesta de datos ──────────────────────────────────────────────────────────
ingest:
	@echo "=== Ingesta de datos ==="
	python src/ingest_data.py

# ── Validación de calidad de datos ────────────────────────────────────────────
validate-data:
	@echo "=== Validación de calidad de datos ==="
	python src/validate_data.py

# ── Construcción de features ──────────────────────────────────────────────────
build-features:
	@echo "=== Construcción de variables/features ==="
	python src/build_features.py

# ── Entrenamiento del modelo ──────────────────────────────────────────────────
train:
	@echo "=== Entrenamiento del modelo ==="
	python src/train_pipeline.py

# ── Evaluación del modelo ─────────────────────────────────────────────────────
evaluate:
	@echo "=== Evaluación del modelo ==="
	python src/evaluate_model.py

# ── Quality gate de métricas ──────────────────────────────────────────────────
validate-model:
	@echo "=== Validando métricas del modelo Quality Gate ==="
	python src/validate_model.py

# ── Registro del modelo en MLflow ─────────────────────────────────────────────
register:
	@echo "=== Registro del modelo en MLflow ==="
	python src/register_model.py

# ── Pipeline completo sin Docker ──────────────────────────────────────────────
pipeline: ingest validate-data build-features train evaluate validate-model register
	@echo ""
	@echo "✓ Pipeline ML completado correctamente."

# ── Build de imagen Docker ────────────────────────────────────────────────────
docker:
	@echo "=== Construyendo imagen Docker ==="
	docker build -t renovacion-credito-mlops:local .
	@echo "=== Smoke test de la imagen Docker ==="
	docker run --rm renovacion-credito-mlops:local python -c \
		"import joblib; m=joblib.load('artifacts/modelo.pkl'); print('Modelo OK:', type(m).__name__)"

# ── Limpiar artefactos generados ──────────────────────────────────────────────
clean:
	@echo "=== Limpiando artefactos y cachés ==="
	rm -rf artifacts/*.pkl
	rm -rf artifacts/*.json
	rm -rf artifacts/*.csv
	rm -rf reports/*.csv
	rm -rf reports/*.md
	rm -rf reports/figures/*.png
	rm -rf mlruns/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
	@echo "Limpieza completada."

# ── Ayuda ─────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "Targets disponibles:"
	@echo "  make install          — Instalar dependencias"
	@echo "  make lint             — Ejecutar linting con flake8"
	@echo "  make test             — Ejecutar tests con pytest"
	@echo "  make ingest           — Ejecutar ingesta de datos"
	@echo "  make validate-data    — Validar calidad de datos"
	@echo "  make build-features   — Construir variables/features"
	@echo "  make train            — Entrenar modelo"
	@echo "  make evaluate         — Evaluar modelo y generar métricas"
	@echo "  make validate-model   — Ejecutar quality gate de métricas"
	@echo "  make register         — Registrar modelo en MLflow"
	@echo "  make pipeline         — Ejecutar pipeline ML completo sin Docker"
	@echo "  make docker           — Construir imagen Docker y smoke test"
	@echo "  make all              — Ejecutar lint + test + pipeline + docker"
	@echo "  make clean            — Limpiar artefactos y cachés"
	@echo ""