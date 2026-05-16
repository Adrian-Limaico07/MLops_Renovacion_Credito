# Dockerfile — Modelo de Renovación de Crédito MLOps
# Usado por GitHub Actions y por make docker

FROM python:3.12-slim

LABEL maintainer="MLOps Renovacion Credito"
LABEL description="Imagen Docker del pipeline/modelo de renovación de crédito"
LABEL version="1.0.0"

# Evita archivos .pyc y fuerza salida inmediata en logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Carpeta de trabajo dentro del contenedor
WORKDIR /app

# Instalar dependencias del sistema
# libgomp1 es importante para librerías como xgboost, scikit-learn y numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero para aprovechar cache de Docker
COPY requirements.txt .

# Instalar dependencias Python
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY src/ ./src/

# Copiar artefactos generados por el pipeline
# Deben existir después de ejecutar:
# python src/train_pipeline.py
# python src/evaluate_model.py
COPY artifacts/ ./artifacts/

# Copiar reportes si existen
COPY reports/ ./reports/

# Crear carpetas por seguridad
RUN mkdir -p artifacts reports reports/figures mlruns data

# Smoke test durante build:
# Si existe modelo.pkl, intenta cargarlo.
# Si no existe, no rompe la construcción.
RUN if [ -f "artifacts/modelo.pkl" ]; then \
        python -c "import joblib; m = joblib.load('artifacts/modelo.pkl'); print('Modelo cargado OK:', type(m).__name__)"; \
    else \
        echo 'WARNING: artifacts/modelo.pkl no existe en build. Ejecuta primero el entrenamiento.'; \
    fi

# Puerto opcional si luego expones una API
EXPOSE 8000

# Comando por defecto:
# Verifica que el modelo pueda cargarse dentro del contenedor
CMD ["python", "-c", "import joblib; m = joblib.load('artifacts/modelo.pkl'); print('Modelo listo:', type(m).__name__)"]