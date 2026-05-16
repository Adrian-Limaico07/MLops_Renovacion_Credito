# MLops_Renovacion_Credito
Proyecto de Renovación de Crédito aplicando MLops.

# **Autores:**

## ADRIAN ENRIQUE LIMAICO JARA
## OMAR ANTONIO PINOS GUILLEN


# **DESCRIPCIÓN DEL CASO**

**Renovación de préstamo**

Desde varios meses atrás una financiera ha identificado que existe una oportunidad comercial de renovar el préstamo a los clientes. Y esto resulta ser más  exitosos si se llega antes que las financieras competidoras.
Por ello a partir, de la información histórica de llamadas de renovación de prestamos se quiere identificar el perfil de los clientes más interesados en una renovación.


Para lograr esta labor se ha provisto a los analistas de Marketing información  histórica de 2 meses de ventas vía el call e información de la entidad financiera de los clientes


## ⚙️ Requisitos Previos

- Python 3.10+
- Cuenta GitHub
- Docker
- VS Code
- Conocimientos básicos de Python, Pandas y Scikit-learn

---

## 📁 Estructura del Repositorio

```

renovacion-credito-mlops/
├── .github/
│   └── workflows/
│       └── ml_pipeline.yml                  ← CI/CD: lint + tests + train + validate + docker
│
├── data/
│
├── src/
│   ├── __init__.py
│   ├── config.py                            ← Parámetros generales del proyecto
│   ├── ingest_data.py                       ← Ingesta de datos
│   ├── validate_data.py                     ← Validación de calidad de datos
│   ├── build_features.py                    ← Limpieza, imputación, logs, one-hot
│   ├── train_pipeline.py                    ← Entrenamiento: train/test + balanceo + GridSearchCV
│   ├── evaluate_model.py                    ← Métricas: accuracy, precision, recall, F1, AUC
│   ├── register_model.py                    ← Registro del modelo en MLflow
│   └── validate_model.py                    ← Quality gate de métricas
│
│
├── tests/
│   ├── __init__.py
│   ├── test_ingest_data.py                  ← Tests de carga e integración de fuentes
│   ├── test_validate_data.py                ← Tests de calidad de datos
│   ├── test_build_features.py               ← Tests de variables, imputaciones y encoding
│   ├── test_train_pipeline.py               ← Tests del entrenamiento
│   └── test_model.py                        ← Tests del modelo serializado
│
├── artifacts/
│   ├── modelo.pkl                           ← Modelo entrenado serializado
│   ├── preprocessor.pkl                     ← Transformaciones usadas en producción
│   ├── metrics.json                         ← Métricas del modelo
│   ├── feature_importance.csv               ← Importancia de variables
│   └── model_signature.json                 ← Firma de entrada/salida del modelo
│
├── mlruns/                                  ← Experimentos MLflow
│
├── reports/
│   ├── figures/
│   │   ├── matriz_confusion.png
│   │   ├── curva_roc.png
│   │   ├── importancia_variables.png
│   │   └── distribucion_score.png
│   │
│   └── model_report.md                      ← Reporte técnico del modelo
│
├── Dockerfile                               ← Imagen Docker del pipeline/modelo
├── Makefile                                 ← Ejecutar pipeline localmente
├── requirements.txt                         ← Dependencias Python
├── setup.cfg                                ← Configuración flake8 + pytest
├── .gitignore                               ← Ignora artifacts, mlruns, data sensible, cachés
└── README.md                                ← Descripción del proyecto y badge CI/CD

```
