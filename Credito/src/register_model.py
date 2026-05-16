"""
register_model.py — Registro del modelo de renovación de crédito en MLflow.

Este script:
1. Carga el modelo entrenado desde artifacts/modelo.pkl
2. Carga las métricas desde artifacts/metrics.json
3. Carga metadata del entrenamiento si existe
4. Registra parámetros, métricas y artefactos en MLflow
5. Registra el modelo en MLflow como modelo sklearn
6. Guarda información del registro en artifacts/registered_model_info.json
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

try:
    from mlflow.models import infer_signature
except ImportError:
    infer_signature = None

try:
    from src.config import ARTIFACTS_DIR, PROJECT_ROOT, TARGET
except ImportError:
    from config import ARTIFACTS_DIR, PROJECT_ROOT, TARGET


# ============================================================
# Configuración de logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# Rutas
# ============================================================

MLRUNS_DIR = PROJECT_ROOT / "mlruns"

MODEL_PATH = ARTIFACTS_DIR / "modelo.pkl"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
FEATURE_IMPORTANCE_PATH = ARTIFACTS_DIR / "feature_importance.csv"
MODEL_SIGNATURE_PATH = ARTIFACTS_DIR / "model_signature.json"
TRAIN_METADATA_PATH = ARTIFACTS_DIR / "training_metadata.json"
TRAIN_TEST_SPLIT_PATH = ARTIFACTS_DIR / "train_test_split.pkl"
REGISTERED_MODEL_INFO_PATH = ARTIFACTS_DIR / "registered_model_info.json"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


# ============================================================
# Configuración MLflow
# ============================================================

EXPERIMENT_NAME = "renovacion_credito_mlops"
REGISTERED_MODEL_NAME = "renovacion_credito_model"


# ============================================================
# Funciones auxiliares
# ============================================================

def crear_directorios() -> None:
    """
    Crea carpetas necesarias.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Carpeta artifacts verificada: %s", ARTIFACTS_DIR)
    logger.info("Carpeta mlruns verificada: %s", MLRUNS_DIR)


def validar_archivo_existe(path: Path, descripcion: str) -> None:
    """
    Valida que un archivo exista.

    Args:
        path: Ruta del archivo.
        descripcion: Descripción del archivo.

    Raises:
        FileNotFoundError: Si el archivo no existe.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró {descripcion}: {path}. "
            "Ejecuta primero el pipeline de entrenamiento y evaluación."
        )


def cargar_json(path: Path, descripcion: str) -> dict[str, Any]:
    """
    Carga un archivo JSON.

    Args:
        path: Ruta del archivo JSON.
        descripcion: Descripción del archivo.

    Returns:
        Diccionario con el contenido del JSON.
    """
    validar_archivo_existe(path, descripcion)

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    logger.info("%s cargado desde: %s", descripcion, path)

    return data


def cargar_modelo(path: Path = MODEL_PATH) -> Any:
    """
    Carga el modelo entrenado.

    Args:
        path: Ruta del modelo.

    Returns:
        Modelo cargado.
    """
    validar_archivo_existe(path, "modelo entrenado")

    modelo = joblib.load(path)

    logger.info("Modelo cargado desde: %s", path)
    logger.info("Tipo de objeto cargado: %s", type(modelo).__name__)

    return modelo


def convertir_a_json_serializable(obj: Any) -> Any:
    """
    Convierte objetos no serializables a tipos compatibles con JSON.

    Args:
        obj: Objeto a convertir.

    Returns:
        Objeto compatible con JSON.
    """
    if isinstance(obj, dict):
        return {
            str(key): convertir_a_json_serializable(value)
            for key, value in obj.items()
        }

    if isinstance(obj, list):
        return [convertir_a_json_serializable(value) for value in obj]

    if isinstance(obj, tuple):
        return [convertir_a_json_serializable(value) for value in obj]

    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj

    if isinstance(obj, float):
        if pd.isna(obj):
            return None
        return obj

    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass

    if hasattr(obj, "item"):
        return obj.item()

    return str(obj)


def es_numero(valor: Any) -> bool:
    """
    Verifica si un valor puede registrarse como métrica numérica en MLflow.

    Args:
        valor: Valor a evaluar.

    Returns:
        True si es numérico.
    """
    if isinstance(valor, bool):
        return False

    try:
        float(valor)
        return True
    except (TypeError, ValueError):
        return False


def extraer_metricas_mlflow(metricas: dict[str, Any]) -> dict[str, float]:
    """
    Extrae métricas numéricas principales para MLflow.

    Args:
        metricas: Diccionario completo de métricas.

    Returns:
        Diccionario de métricas numéricas.
    """
    metricas_mlflow = {}

    metricas_permitidas = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "cv_best_score",
        "threshold",
        "n_train",
        "n_test",
        "n_features",
    ]

    for nombre in metricas_permitidas:
        if nombre in metricas and es_numero(metricas[nombre]):
            metricas_mlflow[nombre] = float(metricas[nombre])

    return metricas_mlflow


def extraer_parametros_mlflow(
    metricas: dict[str, Any],
    training_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Extrae parámetros relevantes para registrar en MLflow.

    Args:
        metricas: Métricas del modelo.
        training_metadata: Metadata del entrenamiento.

    Returns:
        Diccionario de parámetros.
    """
    parametros = {
        "target": TARGET,
        "best_model": metricas.get("best_model", metricas.get("model_type", "unknown")),
        "model_type": metricas.get("model_type", "unknown"),
        "selection_metric": metricas.get("selection_metric", "unknown"),
        "random_state": metricas.get("random_state", "unknown"),
        "test_size": metricas.get("test_size", "unknown"),
    }

    params_modelo = metricas.get("params", {})

    if isinstance(params_modelo, dict):
        for key, value in params_modelo.items():
            parametros[f"param_{key}"] = value

    if training_metadata:
        parametros["rows"] = training_metadata.get("rows", "unknown")
        parametros["columns"] = training_metadata.get("columns", "unknown")
        parametros["n_features"] = training_metadata.get("n_features", "unknown")
        parametros["models_trained"] = ",".join(
            training_metadata.get("models_trained", [])
        )

    return convertir_a_json_serializable(parametros)


def cargar_split_para_signature(
    modelo: Any,
) -> tuple[pd.DataFrame | None, Any, Any]:
    """
    Carga una muestra de X_test para crear input_example y signature.

    Args:
        modelo: Modelo entrenado.

    Returns:
        input_example, signature, y_pred_sample.
    """
    if not TRAIN_TEST_SPLIT_PATH.exists():
        logger.warning(
            "No existe train_test_split.pkl. "
            "El modelo se registrará sin input_example ni signature."
        )
        return None, None, None

    split_data = joblib.load(TRAIN_TEST_SPLIT_PATH)

    if "X_test" not in split_data:
        logger.warning("train_test_split.pkl no contiene X_test.")
        return None, None, None

    X_test = split_data["X_test"]

    if not isinstance(X_test, pd.DataFrame):
        logger.warning("X_test no es DataFrame. No se genera signature.")
        return None, None, None

    input_example = X_test.head(5).copy()

    try:
        y_pred_sample = modelo.predict(input_example)
    except Exception as error:
        logger.warning("No se pudo predecir muestra para signature: %s", error)
        return input_example, None, None

    signature = None

    if infer_signature is not None:
        try:
            signature = infer_signature(input_example, y_pred_sample)
        except Exception as error:
            logger.warning("No se pudo inferir signature MLflow: %s", error)

    return input_example, signature, y_pred_sample


def registrar_metricas(metricas: dict[str, Any]) -> None:
    """
    Registra métricas numéricas en MLflow.

    Args:
        metricas: Diccionario de métricas.
    """
    metricas_mlflow = extraer_metricas_mlflow(metricas)

    for nombre, valor in metricas_mlflow.items():
        mlflow.log_metric(nombre, valor)

    logger.info("Métricas registradas en MLflow: %s", metricas_mlflow)


def registrar_parametros(parametros: dict[str, Any]) -> None:
    """
    Registra parámetros en MLflow.

    Args:
        parametros: Diccionario de parámetros.
    """
    for nombre, valor in parametros.items():
        if isinstance(valor, (dict, list, tuple)):
            valor = json.dumps(convertir_a_json_serializable(valor), ensure_ascii=False)

        mlflow.log_param(nombre, str(valor))

    logger.info("Parámetros registrados en MLflow.")


def registrar_artifactos() -> None:
    """
    Registra artefactos generados por el pipeline en MLflow.
    """
    archivos_artifacts = [
        METRICS_PATH,
        FEATURE_IMPORTANCE_PATH,
        MODEL_SIGNATURE_PATH,
        TRAIN_METADATA_PATH,
        TRAIN_TEST_SPLIT_PATH,
    ]

    for path in archivos_artifacts:
        if path.exists():
            mlflow.log_artifact(str(path), artifact_path="artifacts")
            logger.info("Artefacto registrado: %s", path)
        else:
            logger.warning("Artefacto no encontrado, se omite: %s", path)

    if REPORTS_DIR.exists():
        model_report = REPORTS_DIR / "model_report.md"
        classification_report = REPORTS_DIR / "classification_report.csv"

        for path in [model_report, classification_report]:
            if path.exists():
                mlflow.log_artifact(str(path), artifact_path="reports")
                logger.info("Reporte registrado: %s", path)

    if FIGURES_DIR.exists():
        for figure_path in FIGURES_DIR.glob("*.png"):
            mlflow.log_artifact(str(figure_path), artifact_path="reports/figures")
            logger.info("Figura registrada: %s", figure_path)


def guardar_info_registro(info: dict[str, Any]) -> None:
    """
    Guarda información del registro MLflow en artifacts/.

    Args:
        info: Diccionario con información del run.
    """
    info = convertir_a_json_serializable(info)

    with open(REGISTERED_MODEL_INFO_PATH, "w", encoding="utf-8") as file:
        json.dump(info, file, indent=4, ensure_ascii=False)

    logger.info(
        "Información del registro guardada en: %s",
        REGISTERED_MODEL_INFO_PATH,
    )


# ============================================================
# Función principal
# ============================================================

def registrar_modelo() -> dict[str, Any]:
    """
    Registra el modelo entrenado en MLflow.

    Returns:
        Información del registro realizado.
    """
    crear_directorios()

    modelo = cargar_modelo()
    metricas = cargar_json(METRICS_PATH, "métricas del modelo")

    training_metadata = None

    if TRAIN_METADATA_PATH.exists():
        training_metadata = cargar_json(
            TRAIN_METADATA_PATH,
            "metadata del entrenamiento",
        )
    else:
        logger.warning("No existe training_metadata.json. Se continúa sin metadata.")

    tracking_uri = f"file:{MLRUNS_DIR}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    logger.info("MLflow tracking URI: %s", tracking_uri)
    logger.info("MLflow experiment: %s", EXPERIMENT_NAME)

    parametros = extraer_parametros_mlflow(
        metricas=metricas,
        training_metadata=training_metadata,
    )

    input_example, signature, _ = cargar_split_para_signature(modelo)

    with mlflow.start_run(run_name="registro_modelo_renovacion_credito") as run:
        run_id = run.info.run_id
        experiment_id = run.info.experiment_id

        logger.info("MLflow run iniciado: %s", run_id)

        registrar_parametros(parametros)
        registrar_metricas(metricas)
        registrar_artifactos()

        mlflow.sklearn.log_model(
            sk_model=modelo,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        model_uri = f"runs:/{run_id}/model"

        info_registro = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "experiment_name": EXPERIMENT_NAME,
            "experiment_id": experiment_id,
            "run_id": run_id,
            "tracking_uri": tracking_uri,
            "artifact_uri": run.info.artifact_uri,
            "model_uri": model_uri,
            "registered_model_name": REGISTERED_MODEL_NAME,
            "model_path": str(MODEL_PATH),
            "metrics_path": str(METRICS_PATH),
            "best_model": metricas.get("best_model", metricas.get("model_type")),
            "metrics": {
                "accuracy": metricas.get("accuracy"),
                "precision": metricas.get("precision"),
                "recall": metricas.get("recall"),
                "f1": metricas.get("f1"),
                "roc_auc": metricas.get("roc_auc"),
            },
        }

        guardar_info_registro(info_registro)

        mlflow.log_artifact(
            str(REGISTERED_MODEL_INFO_PATH),
            artifact_path="artifacts",
        )

        logger.info("Modelo registrado correctamente en MLflow.")
        logger.info("Model URI: %s", model_uri)

    return info_registro


# ============================================================
# Ejecución desde terminal
# ============================================================

if __name__ == "__main__":
    try:
        resultado = registrar_modelo()

        print("Registro en MLflow finalizado correctamente.")
        print(json.dumps(convertir_a_json_serializable(resultado), indent=4))

    except Exception as error:
        logger.exception("Error durante el registro del modelo en MLflow: %s", error)
        sys.exit(1)