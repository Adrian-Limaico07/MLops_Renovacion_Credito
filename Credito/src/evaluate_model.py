"""
evaluate_model.py — Evaluación final del modelo de renovación de crédito.

Este script:
1. Carga artifacts/modelo.pkl
2. Carga artifacts/train_test_split.pkl
3. Calcula métricas: accuracy, precision, recall, F1 y ROC AUC
4. Genera matriz de confusión
5. Genera curva ROC
6. Genera distribución de scores
7. Actualiza artifacts/metrics.json
8. Genera reports/model_report.md
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

try:
    from src.config import (
        ARTIFACTS_DIR,
        F1_MIN,
        FIGURES_DIR,
        RANDOM_STATE,
        RECALL_MIN,
        REPORTS_DIR,
        ROC_AUC_MIN,
        TARGET,
        TEST_SIZE,
    )
except ImportError:
    from config import (
        ARTIFACTS_DIR,
        F1_MIN,
        FIGURES_DIR,
        RANDOM_STATE,
        RECALL_MIN,
        REPORTS_DIR,
        ROC_AUC_MIN,
        TARGET,
        TEST_SIZE,
    )


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


MODEL_PATH = ARTIFACTS_DIR / "modelo.pkl"
TRAIN_TEST_SPLIT_PATH = ARTIFACTS_DIR / "train_test_split.pkl"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

MODEL_REPORT_PATH = REPORTS_DIR / "model_report.md"
CLASSIFICATION_REPORT_PATH = REPORTS_DIR / "classification_report.csv"

CONFUSION_MATRIX_PATH = FIGURES_DIR / "matriz_confusion.png"
ROC_CURVE_PATH = FIGURES_DIR / "curva_roc.png"
SCORE_DISTRIBUTION_PATH = FIGURES_DIR / "distribucion_score.png"

THRESHOLD = 0.50


def crear_directorios() -> None:
    """Crea las carpetas necesarias para guardar reportes y figuras."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def convertir_a_json_serializable(obj: Any) -> Any:
    """Convierte objetos numpy/pandas a formatos compatibles con JSON."""
    if isinstance(obj, dict):
        return {str(k): convertir_a_json_serializable(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [convertir_a_json_serializable(v) for v in obj]

    if isinstance(obj, tuple):
        return [convertir_a_json_serializable(v) for v in obj]

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


def validar_archivo_existe(path: Path, descripcion: str) -> None:
    """Valida que exista un archivo requerido."""
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró {descripcion}: {path}. "
            "Ejecuta primero python src/train_pipeline.py"
        )


def cargar_modelo(path: Path = MODEL_PATH) -> Any:
    """Carga el modelo entrenado."""
    validar_archivo_existe(path, "el modelo entrenado")

    modelo = joblib.load(path)

    logger.info("Modelo cargado desde: %s", path)
    logger.info("Tipo de objeto cargado: %s", type(modelo).__name__)

    return modelo


def cargar_train_test_split(path: Path = TRAIN_TEST_SPLIT_PATH) -> dict[str, Any]:
    """Carga la partición train/test generada por train_pipeline.py."""
    validar_archivo_existe(path, "la partición train/test")

    split_data = joblib.load(path)

    claves_requeridas = {"X_train", "X_test", "y_train", "y_test"}
    faltantes = claves_requeridas - set(split_data.keys())

    if faltantes:
        raise ValueError(
            f"El archivo {path} no contiene las claves requeridas: {faltantes}"
        )

    logger.info("Train/test split cargado desde: %s", path)
    logger.info("X_test: %d filas x %d columnas", *split_data["X_test"].shape)

    return split_data


def obtener_probabilidades(modelo: Any, X: pd.DataFrame) -> np.ndarray:
    """Obtiene la probabilidad estimada para la clase positiva."""
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(X)[:, 1]

    if hasattr(modelo, "decision_function"):
        scores = modelo.decision_function(X)
        return 1 / (1 + np.exp(-scores))

    logger.warning(
        "El modelo no tiene predict_proba ni decision_function. "
        "Se usarán predicciones binarias como score."
    )

    return modelo.predict(X)


def predecir_con_umbral(y_score: np.ndarray, threshold: float = THRESHOLD) -> np.ndarray:
    """Convierte probabilidades en clases 0/1."""
    return (y_score >= threshold).astype(int)


def calcular_metricas(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, Any]:
    """Calcula métricas principales del modelo."""
    metricas = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_score), 4),
        "threshold": THRESHOLD,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "target": TARGET,
        "recall_minimo": RECALL_MIN,
        "roc_auc_minimo": ROC_AUC_MIN,
        "f1_minimo": F1_MIN,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
    }

    logger.info("Métricas calculadas: %s", metricas)

    return metricas


def cargar_metricas_previas(path: Path = METRICS_PATH) -> dict[str, Any]:
    """Carga métricas previas si existen, por ejemplo best_model y params."""
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def guardar_metricas(metricas: dict[str, Any], path: Path = METRICS_PATH) -> None:
    """Guarda métricas finales en artifacts/metrics.json."""
    metricas_previas = cargar_metricas_previas(path)

    metricas_finales = {
        **metricas_previas,
        **metricas,
    }

    metricas_finales = convertir_a_json_serializable(metricas_finales)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(metricas_finales, file, indent=4, ensure_ascii=False)

    logger.info("Métricas guardadas en: %s", path)


def guardar_classification_report(metricas: dict[str, Any]) -> None:
    """Guarda classification_report en CSV."""
    report = metricas.get("classification_report", {})

    if not report:
        logger.warning("No existe classification_report para guardar.")
        return

    df_report = pd.DataFrame(report).T
    df_report.to_csv(CLASSIFICATION_REPORT_PATH, index=True)

    logger.info("Classification report guardado en: %s", CLASSIFICATION_REPORT_PATH)


def graficar_matriz_confusion(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> None:
    """Guarda la matriz de confusión."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title("Matriz de confusión")
    plt.xlabel("Predicción")
    plt.ylabel("Valor real")
    plt.xticks([0, 1], ["No venta (0)", "Venta (1)"])
    plt.yticks([0, 1], ["No venta (0)", "Venta (1)"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.close()

    logger.info("Matriz de confusión guardada en: %s", CONFUSION_MATRIX_PATH)


def graficar_curva_roc(
    y_true: pd.Series,
    y_score: np.ndarray,
) -> None:
    """Guarda la curva ROC."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("Curva ROC")
    plt.xlabel("Tasa de falsos positivos")
    plt.ylabel("Tasa de verdaderos positivos")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(ROC_CURVE_PATH)
    plt.close()

    logger.info("Curva ROC guardada en: %s", ROC_CURVE_PATH)


def graficar_distribucion_score(
    y_true: pd.Series,
    y_score: np.ndarray,
) -> None:
    """Guarda la distribución de probabilidades estimadas."""
    df_scores = pd.DataFrame(
        {
            "y_true": y_true.values,
            "score": y_score,
        }
    )

    plt.figure(figsize=(8, 5))

    for clase in sorted(df_scores["y_true"].unique()):
        subset = df_scores[df_scores["y_true"] == clase]
        plt.hist(
            subset["score"],
            bins=30,
            alpha=0.6,
            label=f"Clase {clase}",
        )

    plt.axvline(
        THRESHOLD,
        linestyle="--",
        label=f"Umbral = {THRESHOLD:.2f}",
    )

    plt.title("Distribución de probabilidades estimadas")
    plt.xlabel("Probabilidad estimada de FLAG_VENTA = 1")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.tight_layout()
    plt.savefig(SCORE_DISTRIBUTION_PATH)
    plt.close()

    logger.info("Distribución de scores guardada en: %s", SCORE_DISTRIBUTION_PATH)


def obtener_tipo_modelo(modelo: Any) -> str:
    """Obtiene el tipo de modelo final dentro del pipeline."""
    if hasattr(modelo, "named_steps"):
        estimador = modelo.named_steps.get("model")
        if estimador is not None:
            return type(estimador).__name__

    return type(modelo).__name__


def generar_model_report(
    metricas: dict[str, Any],
    n_train: int,
    n_test: int,
    n_features: int,
    model_type: str,
) -> str:
    """
    Genera reporte técnico en Markdown.

    Args:
        metricas: Diccionario con métricas calculadas.
        n_train: Número de filas de entrenamiento.
        n_test: Número de filas de prueba.
        n_features: Número de variables predictoras.
        model_type: Tipo de modelo final.

    Returns:
        Texto del reporte técnico en formato Markdown.
    """
    cm = metricas["confusion_matrix"]

    reporte = (
        "# Reporte técnico del modelo - Renovación de crédito\n\n"
        "## 1. Objetivo\n\n"
        "El objetivo del modelo es estimar la probabilidad de que un cliente "
        "acepte o concrete una renovación de préstamo.\n\n"
        f"Variable objetivo: `{TARGET}`.\n\n"
        "## 2. Configuración de evaluación\n\n"
        "| Elemento | Valor |\n"
        "|---|---:|\n"
        f"| Modelo | `{model_type}` |\n"
        f"| Filas de entrenamiento | {n_train} |\n"
        f"| Filas de prueba | {n_test} |\n"
        f"| Variables predictoras | {n_features} |\n"
        f"| Tamaño de test | {TEST_SIZE} |\n"
        f"| Semilla aleatoria | {RANDOM_STATE} |\n"
        f"| Umbral de clasificación | {THRESHOLD} |\n\n"
        "## 3. Métricas principales\n\n"
        "| Métrica | Valor |\n"
        "|---|---:|\n"
        f"| Accuracy | {metricas['accuracy']:.4f} |\n"
        f"| Precision | {metricas['precision']:.4f} |\n"
        f"| Recall | {metricas['recall']:.4f} |\n"
        f"| F1-score | {metricas['f1']:.4f} |\n"
        f"| ROC AUC | {metricas['roc_auc']:.4f} |\n\n"
        "## 4. Umbrales mínimos definidos\n\n"
        "| Métrica | Umbral mínimo |\n"
        "|---|---:|\n"
        f"| Recall mínimo | {RECALL_MIN:.4f} |\n"
        f"| ROC AUC mínimo | {ROC_AUC_MIN:.4f} |\n"
        f"| F1 mínimo | {F1_MIN:.4f} |\n\n"
        "## 5. Matriz de confusión\n\n"
        "La matriz de confusión tiene la siguiente forma:\n\n"
        "[[TN, FP],\n"
        " [FN, TP]]\n\n"
        "Resultado obtenido:\n\n"
        f"{cm}\n\n"
        "## 6. Interpretación\n\n"
        "- **Accuracy**: porcentaje total de aciertos del modelo.\n"
        "- **Precision**: de los clientes predichos como venta, cuántos realmente fueron venta.\n"
        "- **Recall**: de las ventas reales, cuántas logró detectar el modelo.\n"
        "- **F1-score**: balance entre precision y recall.\n"
        "- **ROC AUC**: capacidad general del modelo para separar clientes con y sin venta.\n\n"
        "En este problema, el **recall** es importante porque ayuda a detectar "
        "más clientes con probabilidad de renovar.\n\n"
        "## 7. Archivos generados\n\n"
        "- `artifacts/metrics.json`\n"
        "- `reports/classification_report.csv`\n        "
        "- `reports/figures/matriz_confusion.png`\n"
        "- `reports/figures/curva_roc.png`\n"
        "- `reports/figures/distribucion_score.png`\n"
        "- `reports/model_report.md`\n"
    )

    return reporte


def guardar_model_report(reporte: str) -> None:
    """
    Guarda el reporte técnico del modelo.

    Args:
        reporte: Texto del reporte técnico.
    """
    with open(MODEL_REPORT_PATH, "w", encoding="utf-8") as file:
        file.write(reporte)

    logger.info("Reporte técnico guardado en: %s", MODEL_REPORT_PATH)


def evaluar_modelo() -> dict[str, Any]:
    """
    Ejecuta la evaluación completa del modelo.

    Returns:
        Diccionario con métricas finales del modelo.
    """
    crear_directorios()

    modelo = cargar_modelo()
    split_data = cargar_train_test_split()

    X_train = split_data["X_train"]
    X_test = split_data["X_test"]
    y_test = split_data["y_test"]

    y_score = obtener_probabilidades(modelo, X_test)
    y_pred = predecir_con_umbral(y_score)

    metricas = calcular_metricas(
        y_true=y_test,
        y_pred=y_pred,
        y_score=y_score,
    )

    model_type = obtener_tipo_modelo(modelo)

    metricas["model_type"] = model_type
    metricas["n_train"] = int(X_train.shape[0])
    metricas["n_test"] = int(X_test.shape[0])
    metricas["n_features"] = int(X_test.shape[1])

    guardar_metricas(metricas)
    guardar_classification_report(metricas)

    graficar_matriz_confusion(y_test, y_pred)
    graficar_curva_roc(y_test, y_score)
    graficar_distribucion_score(y_test, y_score)

    reporte = generar_model_report(
        metricas=metricas,
        n_train=int(X_train.shape[0]),
        n_test=int(X_test.shape[0]),
        n_features=int(X_test.shape[1]),
        model_type=model_type,
    )

    guardar_model_report(reporte)

    logger.info("Evaluación finalizada correctamente.")

    return metricas


if __name__ == "__main__":
    try:
        resultados = evaluar_modelo()

        print("Evaluación finalizada correctamente.")
        print(json.dumps(convertir_a_json_serializable(resultados), indent=4))

    except Exception as error:
        logger.exception("Error durante la evaluación del modelo: %s", error)
        sys.exit(1)
