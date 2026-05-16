"""
va_model.py — Quality gate de métricas del modelo.

Este script:
1. Lee artifacts/metrics.json
2. Valida que el modelo cumpla los umbrales mínimos
3. Si falla, detiene el pipeline con sys.exit(1)
4. Si aprueba, permite continuar el flujo CI/CD
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

try:
    from src.config import (
        ARTIFACTS_DIR,
        F1_MIN,
        RECALL_MIN,
        ROC_AUC_MIN,
    )
except ImportError:
    from config import (
        ARTIFACTS_DIR,
        F1_MIN,
        RECALL_MIN,
        ROC_AUC_MIN,
    )


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

METRICS_PATH = ARTIFACTS_DIR / "metrics.json"


# ============================================================
# Funciones auxiliares
# ============================================================

def cargar_metricas(path: Path = METRICS_PATH) -> dict[str, Any]:
    """
    Carga las métricas del modelo desde metrics.json.

    Args:
        path: Ruta del archivo de métricas.

    Returns:
        Diccionario con métricas.

    Raises:
        FileNotFoundError: Si no existe metrics.json.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de métricas: {path}. "
            "Ejecuta primero python src/train_pipeline.py y python src/evaluate_model.py"
        )

    with open(path, "r", encoding="utf-8") as file:
        metricas = json.load(file)

    logger.info("Métricas cargadas desde: %s", path)

    return metricas


def obtener_metrica(metricas: dict[str, Any], nombre: str) -> float:
    """
    Obtiene una métrica numérica desde el diccionario de métricas.

    Args:
        metricas: Diccionario con métricas.
        nombre: Nombre de la métrica.

    Returns:
        Valor numérico de la métrica.

    Raises:
        ValueError: Si la métrica no existe o no es numérica.
    """
    if nombre not in metricas:
        raise ValueError(f"No existe la métrica requerida: {nombre}")

    try:
        return float(metricas[nombre])
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"La métrica {nombre} no tiene un valor numérico válido: {metricas[nombre]}"
        ) from error


def validar_umbral(
    nombre: str,
    valor: float,
    umbral: float,
) -> bool:
    """
    Valida si una métrica cumple su umbral mínimo.

    Args:
        nombre: Nombre de la métrica.
        valor: Valor obtenido.
        umbral: Valor mínimo requerido.

    Returns:
        True si cumple, False si falla.
    """
    cumple = valor >= umbral

    estado = "APROBADO" if cumple else "FALLÓ"

    logger.info(
        "%s | %s = %.4f | umbral mínimo = %.4f",
        estado,
        nombre,
        valor,
        umbral,
    )

    return cumple


def validar_modelo(metricas: dict[str, Any]) -> bool:
    """
    Valida las métricas principales del modelo.

    Args:
        metricas: Diccionario con métricas del modelo.

    Returns:
        True si el modelo cumple todos los umbrales.
    """
    recall = obtener_metrica(metricas, "recall")
    roc_auc = obtener_metrica(metricas, "roc_auc")
    f1 = obtener_metrica(metricas, "f1")

    resultados = {
        "recall": validar_umbral("recall", recall, RECALL_MIN),
        "roc_auc": validar_umbral("roc_auc", roc_auc, ROC_AUC_MIN),
        "f1": validar_umbral("f1", f1, F1_MIN),
    }

    aprobado = all(resultados.values())

    return aprobado


def imprimir_resumen(metricas: dict[str, Any]) -> None:
    """
    Imprime resumen de métricas en consola.

    Args:
        metricas: Diccionario con métricas.
    """
    print("=" * 70)
    print("QUALITY GATE — VALIDACIÓN DEL MODELO")
    print("=" * 70)

    if "best_model" in metricas:
        print(f"Modelo seleccionado : {metricas['best_model']}")

    if "model_type" in metricas:
        print(f"Tipo de modelo      : {metricas['model_type']}")

    print(f"Accuracy            : {float(metricas.get('accuracy', 0)):.4f}")
    print(f"Precision           : {float(metricas.get('precision', 0)):.4f}")
    print(f"Recall              : {float(metricas.get('recall', 0)):.4f}")
    print(f"F1-score            : {float(metricas.get('f1', 0)):.4f}")
    print(f"ROC AUC             : {float(metricas.get('roc_auc', 0)):.4f}")

    print("-" * 70)
    print(f"Recall mínimo       : {RECALL_MIN:.4f}")
    print(f"ROC AUC mínimo      : {ROC_AUC_MIN:.4f}")
    print(f"F1 mínimo           : {F1_MIN:.4f}")
    print("=" * 70)


# ============================================================
# Ejecución principal
# ============================================================

def main() -> None:
    """
    Ejecuta el quality gate del modelo.
    """
    try:
        metricas = cargar_metricas()
        imprimir_resumen(metricas)

        aprobado = validar_modelo(metricas)

        if not aprobado:
            logger.error("QUALITY GATE FALLÓ. El modelo no cumple los umbrales mínimos.")
            sys.exit(1)

        logger.info("QUALITY GATE APROBADO. El modelo cumple los umbrales mínimos.")
        print("Modelo aprobado correctamente.")

    except Exception as error:
        logger.exception("Error durante la validación del modelo: %s", error)
        sys.exit(1)


if __name__ == "__main__":
    main()
