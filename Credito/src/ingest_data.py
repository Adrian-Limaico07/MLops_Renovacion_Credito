"""
ingest_data.py — Ingesta y validación inicial del dataset de renovación de préstamo.
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:
    from src.config import (
        ARTIFACTS_DIR,
        COLUMNAS_REQUERIDAS,
        CSV_DECIMAL,
        CSV_ENCODING,
        CSV_SEPARATOR,
        DATASET_PATH,
        INGESTION_METADATA_PATH,
        TARGET,
    )
except ImportError:
    from config import (
        ARTIFACTS_DIR,
        COLUMNAS_REQUERIDAS,
        CSV_DECIMAL,
        CSV_ENCODING,
        CSV_SEPARATOR,
        DATASET_PATH,
        INGESTION_METADATA_PATH,
        TARGET,
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
# Funciones auxiliares
# ============================================================

def calcular_hash_archivo(path: Path) -> str:
    """
    Calcula el hash MD5 del archivo.

    Args:
        path: Ruta del archivo.

    Returns:
        Hash MD5 del archivo.
    """
    hash_md5 = hashlib.md5()

    with open(path, "rb") as archivo:
        for bloque in iter(lambda: archivo.read(4096), b""):
            hash_md5.update(bloque)

    return hash_md5.hexdigest()


def validar_existencia_archivo(path: Path) -> None:
    """
    Valida que el archivo exista.

    Args:
        path: Ruta del archivo.

    Raises:
        FileNotFoundError: Si el archivo no existe.
    """
    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {path}")


def limpiar_nombres_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia espacios y caracteres especiales de los nombres de columnas.

    Args:
        df: DataFrame original.

    Returns:
        DataFrame con nombres de columnas normalizados.
    """
    df = df.copy()

    df.columns = (
        df.columns
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    return df


def validar_columnas(df: pd.DataFrame) -> None:
    """
    Valida que existan las columnas requeridas.

    Args:
        df: DataFrame cargado.

    Raises:
        ValueError: Si faltan columnas requeridas.
    """
    columnas_actuales = set(df.columns)
    columnas_faltantes = COLUMNAS_REQUERIDAS - columnas_actuales

    if columnas_faltantes:
        raise ValueError(
            f"Columnas faltantes en el dataset: {sorted(columnas_faltantes)}"
        )

    logger.info("Columnas requeridas validadas correctamente.")


def validar_dataset_basico(df: pd.DataFrame) -> None:
    """
    Ejecuta validaciones básicas del dataset.

    Args:
        df: DataFrame cargado.

    Raises:
        ValueError: Si el dataset está vacío o mal cargado.
    """
    if df.empty:
        raise ValueError("El dataset está vacío.")

    if df.shape[1] == 1:
        columna_unica = df.columns[0]
        raise ValueError(
            "El dataset se cargó con una sola columna. "
            "Probablemente el separador del CSV está mal configurado. "
            f"Columna detectada: {columna_unica}"
        )

    validar_columnas(df)

    if TARGET not in df.columns:
        raise ValueError(f"No existe la variable objetivo: {TARGET}")


def obtener_distribucion_target(df: pd.DataFrame) -> dict:
    """
    Obtiene la distribución de la variable objetivo.

    Args:
        df: DataFrame cargado.

    Returns:
        Diccionario con conteos del target.
    """
    if TARGET not in df.columns:
        return {}

    distribucion = df[TARGET].value_counts(dropna=False).to_dict()

    return {
        str(clase): int(conteo)
        for clase, conteo in distribucion.items()
    }


def generar_metadata(df: pd.DataFrame, source_path: Path) -> dict:
    """
    Genera metadata de ingesta.

    Args:
        df: DataFrame cargado.
        source_path: Ruta del archivo original.

    Returns:
        Diccionario con metadata de ingesta.
    """
    metadata = {
        "source": str(source_path),
        "filename": source_path.name,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "target": TARGET,
        "target_distribution": obtener_distribucion_target(df),
        "file_hash_md5": calcular_hash_archivo(source_path),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    return metadata


def guardar_metadata(metadata: dict, output_path: Path = INGESTION_METADATA_PATH) -> None:
    """
    Guarda metadata en formato JSON.

    Args:
        metadata: Diccionario de metadata.
        output_path: Ruta de salida del archivo JSON.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as archivo:
        json.dump(metadata, archivo, indent=4, ensure_ascii=False)

    logger.info("Metadata guardada en: %s", output_path)


# ============================================================
# Función principal de ingesta
# ============================================================

def cargar_datos(ruta: str | Path = DATASET_PATH) -> pd.DataFrame:
    """
    Carga y valida el dataset de renovación de préstamo.

    Args:
        ruta: Ruta del archivo CSV.

    Returns:
        DataFrame cargado y validado.
    """
    path = Path(ruta)

    validar_existencia_archivo(path)

    logger.info("Cargando dataset desde: %s", path)

    df = pd.read_csv(
        path,
        sep=CSV_SEPARATOR,
        decimal=CSV_DECIMAL,
        encoding=CSV_ENCODING,
    )

    df = limpiar_nombres_columnas(df)

    validar_dataset_basico(df)

    logger.info(
        "Dataset cargado correctamente: %d filas x %d columnas",
        *df.shape,
    )

    logger.info(
        "Distribución de %s: %s",
        TARGET,
        obtener_distribucion_target(df),
    )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    metadata = generar_metadata(df, path)
    guardar_metadata(metadata)

    return df


# ============================================================
# Ejecución directa desde terminal
# ============================================================

if __name__ == "__main__":
    dataframe = cargar_datos()

    print("Ingesta finalizada correctamente.")
    print(f"Filas: {dataframe.shape[0]}")
    print(f"Columnas: {dataframe.shape[1]}")
    print(f"Metadata: {INGESTION_METADATA_PATH}")