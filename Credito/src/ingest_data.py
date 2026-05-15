"""
ingest_data.py — Ingesta, validación y registro de metadata
del dataset de renovación de préstamo.
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# ============================================================
# Configuración de logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# Rutas del proyecto
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

DATASET_PATH = DATA_DIR / "Dataset Renovacion_prestamo.csv"
METADATA_PATH = ARTIFACTS_DIR / "ingestion_metadata.json"


# ============================================================
# Columnas requeridas
# Ajusta esta lista según tu dataset real
# ============================================================

COLUMNAS_REQUERIDAS = {
    'MES',
    'CLIENTE',
    'LINEA_RENOVADO',
    'PLAZO_RENOVADO',
    'FLAG_VENTA',
    'USO_LINEA_TOTAL_TC_T2',
    'USO_TRIM_LINEA_BBVA',
    'NR_ENTIDADES_TOTAL_T2',
    'DIFF_NRO_ENTIDA_TOTALES_T2_T12',
    'SDO_CONSUMO_T2',
    'RESENCIA_OFERTA_PLD_RENOVADO',
    'Ahorro_Sldo_Bco_T1',
    'PConsumo_Sldo_Bco_T1',
    'SDO_BCO_tot_sm_pasivo_Bco_6M',
    'EDAD',
    'SEXO',
    'EST_CIVIL',
    'ANTIGUEDAD_MES',
    'REGION',
    'FLAG_LIMA_PROVINCIA',
    'SUELDO_ESTIMADO',
    'CUBRIR_DEUDA_CONSUMO_SF_RENOVA_PLD'}


# ============================================================
# Funciones auxiliares
# ============================================================

def calcular_hash_archivo(path: Path) -> str:
    """
    Calcula el hash MD5 del archivo para controlar si el dataset cambió.

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
    Verifica que el archivo exista.

    Args:
        path: Ruta del archivo.

    Raises:
        FileNotFoundError: Si el archivo no existe.
    """
    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {path}")


def validar_columnas(df: pd.DataFrame) -> None:
    """
    Valida que el dataset tenga las columnas requeridas.

    Args:
        df: DataFrame cargado.

    Raises:
        ValueError: Si faltan columnas requeridas.
    """
    columnas_actuales = set(df.columns)
    columnas_faltantes = COLUMNAS_REQUERIDAS - columnas_actuales

    if columnas_faltantes:
        raise ValueError(
            f"Columnas faltantes en el dataset: {columnas_faltantes}"
        )


def validar_dataset(df: pd.DataFrame) -> None:
    """
    Valida condiciones mínimas del dataset.

    Args:
        df: DataFrame cargado.

    Raises:
        ValueError: Si el dataset está vacío o no tiene datos válidos.
    """
    if df.empty:
        raise ValueError("El dataset está vacío.")

    if df.shape[0] == 0:
        raise ValueError("El dataset no tiene filas.")

    if df.shape[1] == 0:
        raise ValueError("El dataset no tiene columnas.")

    validar_columnas(df)


def generar_metadata(df: pd.DataFrame, source_path: Path) -> dict:
    """
    Genera metadata de ingesta del dataset.

    Args:
        df: DataFrame cargado.
        source_path: Ruta del archivo original.

    Returns:
        Diccionario con metadata.
    """
    metadata = {
        "source": str(source_path),
        "filename": source_path.name,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "file_hash_md5": calcular_hash_archivo(source_path),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    if "Default" in df.columns:
        metadata["target_distribution"] = (
            df["Default"]
            .value_counts(dropna=False)
            .to_dict()
        )

    return metadata


def guardar_metadata(metadata: dict, output_path: Path) -> None:
    """
    Guarda la metadata en formato JSON.

    Args:
        metadata: Diccionario de metadata.
        output_path: Ruta donde se guardará el JSON.
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
    Carga, valida y registra metadata del dataset.

    Args:
        ruta: Ruta del archivo CSV.

    Returns:
        DataFrame validado.
    """
    path = Path(ruta)

    validar_existencia_archivo(path)

    logger.info("Cargando datos desde: %s", path)

    df = pd.read_csv(path, sep=";")

    # Limpieza básica de nombres de columnas
    df.columns = df.columns.str.strip()

    validar_dataset(df)

    logger.info("Dataset cargado correctamente: %d filas x %d columnas", *df.shape)

    if "Default" in df.columns:
        logger.info(
            "Distribución de Default: %s",
            df["Default"].value_counts(dropna=False).to_dict(),
        )

    metadata = generar_metadata(df, path)
    guardar_metadata(metadata, METADATA_PATH)

    return df


# ============================================================
# Ejecución directa desde terminal
# ============================================================

if __name__ == "__main__":
    df = cargar_datos()
    print("Ingesta finalizada correctamente.")
    print(f"Filas: {df.shape[0]}")
    print(f"Columnas: {df.shape[1]}")
    print(f"Metadata: {METADATA_PATH}")