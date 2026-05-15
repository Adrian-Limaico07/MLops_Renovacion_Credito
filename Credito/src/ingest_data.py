"""
ingest_data.py — Ingesta y validación inicial del dataset de renovación de préstamo.
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

DATASET_PATH = DATA_DIR / "Dataset Renovacion_prestamo.csv"
METADATA_PATH = ARTIFACTS_DIR / "ingestion_metadata.json"


COLUMNAS_REQUERIDAS = {
    "MES",
    "CLIENTE",
    "LINEA_RENOVADO",
    "PLAZO_RENOVADO",
    "FLAG_VENTA",
    "USO_LINEA_TOTAL_TC_T2",
    "USO_TRIM_LINEA_BBVA",
    "NR_ENTIDADES_TOTAL_T2",
    "DIFF_NRO_ENTIDA_TOTALES_T2_T12",
    "SDO_CONSUMO_T2",
    "RESENCIA_OFERTA_PLD_RENOVADO",
    "Ahorro_Sldo_Bco_T1",
    "PConsumo_Sldo_Bco_T1",
    "SDO_BCO_tot_sm_pasivo_Bco_6M",
    "EDAD",
    "SEXO",
    "EST_CIVIL",
    "ANTIGUEDAD_MES",
    "REGION",
    "FLAG_LIMA_PROVINCIA",
    "SUELDO_ESTIMADO",
    "CUBRIR_DEUDA_CONSUMO_SF_RENOVA_PLD",
}


def calcular_hash_archivo(path: Path) -> str:
    """Calcula el hash MD5 del archivo."""
    hash_md5 = hashlib.md5()

    with open(path, "rb") as archivo:
        for bloque in iter(lambda: archivo.read(4096), b""):
            hash_md5.update(bloque)

    return hash_md5.hexdigest()


def validar_existencia_archivo(path: Path) -> None:
    """Valida que el archivo exista."""
    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {path}")


def validar_columnas(df: pd.DataFrame) -> None:
    """Valida que existan las columnas requeridas."""
    columnas_faltantes = COLUMNAS_REQUERIDAS - set(df.columns)

    if columnas_faltantes:
        raise ValueError(
            f"Columnas faltantes en el dataset: {columnas_faltantes}"
        )

    logger.info("Columnas requeridas validadas correctamente.")


def generar_metadata(df: pd.DataFrame, source_path: Path) -> dict:
    """Genera metadata de ingesta."""
    metadata = {
        "source": str(source_path),
        "filename": source_path.name,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "file_hash_md5": calcular_hash_archivo(source_path),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    if "FLAG_VENTA" in df.columns:
        metadata["target_distribution"] = (
            df["FLAG_VENTA"]
            .value_counts(dropna=False)
            .to_dict()
        )

    return metadata


def guardar_metadata(metadata: dict, output_path: Path) -> None:
    """Guarda metadata en JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as archivo:
        json.dump(metadata, archivo, indent=4, ensure_ascii=False)

    logger.info("Metadata guardada en: %s", output_path)


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

    df = pd.read_csv(path, sep=";", decimal=".")

    df.columns = df.columns.str.strip()

    if df.empty:
        raise ValueError("El dataset está vacío.")

    validar_columnas(df)

    logger.info("Dataset cargado correctamente: %d filas x %d columnas", *df.shape)

    if "FLAG_VENTA" in df.columns:
        logger.info(
            "Distribución de FLAG_VENTA: %s",
            df["FLAG_VENTA"].value_counts(dropna=False).to_dict(),
        )

    metadata = generar_metadata(df, path)
    guardar_metadata(metadata, METADATA_PATH)

    return df


if __name__ == "__main__":
    df = cargar_datos()

    print("Ingesta finalizada correctamente.")
    print(f"Filas: {df.shape[0]}")
    print(f"Columnas: {df.shape[1]}")
    print(f"Metadata: {METADATA_PATH}")