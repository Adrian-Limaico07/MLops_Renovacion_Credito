"""
validate_data.py — Validación de calidad del dataset de renovación de préstamo.

Este script:
1. Carga el dataset usando ingest_data.py
2. Renombra columnas para análisis y reportes
3. Valida columnas mínimas necesarias
4. Analiza tamaño, tipos de datos, nulos y duplicados
5. Analiza la distribución del target FLAG_VENTA
6. Analiza la frecuencia de MES
7. Guarda reportes en reports/
8. Guarda gráficos en reports/figures/
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import missingno as msno  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

try:
    from src.config import (
        COLUMNAS_RENAME,
        DATASET_PATH,
        FIGURES_DIR,
        REPORTS_DIR,
        TARGET,
    )
    from src.ingest_data import cargar_datos
except ImportError:
    from config import (
        COLUMNAS_RENAME,
        DATASET_PATH,
        FIGURES_DIR,
        REPORTS_DIR,
        TARGET,
    )
    from ingest_data import cargar_datos


# ============================================================
# Configuración de logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# Rutas de salida
# ============================================================

VALIDATION_SUMMARY_PATH = REPORTS_DIR / "data_validation_summary.json"
NULL_REPORT_PATH = REPORTS_DIR / "null_report.csv"
TARGET_DISTRIBUTION_PATH = REPORTS_DIR / "target_distribution.csv"
MES_DISTRIBUTION_PATH = REPORTS_DIR / "mes_distribution.csv"
DESCRIBE_REPORT_PATH = REPORTS_DIR / "describe_report.csv"

TARGET_PLOT_PATH = FIGURES_DIR / "target_flag_venta.png"
MISSING_MATRIX_PATH = FIGURES_DIR / "missing_matrix.png"


# ============================================================
# Columnas mínimas esperadas para validación
# ============================================================

MES_COLUMN = "MES"

COLUMNAS_OBLIGATORIAS_VALIDACION = {
    TARGET,
    MES_COLUMN,
}


# ============================================================
# Funciones auxiliares
# ============================================================

def crear_directorios() -> None:
    """
    Crea las carpetas necesarias para guardar reportes y figuras.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Carpetas de reportes verificadas correctamente.")


def convertir_a_json_serializable(obj: Any) -> Any:
    """
    Convierte objetos de pandas/numpy a tipos compatibles con JSON.

    Args:
        obj: Objeto a convertir.

    Returns:
        Objeto compatible con json.dump.
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


def cargar_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    """
    Carga el dataset usando la función oficial de ingesta.

    Esto evita duplicar la lógica de lectura del CSV y asegura que
    validate_data.py use el mismo separador, encoding y validaciones
    definidos en ingest_data.py.

    Args:
        path: Ruta del archivo CSV.

    Returns:
        DataFrame cargado y validado desde ingest_data.py.
    """
    logger.info("Cargando dataset desde ingest_data.py")

    df = cargar_datos(path)

    logger.info(
        "Dataset recibido desde ingest_data.py: %d filas x %d columnas",
        *df.shape,
    )

    return df


def renombrar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renombra columnas del dataset para facilitar el entendimiento.

    Args:
        df: DataFrame original.

    Returns:
        DataFrame con columnas renombradas.
    """
    df = df.copy()

    columnas_existentes = set(df.columns)

    columnas_a_renombrar = {
        col_original: col_nueva
        for col_original, col_nueva in COLUMNAS_RENAME.items()
        if col_original in columnas_existentes
    }

    columnas_no_encontradas = set(COLUMNAS_RENAME.keys()) - columnas_existentes

    if columnas_no_encontradas:
        logger.warning(
            "Estas columnas no se encontraron y no serán renombradas: %s",
            sorted(columnas_no_encontradas),
        )

    df = df.rename(columns=columnas_a_renombrar)

    logger.info(
        "Columnas renombradas correctamente: %d",
        len(columnas_a_renombrar),
    )

    return df


def validar_columnas_obligatorias(df: pd.DataFrame) -> None:
    """
    Valida que existan las columnas mínimas requeridas para el análisis.

    Args:
        df: DataFrame cargado.

    Raises:
        ValueError: Si faltan columnas obligatorias.
    """
    columnas_faltantes = COLUMNAS_OBLIGATORIAS_VALIDACION - set(df.columns)

    if columnas_faltantes:
        raise ValueError(
            "Faltan columnas obligatorias en el dataset: "
            f"{sorted(columnas_faltantes)}"
        )

    logger.info("Columnas obligatorias validadas correctamente.")


def validar_target(df: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    """
    Calcula la distribución absoluta y porcentual del target.

    Args:
        df: DataFrame.
        target: Nombre de la variable objetivo.

    Returns:
        Tabla con conteo absoluto y porcentaje.
    """
    if target not in df.columns:
        raise ValueError(f"No existe la columna target: {target}")

    nulos_target = df[target].isna().sum()

    if nulos_target > 0:
        logger.warning(
            "La columna %s tiene %d valores nulos.",
            target,
            nulos_target,
        )

    value_counts_abs = df[target].value_counts(dropna=False)
    value_counts_norm = (
        df[target]
        .value_counts(normalize=True, dropna=False)
        .mul(100)
        .round(2)
    )

    distribution_table = pd.DataFrame(
        {
            "Conteo Absoluto": value_counts_abs,
            "Porcentaje (%)": value_counts_norm,
        }
    )

    logger.info("Distribución de %s:\n%s", target, distribution_table)

    return distribution_table


def generar_reporte_nulos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera reporte de valores nulos por columna.

    Args:
        df: DataFrame.

    Returns:
        DataFrame con cantidad y porcentaje de nulos.
    """
    null_counts = df.isna().sum()
    null_percentages = (df.isna().sum() / len(df) * 100).round(2)

    null_info = pd.DataFrame(
        {
            "Nulo": null_counts,
            "Porcentaje Nulo": null_percentages,
        }
    )

    null_info = null_info.sort_values(by="Nulo", ascending=False)

    logger.info("Top 10 columnas con más nulos:\n%s", null_info.head(10))

    return null_info


def generar_reporte_mes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera tabla de frecuencia para la columna MES.

    Args:
        df: DataFrame.

    Returns:
        DataFrame con cantidad y porcentaje por mes.
    """
    if MES_COLUMN not in df.columns:
        logger.warning(
            "No existe la columna %s. No se genera reporte de meses.",
            MES_COLUMN,
        )
        return pd.DataFrame()

    frecuencia_meses = pd.DataFrame(
        {
            "Cantidad": df[MES_COLUMN].value_counts(dropna=False),
            "Porcentaje": (
                df[MES_COLUMN]
                .value_counts(normalize=True, dropna=False)
                .mul(100)
                .round(2)
            ),
        }
    )

    logger.info("Frecuencia de %s:\n%s", MES_COLUMN, frecuencia_meses)

    return frecuencia_meses


def guardar_grafico_target(df: pd.DataFrame, target: str = TARGET) -> None:
    """
    Guarda gráfico de barras del target.

    Args:
        df: DataFrame.
        target: Nombre de la variable objetivo.
    """
    if target not in df.columns:
        logger.warning(
            "No se genera gráfico. No existe la columna %s.",
            target,
        )
        return

    plt.figure(figsize=(7, 5))
    sns.countplot(x=target, data=df)
    plt.title(f"Distribución de {target}")
    plt.xlabel(target)
    plt.ylabel("Cantidad")
    plt.tight_layout()
    plt.savefig(TARGET_PLOT_PATH)
    plt.close()

    logger.info("Gráfico del target guardado en: %s", TARGET_PLOT_PATH)


def guardar_matriz_nulos(df: pd.DataFrame) -> None:
    """
    Guarda la matriz visual de valores nulos.

    Args:
        df: DataFrame.
    """
    msno.matrix(df, figsize=(12, 6))
    plt.tight_layout()
    plt.savefig(MISSING_MATRIX_PATH)
    plt.close("all")

    logger.info("Matriz de nulos guardada en: %s", MISSING_MATRIX_PATH)


def guardar_reportes(
    df: pd.DataFrame,
    target_distribution: pd.DataFrame,
    null_report: pd.DataFrame,
    mes_distribution: pd.DataFrame,
) -> None:
    """
    Guarda los reportes principales en la carpeta reports/.

    Args:
        df: DataFrame validado.
        target_distribution: Distribución del target.
        null_report: Reporte de nulos.
        mes_distribution: Frecuencia de meses.
    """
    target_distribution.to_csv(TARGET_DISTRIBUTION_PATH, index=True)
    null_report.to_csv(NULL_REPORT_PATH, index=True)

    if not mes_distribution.empty:
        mes_distribution.to_csv(MES_DISTRIBUTION_PATH, index=True)

    describe_report = df.describe(include="all").T
    describe_report.to_csv(DESCRIBE_REPORT_PATH, index=True)

    logger.info("Reporte de target guardado en: %s", TARGET_DISTRIBUTION_PATH)
    logger.info("Reporte de nulos guardado en: %s", NULL_REPORT_PATH)
    logger.info("Reporte estadístico guardado en: %s", DESCRIBE_REPORT_PATH)


def obtener_distribucion_columna(df: pd.DataFrame, columna: str) -> dict:
    """
    Obtiene la distribución de una columna en formato JSON serializable.

    Args:
        df: DataFrame.
        columna: Nombre de la columna.

    Returns:
        Diccionario con distribución.
    """
    if columna not in df.columns:
        return {}

    distribucion = df[columna].value_counts(dropna=False).to_dict()

    return convertir_a_json_serializable(distribucion)


def generar_resumen_validacion(
    df: pd.DataFrame,
    null_report: pd.DataFrame,
    duplicated_count: int,
    dataset_path: Path = DATASET_PATH,
) -> dict:
    """
    Genera resumen general de validación del dataset.

    Args:
        df: DataFrame.
        null_report: Reporte de nulos.
        duplicated_count: Cantidad de duplicados.
        dataset_path: Ruta del dataset validado.

    Returns:
        Diccionario de resumen.
    """
    columnas_con_nulos = null_report[null_report["Nulo"] > 0]

    resumen = {
        "dataset_path": str(dataset_path),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "dtypes": {
            column: str(dtype)
            for column, dtype in df.dtypes.items()
        },
        "duplicated_rows": int(duplicated_count),
        "columns_with_nulls": int(columnas_con_nulos.shape[0]),
        "total_null_values": int(df.isna().sum().sum()),
        "target": TARGET,
        "target_distribution": obtener_distribucion_columna(df, TARGET),
        "mes_distribution": obtener_distribucion_columna(df, MES_COLUMN),
    }

    return convertir_a_json_serializable(resumen)


def guardar_resumen_json(resumen: dict) -> None:
    """
    Guarda el resumen de validación en JSON.

    Args:
        resumen: Diccionario de resumen.
    """
    with open(VALIDATION_SUMMARY_PATH, "w", encoding="utf-8") as file:
        json.dump(resumen, file, indent=4, ensure_ascii=False)

    logger.info(
        "Resumen de validación guardado en: %s",
        VALIDATION_SUMMARY_PATH,
    )


# ============================================================
# Función principal
# ============================================================

def validar_datos(path: Path = DATASET_PATH) -> pd.DataFrame:
    """
    Ejecuta todo el proceso de validación de datos.

    Args:
        path: Ruta del dataset.

    Returns:
        DataFrame validado.
    """
    crear_directorios()

    df = cargar_dataset(path)
    df = renombrar_columnas(df)

    logger.info("Columnas finales del dataset: %s", df.columns.tolist())

    validar_columnas_obligatorias(df)

    target_distribution = validar_target(df, target=TARGET)
    null_report = generar_reporte_nulos(df)
    mes_distribution = generar_reporte_mes(df)

    duplicated_count = int(df.duplicated().sum())

    logger.info("Cantidad de registros duplicados: %d", duplicated_count)

    guardar_grafico_target(df, target=TARGET)
    guardar_matriz_nulos(df)

    guardar_reportes(
        df=df,
        target_distribution=target_distribution,
        null_report=null_report,
        mes_distribution=mes_distribution,
    )

    resumen = generar_resumen_validacion(
        df=df,
        null_report=null_report,
        duplicated_count=duplicated_count,
        dataset_path=path,
    )

    guardar_resumen_json(resumen)

    logger.info("Validación de datos finalizada correctamente.")

    return df


# ============================================================
# Ejecución desde terminal
# ============================================================

if __name__ == "__main__":
    try:
        validar_datos()
        print("Validación de datos finalizada correctamente.")
        print(f"Resumen: {VALIDATION_SUMMARY_PATH}")
        print(f"Reporte de nulos: {NULL_REPORT_PATH}")
        print(f"Gráficos: {FIGURES_DIR}")
    except Exception as error:
        logger.exception("Error durante la validación de datos: %s", error)
        sys.exit(1)
