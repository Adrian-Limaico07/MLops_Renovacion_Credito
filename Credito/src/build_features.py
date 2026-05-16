"""
build_features.py — Limpieza, imputación, transformación logarítmica
y codificación de variables para el modelo de renovación de crédito.

Este script:
1. Carga datos desde ingest_data.py
2. Renombra columnas para mejor interpretación
3. Aplica capping a variables monetarias con valores negativos
4. Crea variables logarítmicas
5. Imputa valores nulos
6. Codifica variables categóricas con One-Hot Encoding
7. Elimina columnas que no deben entrar al modelo
8. Guarda el dataset final en artifacts/
9. Guarda metadata del preprocesamiento en artifacts/preprocessor.pkl
"""

import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from src.config import (
        ARTIFACTS_DIR,
        COLUMNAS_RENAME,
        DATASET_PATH,
        RANDOM_STATE,
        TARGET,
    )
    from src.ingest_data import cargar_datos
except ImportError:
    from config import (
        ARTIFACTS_DIR,
        COLUMNAS_RENAME,
        DATASET_PATH,
        RANDOM_STATE,
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

FEATURES_DATA_PATH = ARTIFACTS_DIR / "df_features.csv"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.pkl"
FEATURES_METADATA_PATH = ARTIFACTS_DIR / "features_metadata.json"


# ============================================================
# Configuración de variables
# ============================================================

COLUMNAS_CAPPING_CERO = [
    "Ahorro",
    "Prestamo_vigente",
    "Promed_6Mdeuda",
]

VARIABLES_LOG = [
    "Uso_Linea",
    "Uso_TrimLinea",
    "Saldo_Consumo",
    "SUELDO_ESTIMADO",
    "ANTIGUEDAD_MES",
    "Linea_Renovado",
    "Ahorro",
    "Prestamo_vigente",
    "Promed_6Mdeuda",
    "Deuda_Cubierta%",
]

VARIABLES_IMPUTACION_ALEATORIA = [
    "Uso_TrimLinea_LOG",
    "Uso_Linea_LOG",
    "Meses_oferta",
]

VARIABLES_IMPUTACION_MEDIANA = [
    "Saldo_Consumo_LOG",
    "SUELDO_ESTIMADO_LOG",
    "ANTIGUEDAD_MES_LOG",
    "EDAD",
]

VARIABLES_CATEGORICAS = [
    "REGION",
    "SEXO",
    "EST_CIVIL",
]

COLUMNAS_EXCLUIR_MODELO = [
    # Identificador del cliente: no debe entrar al modelo
    "CLIENTE",

    # Variables originales reemplazadas por versiones logarítmicas
    "Uso_Linea",
    "Uso_TrimLinea",
    "Saldo_Consumo",
    "SUELDO_ESTIMADO",
    "ANTIGUEDAD_MES",
    "Linea_Renovado",
    "Ahorro",
    "Prestamo_vigente",
    "Promed_6Mdeuda",
    "Deuda_Cubierta%",
]


# ============================================================
# Funciones auxiliares
# ============================================================

def crear_directorios() -> None:
    """
    Crea la carpeta artifacts/ si no existe.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Carpeta artifacts verificada: %s", ARTIFACTS_DIR)


def renombrar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renombra columnas del dataset según COLUMNAS_RENAME.

    Args:
        df: DataFrame original.

    Returns:
        DataFrame con columnas renombradas.
    """
    df = df.copy()

    columnas_existentes = set(df.columns)

    columnas_a_renombrar = {
        columna_original: columna_nueva
        for columna_original, columna_nueva in COLUMNAS_RENAME.items()
        if columna_original in columnas_existentes
    }

    columnas_no_encontradas = set(COLUMNAS_RENAME.keys()) - columnas_existentes

    if columnas_no_encontradas:
        logger.warning(
            "Columnas no encontradas para renombrar: %s",
            sorted(columnas_no_encontradas),
        )

    df = df.rename(columns=columnas_a_renombrar)

    logger.info("Columnas renombradas: %d", len(columnas_a_renombrar))

    return df


def convertir_a_numerico(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """
    Convierte columnas seleccionadas a formato numérico.

    Args:
        df: DataFrame.
        columnas: Lista de columnas a convertir.

    Returns:
        DataFrame con columnas convertidas.
    """
    df = df.copy()

    for columna in columnas:
        if columna in df.columns:
            df[columna] = pd.to_numeric(df[columna], errors="coerce")

    return df


def aplicar_capping_cero(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica capping inferior en cero a variables que no deberían ser negativas.

    Args:
        df: DataFrame.

    Returns:
        DataFrame con capping aplicado.
    """
    df = df.copy()

    for columna in COLUMNAS_CAPPING_CERO:
        if columna in df.columns:
            negativos = int((df[columna] < 0).sum())

            if negativos > 0:
                logger.info(
                    "Aplicando capping a cero en %s. Valores negativos: %d",
                    columna,
                    negativos,
                )

            df[columna] = np.maximum(0, df[columna])

    return df


def aplicar_logaritmos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea variables logarítmicas usando np.log1p.

    Antes de aplicar log1p, si existen valores menores que cero,
    se reemplazan por cero para evitar errores matemáticos.

    Args:
        df: DataFrame.

    Returns:
        DataFrame con nuevas columnas *_LOG.
    """
    df = df.copy()

    for columna in VARIABLES_LOG:
        if columna not in df.columns:
            logger.warning(
                "No se crea LOG. La columna no existe: %s",
                columna,
            )
            continue

        df[columna] = pd.to_numeric(df[columna], errors="coerce")

        negativos = int((df[columna] < 0).sum())

        if negativos > 0:
            logger.warning(
                "La columna %s tiene %d valores negativos. Se reemplazan por 0.",
                columna,
                negativos,
            )
            df[columna] = df[columna].clip(lower=0)

        nueva_columna = f"{columna}_LOG"
        df[nueva_columna] = np.log1p(df[columna])

        logger.info("Variable logarítmica creada: %s", nueva_columna)

    return df


def imputar_aleatorio_uniforme(
    df: pd.DataFrame,
    columna: str,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Imputa nulos de una columna usando distribución uniforme
    entre media - desviación estándar y media + desviación estándar.

    Args:
        df: DataFrame.
        columna: Columna a imputar.
        rng: Generador aleatorio de NumPy.

    Returns:
        DataFrame imputado y metadata de imputación.
    """
    df = df.copy()

    metadata = {
        "columna": columna,
        "metodo": "random_uniform_mean_std",
        "nulos_imputados": 0,
        "media": None,
        "std": None,
        "lower_bound": None,
        "upper_bound": None,
    }

    if columna not in df.columns:
        logger.warning("No se imputa. La columna no existe: %s", columna)
        return df, metadata

    df[columna] = pd.to_numeric(df[columna], errors="coerce")

    null_indices = df[columna].isna()
    num_nulls = int(null_indices.sum())

    if num_nulls == 0:
        logger.info("La columna %s no tiene nulos para imputar.", columna)
        return df, metadata

    media = float(df[columna].mean())
    std = float(df[columna].std())

    if np.isnan(media):
        media = 0.0

    if np.isnan(std):
        std = 0.0

    lower_bound = max(0.0, media - std)
    upper_bound = max(lower_bound, media + std)

    valores_imputados = rng.uniform(
        lower_bound,
        upper_bound,
        num_nulls,
    )

    df.loc[null_indices, columna] = valores_imputados

    metadata.update(
        {
            "nulos_imputados": num_nulls,
            "media": media,
            "std": std,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
        }
    )

    logger.info(
        "Imputación aleatoria en %s: %d nulos imputados.",
        columna,
        num_nulls,
    )

    return df, metadata


def imputar_mediana(df: pd.DataFrame, columna: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Imputa valores nulos con la mediana de la columna.

    Args:
        df: DataFrame.
        columna: Columna a imputar.

    Returns:
        DataFrame imputado y metadata de imputación.
    """
    df = df.copy()

    metadata = {
        "columna": columna,
        "metodo": "median",
        "nulos_imputados": 0,
        "valor_imputacion": None,
    }

    if columna not in df.columns:
        logger.warning("No se imputa. La columna no existe: %s", columna)
        return df, metadata

    df[columna] = pd.to_numeric(df[columna], errors="coerce")

    num_nulls = int(df[columna].isna().sum())

    if num_nulls == 0:
        logger.info("La columna %s no tiene nulos para imputar.", columna)
        return df, metadata

    mediana = df[columna].median()

    if pd.isna(mediana):
        mediana = 0.0

    df[columna] = df[columna].fillna(mediana)

    metadata.update(
        {
            "nulos_imputados": num_nulls,
            "valor_imputacion": float(mediana),
        }
    )

    logger.info(
        "Imputación por mediana en %s: %d nulos imputados.",
        columna,
        num_nulls,
    )

    return df, metadata


def imputar_moda_categoricas(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Imputa variables categóricas con la moda.

    Args:
        df: DataFrame.

    Returns:
        DataFrame imputado y metadata de imputación.
    """
    df = df.copy()
    metadata = []

    for columna in VARIABLES_CATEGORICAS:
        info = {
            "columna": columna,
            "metodo": "mode",
            "nulos_imputados": 0,
            "valor_imputacion": None,
        }

        if columna not in df.columns:
            logger.warning(
                "No se imputa categórica. La columna no existe: %s",
                columna,
            )
            metadata.append(info)
            continue

        num_nulls = int(df[columna].isna().sum())

        if num_nulls == 0:
            logger.info("La columna %s no tiene nulos para imputar.", columna)
            metadata.append(info)
            continue

        moda = df[columna].mode(dropna=True)

        if moda.empty:
            valor_imputacion = "DESCONOCIDO"
        else:
            valor_imputacion = moda.iloc[0]

        df[columna] = df[columna].fillna(valor_imputacion)

        info.update(
            {
                "nulos_imputados": num_nulls,
                "valor_imputacion": str(valor_imputacion),
            }
        )

        logger.info(
            "Imputación por moda en %s: %d nulos imputados.",
            columna,
            num_nulls,
        )

        metadata.append(info)

    return df, metadata


def imputar_nulos(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Ejecuta todas las imputaciones definidas para el proyecto.

    Args:
        df: DataFrame.

    Returns:
        DataFrame imputado y metadata de imputaciones.
    """
    df = df.copy()
    rng = np.random.default_rng(RANDOM_STATE)

    metadata = {
        "random_uniform": [],
        "median": [],
        "mode": [],
    }

    for columna in VARIABLES_IMPUTACION_ALEATORIA:
        df, info = imputar_aleatorio_uniforme(df, columna, rng)
        metadata["random_uniform"].append(info)

    for columna in VARIABLES_IMPUTACION_MEDIANA:
        df, info = imputar_mediana(df, columna)
        metadata["median"].append(info)

    df, info_categoricas = imputar_moda_categoricas(df)
    metadata["mode"] = info_categoricas

    return df, metadata


def codificar_categoricas(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """
    Aplica One-Hot Encoding a variables categóricas usando pd.get_dummies.

    Args:
        df: DataFrame.

    Returns:
        DataFrame codificado y niveles categóricos utilizados.
    """
    df = df.copy()

    columnas_presentes = [
        columna
        for columna in VARIABLES_CATEGORICAS
        if columna in df.columns
    ]

    niveles_categoricos = {
        columna: sorted(df[columna].dropna().astype(str).unique().tolist())
        for columna in columnas_presentes
    }

    if not columnas_presentes:
        logger.warning("No hay variables categóricas presentes para codificar.")
        return df, niveles_categoricos

    df_encoded = pd.get_dummies(
        df,
        columns=columnas_presentes,
        drop_first=False,
        dtype=int,
    )

    logger.info(
        "One-Hot Encoding aplicado a columnas: %s",
        columnas_presentes,
    )

    return df_encoded, niveles_categoricos


def eliminar_columnas_no_modelo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina columnas que no deberían entrar al modelo.

    Por ejemplo, CLIENTE es un identificador y puede causar fuga
    o memorización del modelo.

    Args:
        df: DataFrame.

    Returns:
        DataFrame sin columnas excluidas.
    """
    df = df.copy()

    columnas_a_eliminar = [
        columna
        for columna in COLUMNAS_EXCLUIR_MODELO
        if columna in df.columns
    ]

    if columnas_a_eliminar:
        df = df.drop(columns=columnas_a_eliminar)
        logger.info(
            "Columnas eliminadas para modelado: %s",
            columnas_a_eliminar,
        )

    return df


def validar_dataset_final(df: pd.DataFrame) -> None:
    """
    Valida condiciones mínimas del dataset final.

    Args:
        df: DataFrame final.

    Raises:
        ValueError: Si el dataset final no cumple condiciones mínimas.
    """
    if df.empty:
        raise ValueError("El dataset final está vacío.")

    if TARGET not in df.columns:
        raise ValueError(f"No existe la variable objetivo: {TARGET}")

    total_nulos = int(df.isna().sum().sum())

    if total_nulos > 0:
        columnas_con_nulos = df.columns[df.isna().any()].tolist()
        raise ValueError(
            "El dataset final todavía tiene valores nulos. "
            f"Total nulos: {total_nulos}. "
            f"Columnas: {columnas_con_nulos}"
        )

    columnas_object = df.select_dtypes(include=["object"]).columns.tolist()

    if columnas_object:
        raise ValueError(
            "El dataset final todavía tiene columnas tipo object: "
            f"{columnas_object}"
        )

    logger.info("Dataset final validado correctamente.")


def guardar_dataset_final(df: pd.DataFrame) -> None:
    """
    Guarda el dataset final de features en artifacts/.

    Args:
        df: DataFrame final.
    """
    df.to_csv(FEATURES_DATA_PATH, index=False)
    logger.info("Dataset final guardado en: %s", FEATURES_DATA_PATH)


def guardar_preprocessor(metadata: dict[str, Any]) -> None:
    """
    Guarda metadata del preprocesamiento en formato pickle.

    Args:
        metadata: Diccionario con decisiones de preprocesamiento.
    """
    with open(PREPROCESSOR_PATH, "wb") as file:
        pickle.dump(metadata, file)

    logger.info("Preprocessor guardado en: %s", PREPROCESSOR_PATH)


def guardar_metadata_features(metadata: dict[str, Any]) -> None:
    """
    Guarda metadata del proceso de features en JSON.

    Args:
        metadata: Diccionario con metadata.
    """
    with open(FEATURES_METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)

    logger.info("Metadata de features guardada en: %s", FEATURES_METADATA_PATH)


# ============================================================
# Función principal
# ============================================================

def construir_features(path: Path = DATASET_PATH) -> pd.DataFrame:
    """
    Construye el DataFrame final para modelado.

    Args:
        path: Ruta del dataset original.

    Returns:
        DataFrame final procesado y codificado.
    """
    crear_directorios()

    logger.info("Iniciando construcción de features.")

    df = cargar_datos(path)

    logger.info("Dataset original recibido: %d filas x %d columnas", *df.shape)

    df = renombrar_columnas(df)

    columnas_numericas_base = (
        COLUMNAS_CAPPING_CERO
        + VARIABLES_LOG
        + VARIABLES_IMPUTACION_MEDIANA
        + VARIABLES_IMPUTACION_ALEATORIA
    )

    columnas_numericas_base = list(dict.fromkeys(columnas_numericas_base))

    df = convertir_a_numerico(df, columnas_numericas_base)
    df = aplicar_capping_cero(df)
    df = aplicar_logaritmos(df)

    df, metadata_imputacion = imputar_nulos(df)

    df_encoded, niveles_categoricos = codificar_categoricas(df)

    df_final = eliminar_columnas_no_modelo(df_encoded)

    validar_dataset_final(df_final)

    guardar_dataset_final(df_final)

    metadata = {
        "dataset_path": str(path),
        "features_data_path": str(FEATURES_DATA_PATH),
        "target": TARGET,
        "rows": int(df_final.shape[0]),
        "columns": int(df_final.shape[1]),
        "column_names": df_final.columns.tolist(),
        "columnas_renombradas": COLUMNAS_RENAME,
        "columnas_capping_cero": COLUMNAS_CAPPING_CERO,
        "variables_log": VARIABLES_LOG,
        "variables_imputacion_aleatoria": VARIABLES_IMPUTACION_ALEATORIA,
        "variables_imputacion_mediana": VARIABLES_IMPUTACION_MEDIANA,
        "variables_categoricas": VARIABLES_CATEGORICAS,
        "niveles_categoricos": niveles_categoricos,
        "columnas_excluir_modelo": COLUMNAS_EXCLUIR_MODELO,
        "imputacion": metadata_imputacion,
    }

    guardar_preprocessor(metadata)
    guardar_metadata_features(metadata)

    logger.info(
        "Construcción de features finalizada: %d filas x %d columnas",
        *df_final.shape,
    )

    return df_final


# ============================================================
# Ejecución desde terminal
# ============================================================

if __name__ == "__main__":
    try:
        df_features = construir_features()

        print("Construcción de features finalizada correctamente.")
        print(f"Filas: {df_features.shape[0]}")
        print(f"Columnas: {df_features.shape[1]}")
        print(f"Dataset final: {FEATURES_DATA_PATH}")
        print(f"Preprocessor: {PREPROCESSOR_PATH}")

    except Exception as error:
        logger.exception("Error durante la construcción de features: %s", error)
        sys.exit(1)
