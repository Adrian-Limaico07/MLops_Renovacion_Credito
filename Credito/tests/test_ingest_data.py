"""
test_ingest_data.py — Tests de carga e integración de fuentes.

Estos tests validan:
1. Que el CSV se cargue correctamente.
2. Que se use correctamente el separador ';'.
3. Que se detecten archivos inexistentes.
4. Que se detecten columnas faltantes.
5. Que se limpien nombres de columnas.
6. Que se genere metadata de ingesta.
7. Que el hash del archivo sea reproducible.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest


# ============================================================
# Asegurar importación desde la raíz del proyecto
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src import ingest_data  # noqa: E402
from src.config import COLUMNAS_REQUERIDAS, TARGET  # noqa: E402


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """
    Crea un DataFrame sintético con todas las columnas requeridas.

    Returns:
        DataFrame de prueba.
    """
    data = {}

    for columna in COLUMNAS_REQUERIDAS:
        if columna == TARGET:
            data[columna] = [0, 1, 0, 1]
        elif columna == "MES":
            data[columna] = [1, 1, 2, 2]
        elif columna == "CLIENTE":
            data[columna] = [1001, 1002, 1003, 1004]
        elif columna == "SEXO":
            data[columna] = ["M", "F", "M", "F"]
        elif columna == "EST_CIVIL":
            data[columna] = ["SOLTERO", "CASADO", "SOLTERO", "CASADO"]
        elif columna == "REGION":
            data[columna] = ["SIERRA", "COSTA", "SIERRA", "COSTA"]
        else:
            data[columna] = [10.0, 20.0, 30.0, 40.0]

    return pd.DataFrame(data)


@pytest.fixture
def sample_csv_path(tmp_path: Path, sample_dataframe: pd.DataFrame) -> Path:
    """
    Guarda un CSV temporal separado por ';'.

    Args:
        tmp_path: Carpeta temporal de pytest.
        sample_dataframe: DataFrame sintético.

    Returns:
        Ruta del CSV temporal.
    """
    path = tmp_path / "sample_renovacion_credito.csv"
    sample_dataframe.to_csv(path, sep=";", index=False)
    return path


@pytest.fixture
def patch_metadata_writer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """
    Redirige la escritura de metadata a una carpeta temporal.

    Esto evita que los tests modifiquen artifacts/ del proyecto real.

    Args:
        monkeypatch: Fixture de pytest para reemplazar objetos.
        tmp_path: Carpeta temporal de pytest.

    Returns:
        Ruta temporal de metadata.
    """
    metadata_path = tmp_path / "ingestion_metadata.json"

    def fake_guardar_metadata(metadata: dict, output_path=None) -> None:
        with open(metadata_path, "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=4, ensure_ascii=False)

    monkeypatch.setattr(ingest_data, "guardar_metadata", fake_guardar_metadata)

    return metadata_path


# ============================================================
# Tests de carga de datos
# ============================================================

def test_cargar_datos_retorna_dataframe(
    sample_csv_path: Path,
    patch_metadata_writer: Path,
) -> None:
    """
    Valida que cargar_datos retorne un DataFrame.
    """
    df = ingest_data.cargar_datos(sample_csv_path)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert df.shape[0] == 4


def test_cargar_datos_lee_columnas_correctamente_con_separador_punto_coma(
    sample_csv_path: Path,
    patch_metadata_writer: Path,
) -> None:
    """
    Valida que el CSV no se cargue como una sola columna.

    Este test evita el error anterior donde el archivo separado por ';'
    se leía como si tuviera una sola columna.
    """
    df = ingest_data.cargar_datos(sample_csv_path)

    assert df.shape[1] == len(COLUMNAS_REQUERIDAS)
    assert TARGET in df.columns
    assert "MES" in df.columns
    assert "CLIENTE" in df.columns


def test_cargar_datos_contiene_columnas_requeridas(
    sample_csv_path: Path,
    patch_metadata_writer: Path,
) -> None:
    """
    Valida que el dataset cargado contenga todas las columnas requeridas.
    """
    df = ingest_data.cargar_datos(sample_csv_path)

    columnas_faltantes = COLUMNAS_REQUERIDAS - set(df.columns)

    assert columnas_faltantes == set()


def test_cargar_datos_archivo_inexistente_lanza_error(tmp_path: Path) -> None:
    """
    Valida que cargar_datos falle si el archivo no existe.
    """
    ruta_inexistente = tmp_path / "archivo_no_existe.csv"

    with pytest.raises(FileNotFoundError):
        ingest_data.cargar_datos(ruta_inexistente)


def test_cargar_datos_columnas_faltantes_lanza_error(
    tmp_path: Path,
    sample_dataframe: pd.DataFrame,
    patch_metadata_writer: Path,
) -> None:
    """
    Valida que cargar_datos falle si faltan columnas obligatorias.
    """
    df_incompleto = sample_dataframe.drop(columns=[TARGET])

    path = tmp_path / "sample_incompleto.csv"
    df_incompleto.to_csv(path, sep=";", index=False)

    with pytest.raises(ValueError, match="Columnas faltantes"):
        ingest_data.cargar_datos(path)


# ============================================================
# Tests de funciones auxiliares
# ============================================================

def test_limpiar_nombres_columnas_elimina_espacios_y_bom() -> None:
    """
    Valida que limpiar_nombres_columnas quite espacios y BOM.
    """
    df = pd.DataFrame(
        {
            "\ufeffMES ": [1, 2],
            " FLAG_VENTA ": [0, 1],
        }
    )

    df_limpio = ingest_data.limpiar_nombres_columnas(df)

    assert "MES" in df_limpio.columns
    assert "FLAG_VENTA" in df_limpio.columns
    assert "\ufeffMES " not in df_limpio.columns
    assert " FLAG_VENTA " not in df_limpio.columns


def test_validar_columnas_no_lanza_error_con_columnas_completas(
    sample_dataframe: pd.DataFrame,
) -> None:
    """
    Valida que validar_columnas no falle cuando están todas las columnas.
    """
    ingest_data.validar_columnas(sample_dataframe)


def test_validar_columnas_lanza_error_si_falta_columna(
    sample_dataframe: pd.DataFrame,
) -> None:
    """
    Valida que validar_columnas falle cuando falta una columna requerida.
    """
    df_incompleto = sample_dataframe.drop(columns=[TARGET])

    with pytest.raises(ValueError, match="Columnas faltantes"):
        ingest_data.validar_columnas(df_incompleto)


def test_calcular_hash_archivo_es_reproducible(sample_csv_path: Path) -> None:
    """
    Valida que el hash del mismo archivo sea siempre igual.
    """
    hash_1 = ingest_data.calcular_hash_archivo(sample_csv_path)
    hash_2 = ingest_data.calcular_hash_archivo(sample_csv_path)

    assert isinstance(hash_1, str)
    assert len(hash_1) == 32
    assert hash_1 == hash_2


def test_generar_metadata_contiene_campos_basicos(
    sample_csv_path: Path,
    sample_dataframe: pd.DataFrame,
) -> None:
    """
    Valida que generar_metadata cree los campos esperados.
    """
    metadata = ingest_data.generar_metadata(sample_dataframe, sample_csv_path)

    assert metadata["source"] == str(sample_csv_path)
    assert metadata["filename"] == sample_csv_path.name
    assert metadata["rows"] == sample_dataframe.shape[0]
    assert metadata["columns"] == sample_dataframe.shape[1]
    assert metadata["target"] == TARGET
    assert "column_names" in metadata
    assert "file_hash_md5" in metadata
    assert "timestamp_utc" in metadata
    assert "target_distribution" in metadata

    total_target = sum(int(valor) for valor in metadata["target_distribution"].values())

    assert total_target == sample_dataframe.shape[0]


def test_guardar_metadata_crea_archivo_json(
    tmp_path: Path,
    sample_csv_path: Path,
    sample_dataframe: pd.DataFrame,
) -> None:
    """
    Valida que guardar_metadata genere un archivo JSON.
    """
    metadata = ingest_data.generar_metadata(sample_dataframe, sample_csv_path)

    output_path = tmp_path / "metadata.json"

    ingest_data.guardar_metadata(metadata, output_path)

    assert output_path.exists()

    with open(output_path, "r", encoding="utf-8") as file:
        metadata_leida = json.load(file)

    assert metadata_leida["filename"] == sample_csv_path.name
    assert metadata_leida["rows"] == sample_dataframe.shape[0]