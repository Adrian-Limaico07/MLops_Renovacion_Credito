"""
test_build_features.py — Tests de variables, imputaciones y encoding.

Estos tests validan:
1. Renombramiento de columnas.
2. Conversión de columnas a numérico.
3. Capping inferior a cero.
4. Creación de variables logarítmicas.
5. Imputación aleatoria.
6. Imputación por mediana.
7. Imputación por moda.
8. One-Hot Encoding.
9. Eliminación de columnas no usadas en modelado.
10. Validación del dataset final.
11. Guardado de dataset final y metadata.
12. Flujo completo de construir_features sin usar el dataset real.
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ============================================================
# Asegurar importación desde la raíz del proyecto
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src import build_features  # noqa: E402
from src.config import TARGET  # noqa: E402


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def sample_raw_dataframe() -> pd.DataFrame:
    """
    Crea un DataFrame sintético similar al dataset original.

    Returns:
        DataFrame con columnas originales antes de renombrar.
    """
    return pd.DataFrame(
        {
            "MES": [1, 1, 2, 2, 3, 3],
            "CLIENTE": [1001, 1002, 1003, 1004, 1005, 1006],
            "LINEA_RENOVADO": [1000, 2000, 1500, 3000, 2500, 1800],
            "PLAZO_RENOVADO": [12, 24, 18, 36, 24, 12],
            "FLAG_VENTA": [0, 1, 0, 1, 0, 1],
            "USO_LINEA_TOTAL_TC_T2": [10.0, 20.0, np.nan, 30.0, 40.0, np.nan],
            "USO_TRIM_LINEA_BBVA": [5.0, np.nan, 15.0, 20.0, 25.0, np.nan],
            "NR_ENTIDADES_TOTAL_T2": [1, 2, 3, 2, 1, 4],
            "DIFF_NRO_ENTIDA_TOTALES_T2_T12": [0, 1, -1, 2, 0, 1],
            "SDO_CONSUMO_T2": [100.0, 200.0, np.nan, 300.0, 400.0, 500.0],
            "RESENCIA_OFERTA_PLD_RENOVADO": [1, 2, np.nan, 4, 5, np.nan],
            "Ahorro_Sldo_Bco_T1": [100.0, -5.0, 200.0, 300.0, 400.0, -10.0],
            "PConsumo_Sldo_Bco_T1": [1000.0, 2000.0, -1.0, 3000.0, 4000.0, 5000.0],
            "SDO_BCO_tot_sm_pasivo_Bco_6M": [500.0, 600.0, 700.0, -2.0, 800.0, 900.0],
            "EDAD": [25, 35, np.nan, 45, 50, 60],
            "SEXO": ["M", "F", "M", None, "F", "M"],
            "EST_CIVIL": ["SOLTERO", "CASADO", None, "CASADO", "SOLTERO", "DIVORCIADO"],
            "ANTIGUEDAD_MES": [12, 24, 36, np.nan, 48, 60],
            "REGION": ["SIERRA", "COSTA", "SIERRA", None, "COSTA", "ORIENTE"],
            "FLAG_LIMA_PROVINCIA": [1, 0, 1, 0, 1, 0],
            "SUELDO_ESTIMADO": [500.0, 700.0, np.nan, 1000.0, 1200.0, 900.0],
            "CUBRIR_DEUDA_CONSUMO_SF_RENOVA_PLD": [0.2, 0.5, 0.7, 0.9, 1.0, 0.3],
        }
    )


@pytest.fixture
def sample_renamed_dataframe(sample_raw_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna el DataFrame con columnas renombradas.

    Args:
        sample_raw_dataframe: DataFrame original de prueba.

    Returns:
        DataFrame renombrado.
    """
    return build_features.renombrar_columnas(sample_raw_dataframe)


@pytest.fixture
def patch_artifact_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict:
    """
    Redirige artifacts a una carpeta temporal.

    Esto evita modificar artifacts/ reales durante los tests.

    Args:
        monkeypatch: Fixture de pytest.
        tmp_path: Carpeta temporal.

    Returns:
        Diccionario con rutas temporales.
    """
    artifacts_dir = tmp_path / "artifacts"

    paths = {
        "artifacts_dir": artifacts_dir,
        "features_data": artifacts_dir / "df_features.csv",
        "preprocessor": artifacts_dir / "preprocessor.pkl",
        "features_metadata": artifacts_dir / "features_metadata.json",
    }

    monkeypatch.setattr(build_features, "ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr(build_features, "FEATURES_DATA_PATH", paths["features_data"])
    monkeypatch.setattr(build_features, "PREPROCESSOR_PATH", paths["preprocessor"])
    monkeypatch.setattr(
        build_features,
        "FEATURES_METADATA_PATH",
        paths["features_metadata"],
    )

    return paths


# ============================================================
# Tests de renombramiento y conversión
# ============================================================

def test_renombrar_columnas_crea_nombres_interpretables(
    sample_raw_dataframe: pd.DataFrame,
) -> None:
    """
    Valida que las columnas originales sean renombradas.
    """
    df = build_features.renombrar_columnas(sample_raw_dataframe)

    assert "Linea_Renovado" in df.columns
    assert "Plazo_Renovado" in df.columns
    assert "Uso_Linea" in df.columns
    assert "Uso_TrimLinea" in df.columns
    assert "Nro_Entidades" in df.columns
    assert "Dif_Entidades" in df.columns
    assert "Saldo_Consumo" in df.columns
    assert "Meses_oferta" in df.columns
    assert "Ahorro" in df.columns
    assert "Prestamo_vigente" in df.columns
    assert "Promed_6Mdeuda" in df.columns
    assert "Flag_LimProv" in df.columns
    assert "Deuda_Cubierta%" in df.columns

    assert "LINEA_RENOVADO" not in df.columns
    assert "USO_LINEA_TOTAL_TC_T2" not in df.columns


def test_convertir_a_numerico_convierte_columnas(
    sample_renamed_dataframe: pd.DataFrame,
) -> None:
    """
    Valida que convertir_a_numerico convierta columnas indicadas a numérico.
    """
    df = sample_renamed_dataframe.copy()
    df["Uso_Linea"] = df["Uso_Linea"].astype(str)

    df_convertido = build_features.convertir_a_numerico(df, ["Uso_Linea"])

    assert pd.api.types.is_numeric_dtype(df_convertido["Uso_Linea"])


# ============================================================
# Tests de capping y logs
# ============================================================

def test_aplicar_capping_cero_elimina_negativos(
    sample_renamed_dataframe: pd.DataFrame,
) -> None:
    """
    Valida que Ahorro, Prestamo_vigente y Promed_6Mdeuda no queden negativos.
    """
    df = build_features.aplicar_capping_cero(sample_renamed_dataframe)

    assert (df["Ahorro"] < 0).sum() == 0
    assert (df["Prestamo_vigente"] < 0).sum() == 0
    assert (df["Promed_6Mdeuda"] < 0).sum() == 0

    assert df["Ahorro"].min() == 0
    assert df["Prestamo_vigente"].min() == 0
    assert df["Promed_6Mdeuda"].min() == 0


def test_aplicar_logaritmos_crea_columnas_log(
    sample_renamed_dataframe: pd.DataFrame,
) -> None:
    """
    Valida que se creen columnas terminadas en _LOG.
    """
    df = build_features.aplicar_capping_cero(sample_renamed_dataframe)
    df_log = build_features.aplicar_logaritmos(df)

    columnas_esperadas = [
        "Uso_Linea_LOG",
        "Uso_TrimLinea_LOG",
        "Saldo_Consumo_LOG",
        "SUELDO_ESTIMADO_LOG",
        "ANTIGUEDAD_MES_LOG",
        "Linea_Renovado_LOG",
        "Ahorro_LOG",
        "Prestamo_vigente_LOG",
        "Promed_6Mdeuda_LOG",
        "Deuda_Cubierta%_LOG",
    ]

    for columna in columnas_esperadas:
        assert columna in df_log.columns


def test_aplicar_logaritmos_no_genera_infinitos(
    sample_renamed_dataframe: pd.DataFrame,
) -> None:
    """
    Valida que las columnas LOG no tengan infinitos.
    """
    df = build_features.aplicar_capping_cero(sample_renamed_dataframe)
    df_log = build_features.aplicar_logaritmos(df)

    log_cols = [col for col in df_log.columns if col.endswith("_LOG")]

    assert log_cols

    for col in log_cols:
        assert np.isinf(df_log[col].dropna()).sum() == 0


# ============================================================
# Tests de imputación
# ============================================================

def test_imputar_aleatorio_uniforme_rellena_nulos(
    sample_renamed_dataframe: pd.DataFrame,
) -> None:
    """
    Valida que la imputación aleatoria rellene nulos.
    """
    rng = np.random.default_rng(42)

    df = build_features.aplicar_logaritmos(sample_renamed_dataframe)

    assert df["Uso_Linea_LOG"].isna().sum() > 0

    df_imputado, metadata = build_features.imputar_aleatorio_uniforme(
        df=df,
        columna="Uso_Linea_LOG",
        rng=rng,
    )

    assert df_imputado["Uso_Linea_LOG"].isna().sum() == 0
    assert metadata["columna"] == "Uso_Linea_LOG"
    assert metadata["metodo"] == "random_uniform_mean_std"
    assert metadata["nulos_imputados"] > 0


def test_imputar_mediana_rellena_nulos(
    sample_renamed_dataframe: pd.DataFrame,
) -> None:
    """
    Valida que la imputación por mediana rellene nulos.
    """
    df = build_features.aplicar_logaritmos(sample_renamed_dataframe)

    assert df["Saldo_Consumo_LOG"].isna().sum() > 0

    df_imputado, metadata = build_features.imputar_mediana(
        df=df,
        columna="Saldo_Consumo_LOG",
    )

    assert df_imputado["Saldo_Consumo_LOG"].isna().sum() == 0
    assert metadata["columna"] == "Saldo_Consumo_LOG"
    assert metadata["metodo"] == "median"
    assert metadata["nulos_imputados"] > 0
    assert metadata["valor_imputacion"] is not None


def test_imputar_moda_categoricas_rellena_nulos(
    sample_renamed_dataframe: pd.DataFrame,
) -> None:
    """
    Valida que la imputación por moda rellene variables categóricas.
    """
    assert sample_renamed_dataframe["REGION"].isna().sum() > 0
    assert sample_renamed_dataframe["SEXO"].isna().sum() > 0
    assert sample_renamed_dataframe["EST_CIVIL"].isna().sum() > 0

    df_imputado, metadata = build_features.imputar_moda_categoricas(
        sample_renamed_dataframe,
    )

    assert df_imputado["REGION"].isna().sum() == 0
    assert df_imputado["SEXO"].isna().sum() == 0
    assert df_imputado["EST_CIVIL"].isna().sum() == 0

    assert isinstance(metadata, list)
    assert len(metadata) == 3


def test_imputar_nulos_rellena_columnas_definidas(
    sample_renamed_dataframe: pd.DataFrame,
) -> None:
    """
    Valida que imputar_nulos ejecute imputaciones definidas.
    """
    df = build_features.aplicar_capping_cero(sample_renamed_dataframe)
    df = build_features.aplicar_logaritmos(df)

    df_imputado, metadata = build_features.imputar_nulos(df)

    columnas_imputadas = [
        "Uso_TrimLinea_LOG",
        "Uso_Linea_LOG",
        "Meses_oferta",
        "Saldo_Consumo_LOG",
        "SUELDO_ESTIMADO_LOG",
        "ANTIGUEDAD_MES_LOG",
        "EDAD",
        "REGION",
        "SEXO",
        "EST_CIVIL",
    ]

    for columna in columnas_imputadas:
        assert columna in df_imputado.columns
        assert df_imputado[columna].isna().sum() == 0

    assert "random_uniform" in metadata
    assert "median" in metadata
    assert "mode" in metadata


# ============================================================
# Tests de encoding y columnas finales
# ============================================================

def test_codificar_categoricas_crea_dummies(
    sample_renamed_dataframe: pd.DataFrame,
) -> None:
    """
    Valida que One-Hot Encoding cree columnas dummy.
    """
    df = sample_renamed_dataframe.copy()

    df["REGION"] = df["REGION"].fillna("SIERRA")
    df["SEXO"] = df["SEXO"].fillna("M")
    df["EST_CIVIL"] = df["EST_CIVIL"].fillna("SOLTERO")

    df_encoded, niveles = build_features.codificar_categoricas(df)

    assert "REGION" not in df_encoded.columns
    assert "SEXO" not in df_encoded.columns
    assert "EST_CIVIL" not in df_encoded.columns

    assert any(col.startswith("REGION_") for col in df_encoded.columns)
    assert any(col.startswith("SEXO_") for col in df_encoded.columns)
    assert any(col.startswith("EST_CIVIL_") for col in df_encoded.columns)

    assert "REGION" in niveles
    assert "SEXO" in niveles
    assert "EST_CIVIL" in niveles


def test_eliminar_columnas_no_modelo_elimina_cliente_y_originales(
    sample_renamed_dataframe: pd.DataFrame,
) -> None:
    """
    Valida que se eliminen columnas no usadas en modelado.
    """
    df = build_features.eliminar_columnas_no_modelo(sample_renamed_dataframe)

    assert "CLIENTE" not in df.columns

    # Estas columnas deberían excluirse si fueron reemplazadas por versiones LOG.
    columnas_originales_log = [
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

    for columna in columnas_originales_log:
        assert columna not in df.columns


def test_validar_dataset_final_ok_con_dataframe_limpio() -> None:
    """
    Valida que validar_dataset_final no falle con DataFrame limpio.
    """
    df = pd.DataFrame(
        {
            TARGET: [0, 1, 0],
            "feature_1": [1.0, 2.0, 3.0],
            "feature_2": [0, 1, 0],
        }
    )

    build_features.validar_dataset_final(df)


def test_validar_dataset_final_falla_si_hay_nulos() -> None:
    """
    Valida que validar_dataset_final falle si hay nulos.
    """
    df = pd.DataFrame(
        {
            TARGET: [0, 1, 0],
            "feature_1": [1.0, np.nan, 3.0],
        }
    )

    with pytest.raises(ValueError, match="valores nulos"):
        build_features.validar_dataset_final(df)


def test_validar_dataset_final_falla_si_hay_object() -> None:
    """
    Valida que validar_dataset_final falle si quedan columnas object.
    """
    df = pd.DataFrame(
        {
            TARGET: [0, 1, 0],
            "feature_texto": ["A", "B", "C"],
        }
    )

    with pytest.raises(ValueError, match="columnas tipo object"):
        build_features.validar_dataset_final(df)


def test_validar_dataset_final_falla_sin_target() -> None:
    """
    Valida que validar_dataset_final falle si no existe FLAG_VENTA.
    """
    df = pd.DataFrame(
        {
            "feature_1": [1.0, 2.0, 3.0],
        }
    )

    with pytest.raises(ValueError, match="variable objetivo"):
        build_features.validar_dataset_final(df)


# ============================================================
# Tests de guardado de artifacts
# ============================================================

def test_guardar_dataset_final_crea_csv(
    patch_artifact_paths: dict,
) -> None:
    """
    Valida que guardar_dataset_final cree df_features.csv.
    """
    build_features.crear_directorios()

    df = pd.DataFrame(
        {
            TARGET: [0, 1],
            "feature_1": [1.0, 2.0],
        }
    )

    build_features.guardar_dataset_final(df)

    assert patch_artifact_paths["features_data"].exists()

    df_leido = pd.read_csv(patch_artifact_paths["features_data"])

    assert df_leido.shape == df.shape


def test_guardar_preprocessor_crea_pickle(
    patch_artifact_paths: dict,
) -> None:
    """
    Valida que guardar_preprocessor cree preprocessor.pkl.
    """
    build_features.crear_directorios()

    metadata = {
        "variables_log": ["Uso_Linea"],
        "variables_categoricas": ["REGION"],
    }

    build_features.guardar_preprocessor(metadata)

    assert patch_artifact_paths["preprocessor"].exists()

    with open(patch_artifact_paths["preprocessor"], "rb") as file:
        metadata_leida = pickle.load(file)

    assert metadata_leida["variables_log"] == ["Uso_Linea"]


def test_guardar_metadata_features_crea_json(
    patch_artifact_paths: dict,
) -> None:
    """
    Valida que guardar_metadata_features cree features_metadata.json.
    """
    build_features.crear_directorios()

    metadata = {
        "rows": 10,
        "columns": 5,
        "target": TARGET,
    }

    build_features.guardar_metadata_features(metadata)

    assert patch_artifact_paths["features_metadata"].exists()

    with open(patch_artifact_paths["features_metadata"], "r", encoding="utf-8") as file:
        metadata_leida = json.load(file)

    assert metadata_leida["target"] == TARGET


# ============================================================
# Test de flujo completo
# ============================================================

def test_construir_features_flujo_completo_sin_dataset_real(
    monkeypatch: pytest.MonkeyPatch,
    sample_raw_dataframe: pd.DataFrame,
    patch_artifact_paths: dict,
) -> None:
    """
    Valida el flujo completo de construir_features sin usar el CSV real.

    Se reemplaza cargar_datos por una función falsa que retorna
    el DataFrame sintético.
    """
    def fake_cargar_datos(path: Path) -> pd.DataFrame:
        return sample_raw_dataframe.copy()

    monkeypatch.setattr(build_features, "cargar_datos", fake_cargar_datos)

    df_final = build_features.construir_features(path=Path("fake_dataset.csv"))

    assert isinstance(df_final, pd.DataFrame)
    assert not df_final.empty

    assert TARGET in df_final.columns
    assert "CLIENTE" not in df_final.columns

    assert df_final.isna().sum().sum() == 0
    assert df_final.select_dtypes(include=["object"]).empty

    assert "Uso_Linea_LOG" in df_final.columns
    assert "Uso_TrimLinea_LOG" in df_final.columns
    assert "Saldo_Consumo_LOG" in df_final.columns
    assert "SUELDO_ESTIMADO_LOG" in df_final.columns

    assert any(col.startswith("REGION_") for col in df_final.columns)
    assert any(col.startswith("SEXO_") for col in df_final.columns)
    assert any(col.startswith("EST_CIVIL_") for col in df_final.columns)

    assert patch_artifact_paths["features_data"].exists()
    assert patch_artifact_paths["preprocessor"].exists()
    assert patch_artifact_paths["features_metadata"].exists()
