"""
test_validate_data.py — Tests de calidad de datos.

Estos tests validan:
1. Renombramiento de columnas.
2. Validación de columnas obligatorias.
3. Distribución del target FLAG_VENTA.
4. Reporte de nulos.
5. Frecuencia de MES.
6. Generación de resumen JSON.
7. Creación de reportes CSV.
8. Creación de gráficos.
9. Flujo completo de validate_data.py sin usar el dataset real.
"""

import json
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


from src import validate_data  # noqa: E402
from src.config import COLUMNAS_RENAME, TARGET  # noqa: E402


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def sample_raw_dataframe() -> pd.DataFrame:
    """
    Crea un DataFrame sintético con estructura similar al dataset original.

    Returns:
        DataFrame de prueba con columnas originales.
    """
    return pd.DataFrame(
        {
            "MES": [1, 1, 2, 2, 3],
            "CLIENTE": [1001, 1002, 1003, 1004, 1005],
            "LINEA_RENOVADO": [1000, 2000, 1500, 3000, 2500],
            "PLAZO_RENOVADO": [12, 24, 18, 36, 24],
            "FLAG_VENTA": [0, 1, 0, 1, 0],
            "USO_LINEA_TOTAL_TC_T2": [10.0, 20.0, np.nan, 30.0, 40.0],
            "USO_TRIM_LINEA_BBVA": [5.0, np.nan, 15.0, 20.0, 25.0],
            "NR_ENTIDADES_TOTAL_T2": [1, 2, 3, 2, 1],
            "DIFF_NRO_ENTIDA_TOTALES_T2_T12": [0, 1, -1, 2, 0],
            "SDO_CONSUMO_T2": [100.0, 200.0, np.nan, 300.0, 400.0],
            "RESENCIA_OFERTA_PLD_RENOVADO": [1, 2, np.nan, 4, 5],
            "Ahorro_Sldo_Bco_T1": [100.0, -5.0, 200.0, 300.0, 400.0],
            "PConsumo_Sldo_Bco_T1": [1000.0, 2000.0, -1.0, 3000.0, 4000.0],
            "SDO_BCO_tot_sm_pasivo_Bco_6M": [500.0, 600.0, 700.0, -2.0, 800.0],
            "EDAD": [25, 35, np.nan, 45, 50],
            "SEXO": ["M", "F", "M", None, "F"],
            "EST_CIVIL": ["SOLTERO", "CASADO", None, "CASADO", "SOLTERO"],
            "ANTIGUEDAD_MES": [12, 24, 36, np.nan, 48],
            "REGION": ["SIERRA", "COSTA", "SIERRA", None, "COSTA"],
            "FLAG_LIMA_PROVINCIA": [1, 0, 1, 0, 1],
            "SUELDO_ESTIMADO": [500.0, 700.0, np.nan, 1000.0, 1200.0],
            "CUBRIR_DEUDA_CONSUMO_SF_RENOVA_PLD": [0.2, 0.5, 0.7, 0.9, 1.0],
        }
    )


@pytest.fixture
def sample_renamed_dataframe(sample_raw_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna el DataFrame con columnas renombradas.

    Args:
        sample_raw_dataframe: DataFrame original de prueba.

    Returns:
        DataFrame con columnas renombradas.
    """
    return validate_data.renombrar_columnas(sample_raw_dataframe)


@pytest.fixture
def patch_output_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict:
    """
    Redirige reportes y figuras a una carpeta temporal.

    Esto evita que los tests escriban sobre la carpeta reports/ real.

    Args:
        monkeypatch: Fixture de pytest.
        tmp_path: Carpeta temporal.

    Returns:
        Diccionario con rutas temporales.
    """
    reports_dir = tmp_path / "reports"
    figures_dir = reports_dir / "figures"

    paths = {
        "reports_dir": reports_dir,
        "figures_dir": figures_dir,
        "validation_summary": reports_dir / "data_validation_summary.json",
        "null_report": reports_dir / "null_report.csv",
        "target_distribution": reports_dir / "target_distribution.csv",
        "mes_distribution": reports_dir / "mes_distribution.csv",
        "describe_report": reports_dir / "describe_report.csv",
        "target_plot": figures_dir / "target_flag_venta.png",
        "missing_matrix": figures_dir / "missing_matrix.png",
    }

    monkeypatch.setattr(validate_data, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(validate_data, "FIGURES_DIR", figures_dir)
    monkeypatch.setattr(
        validate_data,
        "VALIDATION_SUMMARY_PATH",
        paths["validation_summary"],
    )
    monkeypatch.setattr(validate_data, "NULL_REPORT_PATH", paths["null_report"])
    monkeypatch.setattr(
        validate_data,
        "TARGET_DISTRIBUTION_PATH",
        paths["target_distribution"],
    )
    monkeypatch.setattr(validate_data, "MES_DISTRIBUTION_PATH", paths["mes_distribution"])
    monkeypatch.setattr(validate_data, "DESCRIBE_REPORT_PATH", paths["describe_report"])
    monkeypatch.setattr(validate_data, "TARGET_PLOT_PATH", paths["target_plot"])
    monkeypatch.setattr(validate_data, "MISSING_MATRIX_PATH", paths["missing_matrix"])

    return paths


# ============================================================
# Tests de renombramiento y columnas obligatorias
# ============================================================

def test_renombrar_columnas_aplica_diccionario(
    sample_raw_dataframe: pd.DataFrame,
) -> None:
    """
    Valida que renombrar_columnas aplique COLUMNAS_RENAME.
    """
    df = validate_data.renombrar_columnas(sample_raw_dataframe)

    assert "Linea_Renovado" in df.columns
    assert "Plazo_Renovado" in df.columns
    assert "Uso_Linea" in df.columns
    assert "Uso_TrimLinea" in df.columns
    assert "Nro_Entidades" in df.columns
    assert "Saldo_Consumo" in df.columns
    assert "Ahorro" in df.columns
    assert "Prestamo_vigente" in df.columns
    assert "Promed_6Mdeuda" in df.columns
    assert "Deuda_Cubierta%" in df.columns

    assert "LINEA_RENOVADO" not in df.columns
    assert "USO_LINEA_TOTAL_TC_T2" not in df.columns


def test_columnas_rename_existen_en_dataframe_original(
    sample_raw_dataframe: pd.DataFrame,
) -> None:
    """
    Valida que las columnas esperadas para renombrar existan en el dataset de prueba.
    """
    columnas_originales = set(sample_raw_dataframe.columns)
    columnas_rename = set(COLUMNAS_RENAME.keys())

    columnas_faltantes = columnas_rename - columnas_originales

    assert columnas_faltantes == set()


def test_validar_columnas_obligatorias_ok(
    sample_renamed_dataframe: pd.DataFrame,
) -> None:
    """
    Valida que no falle si existen FLAG_VENTA y MES.
    """
    validate_data.validar_columnas_obligatorias(sample_renamed_dataframe)


def test_validar_columnas_obligatorias_falla_sin_target(
    sample_renamed_dataframe: pd.DataFrame,
) -> None:
    """
    Valida que falle si falta FLAG_VENTA.
    """
    df = sample_renamed_dataframe.drop(columns=[TARGET])

    with pytest.raises(ValueError, match="Faltan columnas obligatorias"):
        validate_data.validar_columnas_obligatorias(df)


def test_validar_columnas_obligatorias_falla_sin_mes(
    sample_renamed_dataframe: pd.DataFrame,
) -> None:
    """
    Valida que falle si falta MES.
    """
    df = sample_renamed_dataframe.drop(columns=["MES"])

    with pytest.raises(ValueError, match="Faltan columnas obligatorias"):
        validate_data.validar_columnas_obligatorias(df)


# ============================================================
# Tests de reportes de calidad
# ============================================================

def test_validar_target_retorna_conteo_y_porcentaje(
    sample_renamed_dataframe: pd.DataFrame,
) -> None:
    """
    Valida la distribución absoluta y porcentual de FLAG_VENTA.
    """
    distribution = validate_data.validar_target(sample_renamed_dataframe)

    assert "Conteo Absoluto" in distribution.columns
    assert "Porcentaje (%)" in distribution.columns

    assert int(distribution.loc[0, "Conteo Absoluto"]) == 3
    assert int(distribution.loc[1, "Conteo Absoluto"]) == 2

    assert round(distribution["Porcentaje (%)"].sum(), 2) == 100.00


def test_validar_target_falla_si_no_existe_target(
    sample_renamed_dataframe: pd.DataFrame,
) -> None:
    """
    Valida que validar_target falle si no existe FLAG_VENTA.
    """
    df = sample_renamed_dataframe.drop(columns=[TARGET])

    with pytest.raises(ValueError, match="No existe la columna target"):
        validate_data.validar_target(df)


def test_generar_reporte_nulos_retorna_dataframe(
    sample_renamed_dataframe: pd.DataFrame,
) -> None:
    """
    Valida que el reporte de nulos tenga columnas esperadas.
    """
    null_report = validate_data.generar_reporte_nulos(sample_renamed_dataframe)

    assert isinstance(null_report, pd.DataFrame)
    assert "Nulo" in null_report.columns
    assert "Porcentaje Nulo" in null_report.columns
    assert int(null_report["Nulo"].sum()) > 0


def test_generar_reporte_nulos_identifica_columnas_con_nulos(
    sample_renamed_dataframe: pd.DataFrame,
) -> None:
    """
    Valida que el reporte detecte columnas con nulos.
    """
    null_report = validate_data.generar_reporte_nulos(sample_renamed_dataframe)

    assert null_report.loc["Uso_Linea", "Nulo"] == 1
    assert null_report.loc["Uso_TrimLinea", "Nulo"] == 1
    assert null_report.loc["REGION", "Nulo"] == 1


def test_generar_reporte_mes_retorna_frecuencia(
    sample_renamed_dataframe: pd.DataFrame,
) -> None:
    """
    Valida que la frecuencia de MES se calcule correctamente.
    """
    mes_report = validate_data.generar_reporte_mes(sample_renamed_dataframe)

    assert isinstance(mes_report, pd.DataFrame)
    assert "Cantidad" in mes_report.columns
    assert "Porcentaje" in mes_report.columns

    assert int(mes_report.loc[1, "Cantidad"]) == 2
    assert int(mes_report.loc[2, "Cantidad"]) == 2
    assert int(mes_report.loc[3, "Cantidad"]) == 1


def test_generar_reporte_mes_retorna_vacio_si_no_existe_mes(
    sample_renamed_dataframe: pd.DataFrame,
) -> None:
    """
    Valida que retorne DataFrame vacío si no existe MES.
    """
    df = sample_renamed_dataframe.drop(columns=["MES"])

    mes_report = validate_data.generar_reporte_mes(df)

    assert isinstance(mes_report, pd.DataFrame)
    assert mes_report.empty


# ============================================================
# Tests de JSON y resumen
# ============================================================

def test_convertir_a_json_serializable_convierte_numpy_y_nan() -> None:
    """
    Valida conversión de tipos numpy/pandas a JSON serializable.
    """
    objeto = {
        "entero": np.int64(10),
        "flotante": np.float64(1.5),
        "nan": np.nan,
        "lista": [np.int64(1), np.float64(2.5)],
    }

    convertido = validate_data.convertir_a_json_serializable(objeto)

    assert convertido["entero"] == 10
    assert convertido["flotante"] == 1.5
    assert convertido["nan"] is None
    assert convertido["lista"] == [1, 2.5]


def test_obtener_distribucion_columna_retorna_diccionario(
    sample_renamed_dataframe: pd.DataFrame,
) -> None:
    """
    Valida que obtener_distribucion_columna retorne dict serializable.
    """
    distribucion = validate_data.obtener_distribucion_columna(
        sample_renamed_dataframe,
        TARGET,
    )

    assert isinstance(distribucion, dict)
    assert distribucion["0"] == 3
    assert distribucion["1"] == 2


def test_generar_resumen_validacion_contiene_campos_basicos(
    sample_renamed_dataframe: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """
    Valida que el resumen general contenga campos clave.
    """
    null_report = validate_data.generar_reporte_nulos(sample_renamed_dataframe)

    resumen = validate_data.generar_resumen_validacion(
        df=sample_renamed_dataframe,
        null_report=null_report,
        duplicated_count=int(sample_renamed_dataframe.duplicated().sum()),
        dataset_path=tmp_path / "dataset.csv",
    )

    assert resumen["rows"] == sample_renamed_dataframe.shape[0]
    assert resumen["columns"] == sample_renamed_dataframe.shape[1]
    assert resumen["target"] == TARGET
    assert "column_names" in resumen
    assert "dtypes" in resumen
    assert "target_distribution" in resumen
    assert "mes_distribution" in resumen
    assert resumen["total_null_values"] > 0


def test_guardar_resumen_json_crea_archivo(
    sample_renamed_dataframe: pd.DataFrame,
    patch_output_paths: dict,
) -> None:
    """
    Valida que guardar_resumen_json cree el archivo JSON.
    """
    validate_data.crear_directorios()

    null_report = validate_data.generar_reporte_nulos(sample_renamed_dataframe)

    resumen = validate_data.generar_resumen_validacion(
        df=sample_renamed_dataframe,
        null_report=null_report,
        duplicated_count=0,
        dataset_path=Path("fake_dataset.csv"),
    )

    validate_data.guardar_resumen_json(resumen)

    assert patch_output_paths["validation_summary"].exists()

    with open(patch_output_paths["validation_summary"], "r", encoding="utf-8") as file:
        data = json.load(file)

    assert data["target"] == TARGET
    assert data["rows"] == sample_renamed_dataframe.shape[0]


# ============================================================
# Tests de archivos de salida
# ============================================================

def test_guardar_reportes_crea_archivos_csv(
    sample_renamed_dataframe: pd.DataFrame,
    patch_output_paths: dict,
) -> None:
    """
    Valida que se creen los reportes CSV principales.
    """
    validate_data.crear_directorios()

    target_distribution = validate_data.validar_target(sample_renamed_dataframe)
    null_report = validate_data.generar_reporte_nulos(sample_renamed_dataframe)
    mes_distribution = validate_data.generar_reporte_mes(sample_renamed_dataframe)

    validate_data.guardar_reportes(
        df=sample_renamed_dataframe,
        target_distribution=target_distribution,
        null_report=null_report,
        mes_distribution=mes_distribution,
    )

    assert patch_output_paths["target_distribution"].exists()
    assert patch_output_paths["null_report"].exists()
    assert patch_output_paths["mes_distribution"].exists()
    assert patch_output_paths["describe_report"].exists()


def test_guardar_graficos_crea_archivos_png(
    sample_renamed_dataframe: pd.DataFrame,
    patch_output_paths: dict,
) -> None:
    """
    Valida que se creen los gráficos principales.
    """
    validate_data.crear_directorios()

    validate_data.guardar_grafico_target(sample_renamed_dataframe)
    validate_data.guardar_matriz_nulos(sample_renamed_dataframe)

    assert patch_output_paths["target_plot"].exists()
    assert patch_output_paths["missing_matrix"].exists()


# ============================================================
# Test de flujo completo
# ============================================================

def test_validar_datos_flujo_completo_sin_dataset_real(
    monkeypatch: pytest.MonkeyPatch,
    sample_raw_dataframe: pd.DataFrame,
    patch_output_paths: dict,
) -> None:
    """
    Valida el flujo completo de validar_datos sin usar el CSV real.

    Se reemplaza cargar_datos por una función falsa que retorna
    el DataFrame sintético.
    """
    def fake_cargar_datos(path: Path) -> pd.DataFrame:
        return sample_raw_dataframe.copy()

    monkeypatch.setattr(validate_data, "cargar_datos", fake_cargar_datos)

    df_validado = validate_data.validar_datos(path=Path("fake_dataset.csv"))

    assert isinstance(df_validado, pd.DataFrame)
    assert not df_validado.empty
    assert TARGET in df_validado.columns
    assert "Linea_Renovado" in df_validado.columns
    assert "Uso_Linea" in df_validado.columns

    assert patch_output_paths["validation_summary"].exists()
    assert patch_output_paths["null_report"].exists()
    assert patch_output_paths["target_distribution"].exists()
    assert patch_output_paths["mes_distribution"].exists()
    assert patch_output_paths["describe_report"].exists()
    assert patch_output_paths["target_plot"].exists()
    assert patch_output_paths["missing_matrix"].exists()
