"""
config.py — Configuración general del proyecto de renovación de crédito.
"""

from pathlib import Path


# ============================================================
# Rutas generales del proyecto
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

DATASET_NAME = "Dataset Renovacion_prestamo.csv"
DATASET_PATH = DATA_DIR / DATASET_NAME

INGESTION_METADATA_PATH = ARTIFACTS_DIR / "ingestion_metadata.json"


# ============================================================
# Configuración de lectura del CSV
# ============================================================

CSV_SEPARATOR = ";"
CSV_DECIMAL = "."
CSV_ENCODING = "utf-8-sig"


# ============================================================
# Variable objetivo
# ============================================================

TARGET = "FLAG_VENTA"


# ============================================================
# Columnas requeridas del dataset original
# ============================================================

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


# ============================================================
# Diccionario de renombramiento
# Se usará principalmente en build_features.py
# ============================================================

COLUMNAS_RENAME = {
    "LINEA_RENOVADO": "Linea_Renovado",
    "PLAZO_RENOVADO": "Plazo_Renovado",
    "USO_LINEA_TOTAL_TC_T2": "Uso_Linea",
    "USO_TRIM_LINEA_BBVA": "Uso_TrimLinea",
    "NR_ENTIDADES_TOTAL_T2": "Nro_Entidades",
    "DIFF_NRO_ENTIDA_TOTALES_T2_T12": "Dif_Entidades",
    "SDO_CONSUMO_T2": "Saldo_Consumo",
    "RESENCIA_OFERTA_PLD_RENOVADO": "Meses_oferta",
    "Ahorro_Sldo_Bco_T1": "Ahorro",
    "PConsumo_Sldo_Bco_T1": "Prestamo_vigente",
    "SDO_BCO_tot_sm_pasivo_Bco_6M": "Promed_6Mdeuda",
    "FLAG_LIMA_PROVINCIA": "Flag_LimProv",
    "CUBRIR_DEUDA_CONSUMO_SF_RENOVA_PLD": "Deuda_Cubierta%",
}


# ============================================================
# Parámetros generales de modelado
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.30

RECALL_MIN = 0.65
ROC_AUC_MIN = 0.65
F1_MIN = 0.60