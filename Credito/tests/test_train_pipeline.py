"""
test_train_pipeline.py — Tests del entrenamiento.

Estos tests validan:
1. Separación de X e y.
2. Validación del target.
3. División train/test estratificada.
4. Definición de modelos y grillas.
5. Creación del Pipeline con SMOTE.
6. Cálculo de probabilidades.
7. Cálculo de métricas.
8. Extracción de importancia de variables.
9. Guardado de modelo, métricas, signature, split y CV results.
10. Flujo completo de train() sin ejecutar GridSearchCV real.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier


# ============================================================
# Asegurar importación desde la raíz del proyecto
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src import train_pipeline  # noqa: E402
from src.config import TARGET  # noqa: E402


# ============================================================
# Clases falsas para evitar entrenamiento pesado en tests
# ============================================================

class FakeModel:
    """
    Modelo falso con predict y predict_proba para probar train()
    sin ejecutar GridSearchCV real.
    """

    def predict(self, X):
        return np.array([0 if i % 2 == 0 else 1 for i in range(len(X))])

    def predict_proba(self, X):
        scores = np.linspace(0.20, 0.80, len(X))
        return np.column_stack([1 - scores, scores])


class FakeGridSearch:
    """
    Objeto falso que simula un GridSearchCV ya entrenado.
    """

    def __init__(self):
        self.best_score_ = 0.88
        self.best_params_ = {"model__max_depth": 3}
        self.best_estimator_ = FakeModel()
        self.cv_results_ = {
            "mean_test_roc_auc": [0.88],
            "mean_test_f1": [0.70],
            "mean_test_recall": [0.75],
            "params": [{"model__max_depth": 3}],
        }


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def sample_features_dataframe() -> pd.DataFrame:
    """
    Crea un DataFrame final de features similar al que genera build_features.py.

    Returns:
        DataFrame numérico, sin nulos, con target binario.
    """
    return pd.DataFrame(
        {
            TARGET: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "MES": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
            "Plazo_Renovado": [12, 24, 18, 36, 24, 12, 24, 18, 36, 12, 24, 36],
            "Nro_Entidades": [1, 2, 3, 2, 1, 4, 2, 3, 1, 2, 3, 4],
            "Dif_Entidades": [0, 1, -1, 2, 0, 1, 1, -1, 0, 2, 1, 0],
            "Meses_oferta": [1, 2, 3, 4, 5, 2, 3, 1, 4, 5, 2, 3],
            "EDAD": [25, 35, 40, 45, 50, 60, 30, 32, 44, 55, 48, 39],
            "Flag_LimProv": [1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0],
            "Uso_Linea_LOG": [1.1, 2.0, 1.5, 2.5, 3.0, 1.8, 2.2, 1.6, 2.7, 3.1, 1.9, 2.4],
            "Uso_TrimLinea_LOG": [0.5, 1.0, 0.8, 1.2, 1.4, 0.9, 1.1, 0.7, 1.3, 1.5, 0.6, 1.0],
            "Saldo_Consumo_LOG": [3.1, 4.0, 3.5, 4.5, 5.0, 3.8, 4.2, 3.6, 4.7, 5.1, 3.9, 4.4],
            "SUELDO_ESTIMADO_LOG": [6.2, 6.5, 6.7, 7.0, 7.2, 6.8, 6.4, 6.6, 7.1, 7.3, 6.9, 7.0],
            "REGION_COSTA": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "REGION_SIERRA": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "SEXO_F": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "SEXO_M": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )


@pytest.fixture
def patch_artifact_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict:
    """
    Redirige artifacts a una carpeta temporal.

    Esto evita escribir sobre artifacts/ reales durante los tests.
    """
    artifacts_dir = tmp_path / "artifacts"

    paths = {
        "artifacts_dir": artifacts_dir,
        "model": artifacts_dir / "modelo.pkl",
        "metrics": artifacts_dir / "metrics.json",
        "feature_importance": artifacts_dir / "feature_importance.csv",
        "model_signature": artifacts_dir / "model_signature.json",
        "training_metadata": artifacts_dir / "training_metadata.json",
        "cv_results": artifacts_dir / "cv_results.csv",
        "train_test_split": artifacts_dir / "train_test_split.pkl",
    }

    monkeypatch.setattr(train_pipeline, "ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr(train_pipeline, "MODEL_PATH", paths["model"])
    monkeypatch.setattr(train_pipeline, "METRICS_PATH", paths["metrics"])
    monkeypatch.setattr(
        train_pipeline,
        "FEATURE_IMPORTANCE_PATH",
        paths["feature_importance"],
    )
    monkeypatch.setattr(
        train_pipeline,
        "MODEL_SIGNATURE_PATH",
        paths["model_signature"],
    )
    monkeypatch.setattr(
        train_pipeline,
        "TRAIN_METADATA_PATH",
        paths["training_metadata"],
    )
    monkeypatch.setattr(train_pipeline, "CV_RESULTS_PATH", paths["cv_results"])
    monkeypatch.setattr(
        train_pipeline,
        "TRAIN_TEST_SPLIT_PATH",
        paths["train_test_split"],
    )

    return paths


# ============================================================
# Tests de preparación de X/y
# ============================================================

def test_preparar_xy_retorna_X_y(
    sample_features_dataframe: pd.DataFrame,
) -> None:
    """
    Valida que preparar_xy separe correctamente X e y.
    """
    X, y = train_pipeline.preparar_xy(sample_features_dataframe)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert TARGET not in X.columns
    assert len(X) == len(y)
    assert set(y.unique()).issubset({0, 1})


def test_preparar_xy_falla_sin_target(
    sample_features_dataframe: pd.DataFrame,
) -> None:
    """
    Valida que preparar_xy falle si no existe FLAG_VENTA.
    """
    df = sample_features_dataframe.drop(columns=[TARGET])

    with pytest.raises(ValueError, match="variable objetivo"):
        train_pipeline.preparar_xy(df)


def test_preparar_xy_falla_si_target_tiene_nulos(
    sample_features_dataframe: pd.DataFrame,
) -> None:
    """
    Valida que preparar_xy falle si el target tiene nulos.
    """
    df = sample_features_dataframe.copy()
    df.loc[0, TARGET] = np.nan

    with pytest.raises(ValueError, match="valores no numéricos o nulos"):
        train_pipeline.preparar_xy(df)


def test_preparar_xy_falla_si_X_tiene_object(
    sample_features_dataframe: pd.DataFrame,
) -> None:
    """
    Valida que preparar_xy falle si X contiene columnas object.
    """
    df = sample_features_dataframe.copy()
    df["columna_texto"] = ["A"] * len(df)

    with pytest.raises(ValueError, match="columnas no numéricas"):
        train_pipeline.preparar_xy(df)


def test_preparar_xy_falla_si_X_tiene_nulos(
    sample_features_dataframe: pd.DataFrame,
) -> None:
    """
    Valida que preparar_xy falle si X tiene nulos.
    """
    df = sample_features_dataframe.copy()
    df.loc[0, "EDAD"] = np.nan

    with pytest.raises(ValueError, match="valores nulos en X"):
        train_pipeline.preparar_xy(df)


# ============================================================
# Tests de train/test split
# ============================================================

def test_dividir_train_test_mantiene_filas(
    sample_features_dataframe: pd.DataFrame,
) -> None:
    """
    Valida que train/test split conserve el número total de filas.
    """
    X, y = train_pipeline.preparar_xy(sample_features_dataframe)

    X_train, X_test, y_train, y_test = train_pipeline.dividir_train_test(X, y)

    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)


def test_dividir_train_test_mantiene_ambas_clases(
    sample_features_dataframe: pd.DataFrame,
) -> None:
    """
    Valida que el split estratificado conserve ambas clases en train y test.
    """
    X, y = train_pipeline.preparar_xy(sample_features_dataframe)

    _, _, y_train, y_test = train_pipeline.dividir_train_test(X, y)

    assert set(y_train.unique()) == {0, 1}
    assert set(y_test.unique()) == {0, 1}


# ============================================================
# Tests de modelos, pipeline y probabilidades
# ============================================================

def test_obtener_modelos_y_grids_incluye_modelos_basicos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Valida que existan modelos candidatos básicos.
    """
    monkeypatch.setattr(train_pipeline, "INCLUDE_XGBOOST", False)

    modelos = train_pipeline.obtener_modelos_y_grids()

    assert "decision_tree" in modelos
    assert "random_forest" in modelos

    for _, spec in modelos.items():
        assert "estimator" in spec
        assert "param_grid" in spec
        assert isinstance(spec["param_grid"], dict)


def test_crear_pipeline_contiene_smote_y_modelo() -> None:
    """
    Valida que el pipeline tenga los pasos smote y model.
    """
    modelo = DecisionTreeClassifier(random_state=42)

    pipeline = train_pipeline.crear_pipeline(modelo)

    assert "smote" in pipeline.named_steps
    assert "model" in pipeline.named_steps


def test_obtener_probabilidades_usa_predict_proba(
    sample_features_dataframe: pd.DataFrame,
) -> None:
    """
    Valida que obtener_probabilidades retorne probabilidades de clase positiva.
    """
    X, _ = train_pipeline.preparar_xy(sample_features_dataframe)

    modelo = FakeModel()

    probas = train_pipeline.obtener_probabilidades(modelo, X)

    assert isinstance(probas, np.ndarray)
    assert len(probas) == len(X)
    assert np.all(probas >= 0)
    assert np.all(probas <= 1)


# ============================================================
# Tests de métricas e importancia
# ============================================================

def test_calcular_metricas_retorna_metricas_principales(
    sample_features_dataframe: pd.DataFrame,
) -> None:
    """
    Valida que calcular_metricas retorne métricas esperadas.
    """
    X, y = train_pipeline.preparar_xy(sample_features_dataframe)

    modelo = FakeModel()

    metricas = train_pipeline.calcular_metricas(modelo, X, y)

    for metrica in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        assert metrica in metricas
        assert 0.0 <= metricas[metrica] <= 1.0

    assert "confusion_matrix" in metricas
    assert "classification_report" in metricas


def test_extraer_importancia_variables_con_decision_tree(
    sample_features_dataframe: pd.DataFrame,
) -> None:
    """
    Valida que se extraiga feature_importances_ si el modelo lo soporta.
    """
    X, y = train_pipeline.preparar_xy(sample_features_dataframe)

    modelo = DecisionTreeClassifier(random_state=42)
    modelo.fit(X, y)

    feature_importance = train_pipeline.extraer_importancia_variables(
        modelo=modelo,
        feature_names=X.columns.tolist(),
    )

    assert isinstance(feature_importance, pd.DataFrame)
    assert "feature" in feature_importance.columns
    assert "importance" in feature_importance.columns
    assert len(feature_importance) == X.shape[1]


def test_extraer_importancia_variables_sin_importancia(
    sample_features_dataframe: pd.DataFrame,
) -> None:
    """
    Valida que retorne DataFrame vacío si el modelo no tiene feature_importances_.
    """
    X, _ = train_pipeline.preparar_xy(sample_features_dataframe)

    modelo = FakeModel()

    feature_importance = train_pipeline.extraer_importancia_variables(
        modelo=modelo,
        feature_names=X.columns.tolist(),
    )

    assert isinstance(feature_importance, pd.DataFrame)
    assert list(feature_importance.columns) == ["feature", "importance"]
    assert feature_importance.empty


# ============================================================
# Tests de guardado de artefactos
# ============================================================

def test_guardar_modelo_crea_archivo_pkl(
    patch_artifact_paths: dict,
) -> None:
    """
    Valida que guardar_modelo cree modelo.pkl.
    """
    train_pipeline.crear_directorios()

    modelo = FakeModel()

    train_pipeline.guardar_modelo(modelo)

    assert patch_artifact_paths["model"].exists()

    modelo_cargado = joblib.load(patch_artifact_paths["model"])

    assert hasattr(modelo_cargado, "predict")
    assert hasattr(modelo_cargado, "predict_proba")


def test_guardar_metricas_crea_json(
    patch_artifact_paths: dict,
) -> None:
    """
    Valida que guardar_metricas cree metrics.json.
    """
    train_pipeline.crear_directorios()

    metricas = {
        "accuracy": 0.80,
        "precision": 0.70,
        "recall": 0.75,
        "f1": 0.72,
        "roc_auc": 0.85,
    }

    train_pipeline.guardar_metricas(metricas)

    assert patch_artifact_paths["metrics"].exists()

    with open(patch_artifact_paths["metrics"], "r", encoding="utf-8") as file:
        data = json.load(file)

    assert data["accuracy"] == 0.80
    assert data["roc_auc"] == 0.85


def test_guardar_importancia_variables_crea_csv(
    patch_artifact_paths: dict,
) -> None:
    """
    Valida que guardar_importancia_variables cree feature_importance.csv.
    """
    train_pipeline.crear_directorios()

    df_importance = pd.DataFrame(
        {
            "feature": ["feature_1", "feature_2"],
            "importance": [0.70, 0.30],
        }
    )

    train_pipeline.guardar_importancia_variables(df_importance)

    assert patch_artifact_paths["feature_importance"].exists()

    df_leido = pd.read_csv(patch_artifact_paths["feature_importance"])

    assert df_leido.shape == df_importance.shape


def test_guardar_model_signature_crea_json(
    patch_artifact_paths: dict,
) -> None:
    """
    Valida que guardar_model_signature cree model_signature.json.
    """
    train_pipeline.crear_directorios()

    feature_names = ["feature_1", "feature_2"]

    train_pipeline.guardar_model_signature(
        feature_names=feature_names,
        model_name="fake_model",
    )

    assert patch_artifact_paths["model_signature"].exists()

    with open(patch_artifact_paths["model_signature"], "r", encoding="utf-8") as file:
        data = json.load(file)

    assert data["model_name"] == "fake_model"
    assert data["target"] == TARGET
    assert data["inputs"] == feature_names


def test_guardar_train_test_split_crea_pickle(
    sample_features_dataframe: pd.DataFrame,
    patch_artifact_paths: dict,
) -> None:
    """
    Valida que guardar_train_test_split cree train_test_split.pkl.
    """
    train_pipeline.crear_directorios()

    X, y = train_pipeline.preparar_xy(sample_features_dataframe)
    X_train, X_test, y_train, y_test = train_pipeline.dividir_train_test(X, y)

    train_pipeline.guardar_train_test_split(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    assert patch_artifact_paths["train_test_split"].exists()

    split_data = joblib.load(patch_artifact_paths["train_test_split"])

    assert "X_train" in split_data
    assert "X_test" in split_data
    assert "y_train" in split_data
    assert "y_test" in split_data
    assert split_data["target"] == TARGET


def test_guardar_cv_results_crea_csv(
    patch_artifact_paths: dict,
) -> None:
    """
    Valida que guardar_cv_results cree cv_results.csv.
    """
    train_pipeline.crear_directorios()

    cv_results = [
        pd.DataFrame(
            {
                "model_name": ["fake_model"],
                "mean_test_roc_auc": [0.88],
                "mean_test_f1": [0.70],
            }
        )
    ]

    train_pipeline.guardar_cv_results(cv_results)

    assert patch_artifact_paths["cv_results"].exists()

    df_cv = pd.read_csv(patch_artifact_paths["cv_results"])

    assert df_cv.loc[0, "model_name"] == "fake_model"


def test_guardar_training_metadata_crea_json(
    patch_artifact_paths: dict,
) -> None:
    """
    Valida que guardar_training_metadata cree training_metadata.json.
    """
    train_pipeline.crear_directorios()

    metadata = {
        "target": TARGET,
        "rows": 100,
        "columns": 20,
        "best_model": "fake_model",
    }

    train_pipeline.guardar_training_metadata(metadata)

    assert patch_artifact_paths["training_metadata"].exists()

    with open(patch_artifact_paths["training_metadata"], "r", encoding="utf-8") as file:
        data = json.load(file)

    assert data["target"] == TARGET
    assert data["best_model"] == "fake_model"


# ============================================================
# Test de flujo completo sin GridSearchCV real
# ============================================================

def test_train_flujo_completo_sin_gridsearch_real(
    monkeypatch: pytest.MonkeyPatch,
    sample_features_dataframe: pd.DataFrame,
    patch_artifact_paths: dict,
) -> None:
    """
    Valida el flujo completo de train() sin ejecutar GridSearchCV real.

    Se reemplazan:
    - obtener_modelos_y_grids()
    - entrenar_gridsearch()

    Así el test es rápido y no consume mucha memoria.
    """
    def fake_obtener_modelos_y_grids():
        return {
            "fake_model": {
                "estimator": FakeModel(),
                "param_grid": {"model__dummy": [1]},
            }
        }

    def fake_entrenar_gridsearch(
        model_name,
        estimator,
        param_grid,
        X_train,
        y_train,
    ):
        return FakeGridSearch()

    monkeypatch.setattr(
        train_pipeline,
        "obtener_modelos_y_grids",
        fake_obtener_modelos_y_grids,
    )
    monkeypatch.setattr(
        train_pipeline,
        "entrenar_gridsearch",
        fake_entrenar_gridsearch,
    )

    metricas = train_pipeline.train(df_features=sample_features_dataframe)

    assert isinstance(metricas, dict)
    assert metricas["best_model"] == "fake_model"

    for metrica in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        assert metrica in metricas
        assert 0.0 <= metricas[metrica] <= 1.0

    assert patch_artifact_paths["model"].exists()
    assert patch_artifact_paths["metrics"].exists()
    assert patch_artifact_paths["feature_importance"].exists()
    assert patch_artifact_paths["model_signature"].exists()
    assert patch_artifact_paths["train_test_split"].exists()
    assert patch_artifact_paths["cv_results"].exists()
    assert patch_artifact_paths["training_metadata"].exists()