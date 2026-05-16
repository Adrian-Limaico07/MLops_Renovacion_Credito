"""
test_model.py — Tests del modelo serializado.

Estos tests validan:
1. Que un modelo entrenado pueda serializarse como modelo.pkl.
2. Que el modelo pueda cargarse correctamente.
3. Que tenga métodos predict y predict_proba.
4. Que las predicciones tengan el tamaño correcto.
5. Que las predicciones sean binarias.
6. Que predict_proba retorne probabilidades válidas.
7. Que las probabilidades sumen 1 por fila.
8. Que el modelo funcione con un DataFrame con nombres de columnas.
9. Que se pueda guardar y leer una firma del modelo.
10. Que se pueda guardar y leer un archivo de métricas.
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


from src.config import TARGET  # noqa: E402


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def sample_features_dataframe() -> pd.DataFrame:
    """
    Crea un DataFrame final de features similar al generado por build_features.py.

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
def X_y(sample_features_dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separa X e y del DataFrame sintético.

    Args:
        sample_features_dataframe: DataFrame de prueba.

    Returns:
        X, y.
    """
    X = sample_features_dataframe.drop(columns=[TARGET])
    y = sample_features_dataframe[TARGET].astype(int)

    return X, y


@pytest.fixture
def trained_model(X_y: tuple[pd.DataFrame, pd.Series]) -> DecisionTreeClassifier:
    """
    Entrena un modelo liviano para pruebas.

    Args:
        X_y: Tupla con X e y.

    Returns:
        Modelo entrenado.
    """
    X, y = X_y

    model = DecisionTreeClassifier(
        max_depth=3,
        random_state=42,
    )

    model.fit(X, y)

    return model


@pytest.fixture
def serialized_model_path(
    tmp_path: Path,
    trained_model: DecisionTreeClassifier,
) -> Path:
    """
    Serializa un modelo de prueba en una carpeta temporal.

    Args:
        tmp_path: Carpeta temporal de pytest.
        trained_model: Modelo entrenado.

    Returns:
        Ruta del modelo serializado.
    """
    model_path = tmp_path / "modelo.pkl"

    joblib.dump(trained_model, model_path)

    return model_path


@pytest.fixture
def loaded_model(serialized_model_path: Path):
    """
    Carga el modelo serializado.

    Args:
        serialized_model_path: Ruta del modelo serializado.

    Returns:
        Modelo cargado.
    """
    return joblib.load(serialized_model_path)


@pytest.fixture
def model_signature_path(
    tmp_path: Path,
    X_y: tuple[pd.DataFrame, pd.Series],
) -> Path:
    """
    Crea una firma de modelo sintética.

    Args:
        tmp_path: Carpeta temporal.
        X_y: Tupla con X e y.

    Returns:
        Ruta del JSON de firma.
    """
    X, _ = X_y

    signature = {
        "model_name": "decision_tree_test",
        "prediction_type": "binary_classification",
        "target": TARGET,
        "inputs": X.columns.tolist(),
        "outputs": {
            "prediction": "Clase predicha 0/1",
            "probability": "Probabilidad estimada de FLAG_VENTA = 1",
        },
        "n_features": X.shape[1],
    }

    path = tmp_path / "model_signature.json"

    with open(path, "w", encoding="utf-8") as file:
        json.dump(signature, file, indent=4, ensure_ascii=False)

    return path


@pytest.fixture
def metrics_path(tmp_path: Path) -> Path:
    """
    Crea un metrics.json sintético.

    Args:
        tmp_path: Carpeta temporal.

    Returns:
        Ruta del archivo de métricas.
    """
    metrics = {
        "accuracy": 0.85,
        "precision": 0.80,
        "recall": 0.75,
        "f1": 0.77,
        "roc_auc": 0.88,
        "confusion_matrix": [[5, 1], [2, 4]],
    }

    path = tmp_path / "metrics.json"

    with open(path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4, ensure_ascii=False)

    return path


# ============================================================
# Tests del archivo serializado
# ============================================================

def test_model_file_exists(serialized_model_path: Path) -> None:
    """
    Valida que el archivo modelo.pkl exista.
    """
    assert serialized_model_path.exists()
    assert serialized_model_path.name == "modelo.pkl"


def test_model_file_is_not_empty(serialized_model_path: Path) -> None:
    """
    Valida que el archivo modelo.pkl no esté vacío.
    """
    assert serialized_model_path.stat().st_size > 0


def test_model_can_be_loaded(serialized_model_path: Path) -> None:
    """
    Valida que el modelo pueda cargarse con joblib.
    """
    model = joblib.load(serialized_model_path)

    assert model is not None


def test_loaded_model_is_decision_tree(loaded_model) -> None:
    """
    Valida que el modelo cargado sea un DecisionTreeClassifier.
    """
    assert isinstance(loaded_model, DecisionTreeClassifier)


# ============================================================
# Tests de métodos del modelo
# ============================================================

def test_loaded_model_has_predict(loaded_model) -> None:
    """
    Valida que el modelo tenga método predict.
    """
    assert hasattr(loaded_model, "predict")


def test_loaded_model_has_predict_proba(loaded_model) -> None:
    """
    Valida que el modelo tenga método predict_proba.
    """
    assert hasattr(loaded_model, "predict_proba")


def test_loaded_model_has_classes_attribute(loaded_model) -> None:
    """
    Valida que el modelo tenga atributo classes_ después de entrenarse.
    """
    assert hasattr(loaded_model, "classes_")
    assert set(loaded_model.classes_).issubset({0, 1})


# ============================================================
# Tests de predicción
# ============================================================

def test_model_predict_returns_same_number_of_rows(
    loaded_model,
    X_y: tuple[pd.DataFrame, pd.Series],
) -> None:
    """
    Valida que predict retorne una predicción por fila.
    """
    X, _ = X_y

    y_pred = loaded_model.predict(X)

    assert len(y_pred) == len(X)


def test_model_predict_returns_binary_values(
    loaded_model,
    X_y: tuple[pd.DataFrame, pd.Series],
) -> None:
    """
    Valida que las predicciones sean binarias.
    """
    X, _ = X_y

    y_pred = loaded_model.predict(X)

    assert set(np.unique(y_pred)).issubset({0, 1})


def test_model_predict_accepts_dataframe_with_feature_names(
    loaded_model,
    X_y: tuple[pd.DataFrame, pd.Series],
) -> None:
    """
    Valida que el modelo acepte DataFrame con nombres de columnas.
    """
    X, _ = X_y

    y_pred = loaded_model.predict(X)

    assert isinstance(y_pred, np.ndarray)
    assert len(y_pred) == X.shape[0]


# ============================================================
# Tests de probabilidades
# ============================================================

def test_model_predict_proba_shape(
    loaded_model,
    X_y: tuple[pd.DataFrame, pd.Series],
) -> None:
    """
    Valida que predict_proba tenga forma (n_filas, 2).
    """
    X, _ = X_y

    proba = loaded_model.predict_proba(X)

    assert proba.shape == (X.shape[0], 2)


def test_model_predict_proba_values_are_between_zero_and_one(
    loaded_model,
    X_y: tuple[pd.DataFrame, pd.Series],
) -> None:
    """
    Valida que las probabilidades estén entre 0 y 1.
    """
    X, _ = X_y

    proba = loaded_model.predict_proba(X)

    assert np.all(proba >= 0)
    assert np.all(proba <= 1)


def test_model_predict_proba_rows_sum_to_one(
    loaded_model,
    X_y: tuple[pd.DataFrame, pd.Series],
) -> None:
    """
    Valida que cada fila de predict_proba sume 1.
    """
    X, _ = X_y

    proba = loaded_model.predict_proba(X)

    np.testing.assert_allclose(
        proba.sum(axis=1),
        1.0,
        atol=1e-6,
    )


def test_positive_class_probability_has_correct_length(
    loaded_model,
    X_y: tuple[pd.DataFrame, pd.Series],
) -> None:
    """
    Valida que la probabilidad de clase positiva tenga una probabilidad por fila.
    """
    X, _ = X_y

    proba_positiva = loaded_model.predict_proba(X)[:, 1]

    assert len(proba_positiva) == len(X)
    assert np.all(proba_positiva >= 0)
    assert np.all(proba_positiva <= 1)


# ============================================================
# Tests de consistencia post-serialización
# ============================================================

def test_predictions_are_equal_before_and_after_serialization(
    trained_model: DecisionTreeClassifier,
    serialized_model_path: Path,
    X_y: tuple[pd.DataFrame, pd.Series],
) -> None:
    """
    Valida que las predicciones sean iguales antes y después de serializar.
    """
    X, _ = X_y

    model_loaded = joblib.load(serialized_model_path)

    pred_original = trained_model.predict(X)
    pred_loaded = model_loaded.predict(X)

    np.testing.assert_array_equal(pred_original, pred_loaded)


def test_probabilities_are_equal_before_and_after_serialization(
    trained_model: DecisionTreeClassifier,
    serialized_model_path: Path,
    X_y: tuple[pd.DataFrame, pd.Series],
) -> None:
    """
    Valida que las probabilidades sean iguales antes y después de serializar.
    """
    X, _ = X_y

    model_loaded = joblib.load(serialized_model_path)

    proba_original = trained_model.predict_proba(X)
    proba_loaded = model_loaded.predict_proba(X)

    np.testing.assert_allclose(proba_original, proba_loaded, atol=1e-6)


# ============================================================
# Tests de firma del modelo y métricas
# ============================================================

def test_model_signature_file_exists(model_signature_path: Path) -> None:
    """
    Valida que model_signature.json exista.
    """
    assert model_signature_path.exists()


def test_model_signature_contains_expected_fields(
    model_signature_path: Path,
    X_y: tuple[pd.DataFrame, pd.Series],
) -> None:
    """
    Valida que la firma del modelo contenga campos esperados.
    """
    X, _ = X_y

    with open(model_signature_path, "r", encoding="utf-8") as file:
        signature = json.load(file)

    assert signature["target"] == TARGET
    assert signature["prediction_type"] == "binary_classification"
    assert signature["inputs"] == X.columns.tolist()
    assert signature["n_features"] == X.shape[1]
    assert "outputs" in signature


def test_metrics_file_exists(metrics_path: Path) -> None:
    """
    Valida que metrics.json exista.
    """
    assert metrics_path.exists()


def test_metrics_file_contains_main_metrics(metrics_path: Path) -> None:
    """
    Valida que metrics.json tenga métricas principales.
    """
    with open(metrics_path, "r", encoding="utf-8") as file:
        metrics = json.load(file)

    for metric_name in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        assert metric_name in metrics
        assert 0.0 <= metrics[metric_name] <= 1.0

    assert "confusion_matrix" in metrics
    assert len(metrics["confusion_matrix"]) == 2


# ============================================================
# Tests de errores esperados
# ============================================================

def test_loading_missing_model_raises_file_not_found(tmp_path: Path) -> None:
    """
    Valida que cargar un modelo inexistente lance FileNotFoundError.
    """
    missing_path = tmp_path / "modelo_no_existe.pkl"

    with pytest.raises(FileNotFoundError):
        joblib.load(missing_path)


def test_model_fails_with_missing_feature_column(
    loaded_model,
    X_y: tuple[pd.DataFrame, pd.Series],
) -> None:
    """
    Valida que el modelo falle si falta una columna usada en entrenamiento.
    """
    X, _ = X_y

    X_incompleto = X.drop(columns=[X.columns[0]])

    with pytest.raises(ValueError):
        loaded_model.predict(X_incompleto)


def test_model_fails_with_extra_unknown_column(
    loaded_model,
    X_y: tuple[pd.DataFrame, pd.Series],
) -> None:
    """
    Valida que el modelo falle si aparece una columna extra no vista en entrenamiento.
    """
    X, _ = X_y

    X_extra = X.copy()
    X_extra["feature_extra"] = 1

    with pytest.raises(ValueError):
        loaded_model.predict(X_extra)
