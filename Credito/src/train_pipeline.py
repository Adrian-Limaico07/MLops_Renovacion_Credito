"""
train_pipeline.py — Entrenamiento del modelo de renovación de crédito.

Este script:
1. Construye el dataset final usando build_features.py
2. Separa X e y
3. Divide en train/test con estratificación
4. Aplica balanceo con SMOTE dentro de un Pipeline
5. Entrena modelos con GridSearchCV
6. Selecciona el mejor modelo según ROC AUC en validación cruzada
7. Evalúa el mejor modelo en test
8. Guarda modelo.pkl, metrics.json, feature_importance.csv y model_signature.json
"""

import json
import logging
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from src.config import (
        ARTIFACTS_DIR,
        DATASET_PATH,
        F1_MIN,
        RANDOM_STATE,
        RECALL_MIN,
        ROC_AUC_MIN,
        TARGET,
        TEST_SIZE,
    )
    from src.build_features import construir_features
except ImportError:
    from config import (
        ARTIFACTS_DIR,
        DATASET_PATH,
        F1_MIN,
        RANDOM_STATE,
        RECALL_MIN,
        ROC_AUC_MIN,
        TARGET,
        TEST_SIZE,
    )
    from build_features import construir_features


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

MODEL_PATH = ARTIFACTS_DIR / "modelo.pkl"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
FEATURE_IMPORTANCE_PATH = ARTIFACTS_DIR / "feature_importance.csv"
MODEL_SIGNATURE_PATH = ARTIFACTS_DIR / "model_signature.json"
TRAIN_METADATA_PATH = ARTIFACTS_DIR / "training_metadata.json"
CV_RESULTS_PATH = ARTIFACTS_DIR / "cv_results.csv"
TRAIN_TEST_SPLIT_PATH = ARTIFACTS_DIR / "train_test_split.pkl"


# ============================================================
# Parámetros de entrenamiento
# ============================================================

N_SPLITS_CV = 3

# Evita saturar memoria en Codespaces
N_JOBS_GRID = 1
N_JOBS_MODEL = 1

SCORING = {
    "roc_auc": "roc_auc",
    "f1": "f1",
    "recall": "recall",
    "precision": "precision",
}

REFIT_METRIC = "roc_auc"

# Primero entrenamos sin XGBoost para estabilizar el pipeline
INCLUDE_XGBOOST = False


# ============================================================
# Funciones auxiliares
# ============================================================

def crear_directorios() -> None:
    """
    Crea la carpeta artifacts/ si no existe.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Carpeta artifacts verificada: %s", ARTIFACTS_DIR)


def convertir_a_json_serializable(obj: Any) -> Any:
    """
    Convierte objetos numpy/pandas a tipos compatibles con JSON.

    Args:
        obj: Objeto a convertir.

    Returns:
        Objeto compatible con JSON.
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


def preparar_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separa variables predictoras X y variable objetivo y.

    Args:
        df: DataFrame final procesado.

    Returns:
        X, y.
    """
    if TARGET not in df.columns:
        raise ValueError(f"No existe la variable objetivo: {TARGET}")

    df = df.copy()

    y = pd.to_numeric(df[TARGET], errors="coerce")

    if y.isna().sum() > 0:
        raise ValueError(
            f"La variable objetivo {TARGET} tiene valores no numéricos o nulos."
        )

    y = y.astype(int)

    clases = sorted(y.unique().tolist())

    if not set(clases).issubset({0, 1}):
        raise ValueError(
            f"La variable objetivo debe ser binaria 0/1. Clases detectadas: {clases}"
        )

    X = df.drop(columns=[TARGET])

    columnas_no_numericas = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()

    if columnas_no_numericas:
        raise ValueError(
            "Existen columnas no numéricas en X. "
            f"Columnas detectadas: {columnas_no_numericas}"
        )

    if X.isna().sum().sum() > 0:
        columnas_con_nulos = X.columns[X.isna().any()].tolist()
        raise ValueError(
            "Existen valores nulos en X después de build_features.py. "
            f"Columnas: {columnas_con_nulos}"
        )

    logger.info("X preparado: %d filas x %d columnas", *X.shape)
    logger.info("Distribución de y: %s", y.value_counts().to_dict())

    return X, y


def dividir_train_test(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide los datos en entrenamiento y prueba usando estratificación.

    Args:
        X: Variables predictoras.
        y: Variable objetivo.

    Returns:
        X_train, X_test, y_train, y_test.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    logger.info("Train: %d filas | Test: %d filas", len(X_train), len(X_test))
    logger.info("Distribución y_train: %s", y_train.value_counts().to_dict())
    logger.info("Distribución y_test: %s", y_test.value_counts().to_dict())

    return X_train, X_test, y_train, y_test


def obtener_modelos_y_grids() -> dict[str, dict[str, Any]]:
    """
    Define los modelos candidatos y sus grillas de hiperparámetros.

    Returns:
        Diccionario de modelos y param_grids.
    """
    modelos = {
        "decision_tree": {
            "estimator": DecisionTreeClassifier(
                random_state=RANDOM_STATE,
            ),
            "param_grid": {
                "model__max_depth": [4, 6, 8, None],
                "model__min_samples_split": [2, 10],
                "model__criterion": ["gini", "entropy"],
            },
        },
        "random_forest": {
            "estimator": RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=N_JOBS_MODEL,
                class_weight="balanced",
            ),
            "param_grid": {
                "model__n_estimators": [100, 200],
                "model__max_depth": [6, 10],
                "model__min_samples_split": [5],
            },
        },
    }

    if INCLUDE_XGBOOST and XGBOOST_AVAILABLE:
        modelos["xgboost"] = {
            "estimator": XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=N_JOBS_MODEL,
                tree_method="hist",
            ),
            "param_grid": {
                "model__n_estimators": [200, 400],
                "model__max_depth": [3, 5],
                "model__learning_rate": [0.05, 0.10],
                "model__subsample": [0.8],
                "model__colsample_bytree": [0.8],
            },
        }
    elif INCLUDE_XGBOOST and not XGBOOST_AVAILABLE:
        logger.warning(
            "XGBoost no está instalado. Se entrenarán solo DecisionTree y RandomForest."
        )

    return modelos


def crear_pipeline(modelo: Any) -> ImbPipeline:
    """
    Crea pipeline con SMOTE y modelo.

    SMOTE se aplica solo durante entrenamiento/CV,
    no durante la predicción sobre test.

    Args:
        modelo: Estimador sklearn/xgboost.

    Returns:
        Pipeline de imbalanced-learn.
    """
    pipeline = ImbPipeline(
        steps=[
            (
                "smote",
                SMOTE(
                    random_state=RANDOM_STATE,
                    k_neighbors=5,
                    sampling_strategy=0.30,
                ),
            ),
            ("model", modelo),
        ]
    )

    return pipeline


def entrenar_gridsearch(
    model_name: str,
    estimator: Any,
    param_grid: dict[str, list[Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> GridSearchCV:
    """
    Entrena un modelo usando GridSearchCV.
    """
    logger.info("=" * 70)
    logger.info("Entrenando modelo: %s", model_name)
    logger.info("=" * 70)

    pipeline = crear_pipeline(estimator)

    cv = StratifiedKFold(
        n_splits=N_SPLITS_CV,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=SCORING,
        refit=REFIT_METRIC,
        cv=cv,
        n_jobs=N_JOBS_GRID,
        verbose=1,
        return_train_score=True,
    )

    grid_search.fit(X_train, y_train)

    logger.info("Mejor score CV %s: %.4f", REFIT_METRIC, grid_search.best_score_)
    logger.info("Mejores parámetros %s: %s", model_name, grid_search.best_params_)

    return grid_search


def obtener_probabilidades(modelo: Any, X: pd.DataFrame) -> np.ndarray:
    """
    Obtiene probabilidades de clase positiva.

    Args:
        modelo: Modelo entrenado.
        X: Datos de entrada.

    Returns:
        Vector de probabilidades de clase 1.
    """
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(X)[:, 1]

    if hasattr(modelo, "decision_function"):
        scores = modelo.decision_function(X)
        return 1 / (1 + np.exp(-scores))

    logger.warning(
        "El modelo no tiene predict_proba ni decision_function. "
        "Se usarán predicciones binarias como score."
    )

    return modelo.predict(X)


def calcular_metricas(
    modelo: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """
    Calcula métricas sobre el conjunto de prueba.

    Args:
        modelo: Modelo entrenado.
        X_test: Variables de prueba.
        y_test: Target real.

    Returns:
        Diccionario con métricas.
    """
    y_pred = modelo.predict(X_test)
    y_score = obtener_probabilidades(modelo, X_test)

    metricas = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_score), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
    }

    return metricas


def extraer_importancia_variables(
    modelo: Any,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Extrae importancia de variables si el modelo lo permite.

    Args:
        modelo: Pipeline entrenado.
        feature_names: Nombres de variables.

    Returns:
        DataFrame con importancia de variables.
    """
    if hasattr(modelo, "named_steps"):
        estimador_final = modelo.named_steps.get("model")
    else:
        estimador_final = modelo

    if not hasattr(estimador_final, "feature_importances_"):
        logger.warning(
            "El modelo seleccionado no tiene atributo feature_importances_."
        )
        return pd.DataFrame(columns=["feature", "importance"])

    importancias = estimador_final.feature_importances_

    feature_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importancias,
        }
    ).sort_values(by="importance", ascending=False)

    return feature_importance


def guardar_modelo(modelo: Any) -> None:
    """
    Guarda el modelo entrenado en artifacts/modelo.pkl.

    Args:
        modelo: Modelo entrenado.
    """
    joblib.dump(modelo, MODEL_PATH)
    logger.info("Modelo guardado en: %s", MODEL_PATH)


def guardar_metricas(metricas: dict[str, Any]) -> None:
    """
    Guarda métricas en artifacts/metrics.json.

    Args:
        metricas: Diccionario de métricas.
    """
    metricas = convertir_a_json_serializable(metricas)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metricas, file, indent=4, ensure_ascii=False)

    logger.info("Métricas guardadas en: %s", METRICS_PATH)


def guardar_importancia_variables(feature_importance: pd.DataFrame) -> None:
    """
    Guarda importancia de variables.

    Args:
        feature_importance: DataFrame de importancias.
    """
    feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    logger.info("Importancia de variables guardada en: %s", FEATURE_IMPORTANCE_PATH)


def guardar_model_signature(
    feature_names: list[str],
    model_name: str,
) -> None:
    """
    Guarda la firma de entrada/salida del modelo.

    Args:
        feature_names: Variables usadas por el modelo.
        model_name: Nombre del modelo seleccionado.
    """
    signature = {
        "model_name": model_name,
        "prediction_type": "binary_classification",
        "target": TARGET,
        "inputs": feature_names,
        "outputs": {
            "prediction": "Clase predicha 0/1",
            "probability": "Probabilidad estimada de FLAG_VENTA = 1",
        },
        "n_features": len(feature_names),
    }

    with open(MODEL_SIGNATURE_PATH, "w", encoding="utf-8") as file:
        json.dump(signature, file, indent=4, ensure_ascii=False)

    logger.info("Firma del modelo guardada en: %s", MODEL_SIGNATURE_PATH)


def guardar_train_test_split(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """
    Guarda la partición train/test para evaluación posterior.

    Args:
        X_train: Variables de entrenamiento.
        X_test: Variables de prueba.
        y_train: Target de entrenamiento.
        y_test: Target de prueba.
    """
    split_data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "target": TARGET,
        "feature_names": X_train.columns.tolist(),
    }

    joblib.dump(split_data, TRAIN_TEST_SPLIT_PATH)
    logger.info("Train/test split guardado en: %s", TRAIN_TEST_SPLIT_PATH)


def guardar_cv_results(cv_results: list[pd.DataFrame]) -> None:
    """
    Guarda los resultados de validación cruzada de todos los modelos.

    Args:
        cv_results: Lista de DataFrames con cv_results_.
    """
    if not cv_results:
        logger.warning("No hay resultados de CV para guardar.")
        return

    df_cv = pd.concat(cv_results, ignore_index=True)
    df_cv.to_csv(CV_RESULTS_PATH, index=False)

    logger.info("Resultados de CV guardados en: %s", CV_RESULTS_PATH)


def guardar_training_metadata(metadata: dict[str, Any]) -> None:
    """
    Guarda metadata del entrenamiento.

    Args:
        metadata: Diccionario de metadata.
    """
    metadata = convertir_a_json_serializable(metadata)

    with open(TRAIN_METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)

    logger.info("Metadata de entrenamiento guardada en: %s", TRAIN_METADATA_PATH)


# ============================================================
# Función principal de entrenamiento
# ============================================================

def train(df_features: pd.DataFrame | None = None) -> dict[str, Any]:
    """
    Entrena el pipeline de modelos.

    Args:
        df_features: DataFrame final ya procesado. Si es None,
            se ejecuta construir_features().

    Returns:
        Diccionario con métricas del mejor modelo.
    """
    crear_directorios()

    if df_features is None:
        logger.info("Construyendo features desde build_features.py")
        df_features = construir_features(DATASET_PATH)

    logger.info(
        "Dataset de features recibido: %d filas x %d columnas",
        *df_features.shape,
    )

    X, y = preparar_xy(df_features)

    X_train, X_test, y_train, y_test = dividir_train_test(X, y)

    modelos = obtener_modelos_y_grids()

    if not modelos:
        raise ValueError("No hay modelos disponibles para entrenar.")

    mejor_modelo = None
    mejor_nombre_modelo = None
    mejor_grid = None
    mejor_score_cv = -np.inf

    cv_results = []
    resultados_modelos = {}

    for model_name, spec in modelos.items():
        grid = entrenar_gridsearch(
            model_name=model_name,
            estimator=spec["estimator"],
            param_grid=spec["param_grid"],
            X_train=X_train,
            y_train=y_train,
        )

        df_cv_model = pd.DataFrame(grid.cv_results_)
        df_cv_model["model_name"] = model_name
        cv_results.append(df_cv_model)

        resultados_modelos[model_name] = {
            "best_cv_score": float(grid.best_score_),
            "best_params": grid.best_params_,
        }

        if grid.best_score_ > mejor_score_cv:
            mejor_score_cv = float(grid.best_score_)
            mejor_modelo = grid.best_estimator_
            mejor_nombre_modelo = model_name
            mejor_grid = grid

    if mejor_modelo is None or mejor_grid is None:
        raise ValueError("No se pudo seleccionar un mejor modelo.")

    logger.info("=" * 70)
    logger.info("Mejor modelo seleccionado: %s", mejor_nombre_modelo)
    logger.info("Mejor score CV %s: %.4f", REFIT_METRIC, mejor_score_cv)
    logger.info("Mejores parámetros: %s", mejor_grid.best_params_)
    logger.info("=" * 70)

    metricas_test = calcular_metricas(
        modelo=mejor_modelo,
        X_test=X_test,
        y_test=y_test,
    )

    metricas = {
        "best_model": mejor_nombre_modelo,
        "selection_metric": REFIT_METRIC,
        "cv_best_score": round(mejor_score_cv, 4),
        "params": mejor_grid.best_params_,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "recall_minimo": RECALL_MIN,
        "roc_auc_minimo": ROC_AUC_MIN,
        "f1_minimo": F1_MIN,
        **metricas_test,
    }

    feature_importance = extraer_importancia_variables(
        modelo=mejor_modelo,
        feature_names=X.columns.tolist(),
    )

    guardar_modelo(mejor_modelo)
    guardar_metricas(metricas)
    guardar_importancia_variables(feature_importance)
    guardar_model_signature(
        feature_names=X.columns.tolist(),
        model_name=mejor_nombre_modelo,
    )
    guardar_train_test_split(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
    guardar_cv_results(cv_results)

    training_metadata = {
        "dataset_path": str(DATASET_PATH),
        "target": TARGET,
        "rows": int(df_features.shape[0]),
        "columns": int(df_features.shape[1]),
        "n_features": int(X.shape[1]),
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "class_distribution_total": y.value_counts().to_dict(),
        "class_distribution_train": y_train.value_counts().to_dict(),
        "class_distribution_test": y_test.value_counts().to_dict(),
        "models_trained": list(modelos.keys()),
        "results_by_model": resultados_modelos,
        "best_model": mejor_nombre_modelo,
        "best_params": mejor_grid.best_params_,
    }

    guardar_training_metadata(training_metadata)

    logger.info("Entrenamiento finalizado correctamente.")
    logger.info("Métricas del mejor modelo: %s", metricas)

    return metricas


def entrenar_pipeline() -> dict[str, Any]:
    """
    Alias principal para ejecutar el entrenamiento completo.

    Returns:
        Métricas del mejor modelo.
    """
    return train()


# ============================================================
# Ejecución desde terminal
# ============================================================

if __name__ == "__main__":
    try:
        resultados = entrenar_pipeline()

        print("Entrenamiento finalizado correctamente.")
        print(json.dumps(convertir_a_json_serializable(resultados), indent=4))

    except Exception as error:
        logger.exception("Error durante el entrenamiento: %s", error)
        sys.exit(1)
