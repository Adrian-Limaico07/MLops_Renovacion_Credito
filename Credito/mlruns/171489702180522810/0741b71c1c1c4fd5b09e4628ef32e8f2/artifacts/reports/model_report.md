# Reporte técnico del modelo - Renovación de crédito

## 1. Objetivo

El objetivo del modelo es estimar la probabilidad de que un cliente acepte o concrete una renovación de préstamo.

Variable objetivo: `FLAG_VENTA`.

## 2. Configuración de evaluación

| Elemento | Valor |
|---|---:|
| Modelo | `RandomForestClassifier` |
| Filas de entrenamiento | 61289 |
| Filas de prueba | 26267 |
| Variables predictoras | 40 |
| Tamaño de test | 0.3 |
| Semilla aleatoria | 42 |
| Umbral de clasificación | 0.5 |

## 3. Métricas principales

| Métrica | Valor |
|---|---:|
| Accuracy | 0.8882 |
| Precision | 0.0713 |
| Recall | 0.1498 |
| F1-score | 0.0966 |
| ROC AUC | 0.6123 |

## 4. Umbrales mínimos definidos

| Métrica | Umbral mínimo |
|---|---:|
| Recall mínimo | 0.6500 |
| ROC AUC mínimo | 0.6500 |
| F1 mínimo | 0.6000 |

## 5. Matriz de confusión

La matriz de confusión tiene la siguiente forma:

[[TN, FP],
 [FN, TP]]

Resultado obtenido:

[[23174, 2045], [891, 157]]

## 6. Interpretación

- **Accuracy**: porcentaje total de aciertos del modelo.
- **Precision**: de los clientes predichos como venta, cuántos realmente fueron venta.
- **Recall**: de las ventas reales, cuántas logró detectar el modelo.
- **F1-score**: balance entre precision y recall.
- **ROC AUC**: capacidad general del modelo para separar clientes con y sin venta.

En este problema, el **recall** es importante porque ayuda a detectar más clientes con probabilidad de renovar.

## 7. Archivos generados

- `artifacts/metrics.json`
- `reports/classification_report.csv`
        - `reports/figures/matriz_confusion.png`
- `reports/figures/curva_roc.png`
- `reports/figures/distribucion_score.png`
- `reports/model_report.md`
