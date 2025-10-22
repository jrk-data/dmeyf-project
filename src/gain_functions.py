import numpy as np
from src.config import GANANCIA_ACIERTO, COSTO_ESTIMULO
import pandas as pd
from logging import getLogger

logger = getLogger(__name__)


def calcular_ganancia(y_true, y_pred):
    """
    Args:
        y_true: Valores reales (0 o 1)
        y_pred: Predicciones (0 o 1)

    Returns:
        float: Ganancia total
    """

    # Convertir a numpy arrays si es necesario
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values

    # Calcular ganancia vectorizada usando configuración
    # Verdaderos positivos: y_true==1 & y_pred==1 => ganancia
    # Falsos positivos: y_true==0 & y_pred==1 => costo
    # Verdaderos negativos y falsos negativos: ganancia = 0
    ganancia_total = np.sum(
        (y_true == 1) & (y_pred == 1) * GANANCIA_ACIERTO +   # TP
        (y_true == 0) & (y_pred == 1) * (-COSTO_ESTIMULO)    # FP
    )

    logger.debug(f"Ganancia calculada: {ganancia_total:.0f}")
    logger.debug(f"(GANANCIA_ACIERTO={GANANCIA_ACIERTO}, COSTO_ESTIMULO={COSTO_ESTIMULO})")

    return ganancia_total


def ganancia_lgb_binary(y_pred, y_true):
    """
    Función de ganancia para LightGBM en clasificación binaria.
    Compatible con callbacks de LightGBM.

    Args:
        y_pred: Predicciones de probabilidad del modelo
        y_true: Dataset de LightGBM con labels verdaderos

    Returns:
        tuple: (eval_name, eval_result, is_higher_better)
    """

    # Obtener labels verdaderos
    y_true_labels = y_true.get_label()

    # Convertir probabilidades a predicciones binarias (umbral 0.025)
    y_pred_binary = (y_pred > 0.025).astype(int)

    # Calcular ganancia usando configuración
    ganancia_total = calcular_ganancia(y_true_labels, y_pred_binary)

    # Retornar en formato esperado por LightGBM
    return 'ganancia', ganancia_total, True  # True = higher is better

def lgb_gan_eval(y_pred, data):
    """
    Esta función se utiliza como eval_metric en LightGBM.
    Devuelve la ganancia máxima que puede tener un modelo.

    Args:
        y_pred:
        data:

    Returns:

    """
    weight = data.get_weight()
    ganancia = np.where(weight == 1.00002, ganancia_acierto, 0) - np.where(weight < 1.00002, costo_estimulo, 0)
    ganancia = ganancia[np.argsort(y_pred)[::-1]]
    ganancia = np.cumsum(ganancia)

    return 'gan_eval', np.max(ganancia) , True