# src/predict.py
import polars as pl
import pandas as pd
from src.features import select_data_lags_deltas
import logging

logger = logging.getLogger(__name__)

TARGET_COLS = [
    "clase_ternaria",
    "clase_peso",
    "clase_binaria",
    "clase_binaria1",
    "clase_binaria2",
]


def prepare_prediction_dataframe(
        table_name: str,
        mes_pred: int,
        k: int
) -> pd.DataFrame:
    """
    Construye el dataframe FINAL de predicción para un mes dado.

    - Descarga datos del mes_pred con lags/deltas.
    - NO hace splits (train/test/pred) porque no aplica en predicción.
    - Elimina columnas objetivo del dataset.
    - Devuelve un DataFrame pandas listo para el ensamble.
    """

    logger.info(f"[PREDICT] Preparando dataset de predicción para mes {mes_pred}...")

    # --- 1) Obtener datos usando tu función original (ya calcula lags/deltas) ---
    df_polars = select_data_lags_deltas(
        table_name,
        mes_pred,  # mes_train dummy
        [mes_pred],  # mes_test  dummy
        [mes_pred],  # mes_pred real
        k=k
    )

    # --- 2) Dropear columnas objetivo ---
    df_polars = df_polars.drop(TARGET_COLS, strict=False)

    # --- 3) Convertir a pandas ----
    df_pandas = df_polars.to_pandas()

    logger.info(f"[PREDICT] Dataset de predicción preparado. Shape final: {df_pandas.shape}")

    return df_pandas
