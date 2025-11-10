from typing import Iterable, Union
import numpy as np
import polars as pl
from google.cloud import bigquery

from logging import getLogger



logger = getLogger(__name__)

def binary_target(df: pl.DataFrame) -> pl.DataFrame:
    """Binariza una columna de un DataFrame."""
    data = df
    try:
        logger.info("Asignando Pesos a los targets")
        data = df.with_columns([
            pl.when(pl.col("clase_ternaria") == "BAJA+2").then(1.00002)
            .when(pl.col("clase_ternaria") == "BAJA+1").then(1.00001)
            .otherwise(1.0)
            .alias("clase_peso")
        ])

        logger.info("Creando targets binarios 1 y 2")
        data = data.with_columns([
            pl.when(pl.col("clase_ternaria") == "BAJA+2").then(1).otherwise(0).alias("clase_binaria1"),
            pl.when(pl.col("clase_ternaria") == "CONTINUA").then(0).otherwise(1).alias("clase_binaria2"),
        ])
    except Exception as e:
        logger.error(f"Error en binarizar target: {e}")
    finally:
        return data

def create_binary_target_column(project, dataset, table):
    """Crea una columna clase_binaria en BigQuery y la rellena según clase_ternaria."""
    client = bigquery.Client(project=project)
    table_id = f"{project}.{dataset}.{table}"

    # 1️⃣ Crear la columna si no existe
    alter_query = f"""
    ALTER TABLE `{table_id}`
    ADD COLUMN IF NOT EXISTS clase_binaria INT64,
    ADD COLUMN IF NOT EXISTS clase_peso FLOAT64
;
    """
    client.query(alter_query).result()

    # 2️⃣ Poblarla según el valor de clase_ternaria
    update_query = f"""
    UPDATE `{table_id}`
    SET clase_binaria = CASE
        WHEN clase_ternaria IN ('BAJA+1', 'BAJA+2') THEN 1
        WHEN clase_ternaria = 'CONTINUA' THEN 0
        ELSE NULL
    END,
    clase_peso = CASE
        WHEN clase_ternaria = 'BAJA+2' THEN 1.00002
        WHEN clase_ternaria = 'BAJA+1' THEN 1.00001
        WHEN clase_ternaria = 'CONTINUA' THEN 1.0
        ELSE NULL
      END
    WHERE clase_ternaria IS NOT NULL;
    """

    job = client.query(update_query)
    job.result()
    logger.info(f"✅ Columna 'clase_binaria' creada y actualizada correctamente en {table_id}.")


# Función auxiliar para train test split
def _to_int_list(x):
    '''
    Esta función convierte una lista de valores a enteros.
    La uso para evitar que se rompa el split_trian_data sin importar que tipo de daot se pase en MES_TRAIN, MES_TEST, MES_PRED.
    '''
    if isinstance(x, (list, tuple, set)):
        return [int(v) for v in x]
    return [int(x)]  # si es un único valor


def _undersampling(df:pl.DataFrame ,undersampling_rate:float , semilla:int) -> pl.DataFrame:
    logger.info("Comienzo del subsampleo")
    np.random.seed(semilla)
    clientes_minoritaria = df.filter(pl.col("clase_ternaria") != "CONTINUA").get_column("numero_de_cliente").unique()
    clientes_mayoritaria = df.filter(pl.col("clase_ternaria") == "CONTINUA").get_column("numero_de_cliente").unique()

    logger.info(f"Clientes minoritarios: {len(clientes_minoritaria)}")
    logger.info(f"Clientes mayoritarios: {len(clientes_mayoritaria)}")

    n_sample = int(len(clientes_mayoritaria) * undersampling_rate)
    clientes_mayoritaria_sample = np.random.choice(clientes_mayoritaria, n_sample, replace=False)

    # Unimos los IDs seleccionados
    clientes_finales = np.concatenate([clientes_minoritaria, clientes_mayoritaria_sample])

    logger.info(f"Clientes finales para hacer undersampling: {len(clientes_finales)}")

    df_train_undersampled = df.filter(pl.col("numero_de_cliente").is_in(clientes_finales))

    logger.info(f"Shape original: {df.shape}")
    logger.info(f"Shape undersampled: {df_train_undersampled.shape}")

    df_train_undersampled = df_train_undersampled.sample(fraction=1.0, seed=semilla)

    return df_train_undersampled



def split_train_data(
    data: pl.DataFrame,
    MES_TRAIN: Union[int, Iterable[int]],
    MES_TEST: Union[int, Iterable[int]],
    MES_PRED: Union[int, Iterable[int]],
    SEMILLA: int,
    SUB_SAMPLE: float = None

) -> dict:
    logger.info("Dividiendo datos en train / test / pred...")

    # Normalizo a listas de int
    train_list = _to_int_list(MES_TRAIN)
    test_list  = _to_int_list(MES_TEST)
    pred_list  = _to_int_list(MES_PRED)

    # Aseguro tipo de foto_mes a INT (si viene como str)
    if data.schema.get("foto_mes") != pl.Int64:
        data = data.with_columns(pl.col("foto_mes").cast(pl.Int64))

    # Filtros (no hace falta crear Series)
    train_data = data.filter(pl.col("foto_mes").is_in(train_list)) if train_list else pl.DataFrame(schema=data.schema)
    test_data  = data.filter(pl.col("foto_mes").is_in(test_list))  if test_list  else pl.DataFrame(schema=data.schema)
    pred_data  = data.filter(pl.col("foto_mes").is_in(pred_list))  if pred_list  else pl.DataFrame(schema=data.schema)

    # Hago subsampleo
    if SUB_SAMPLE is not None:
        train_data=_undersampling(train_data , SUB_SAMPLE,SEMILLA)

    columns_drop = ["clase_ternaria", "clase_peso","clase_binaria", "clase_binaria1", "clase_binaria2"]
    logger.info(f"Dropeando (si existen): {columns_drop}")

    try:
        # drop tolerante a columnas faltantes
        X_train_pl = train_data.drop(columns_drop, strict=False)
        X_test_pl  = test_data.drop(columns_drop, strict=False)
        X_pred_pl  = pred_data.drop(columns_drop, strict=False)

        # Targets / pesos (si no existen, crea arrays vacíos del largo)
        n_tr = train_data.height
        n_te = test_data.height

        def _col_or_empty(df: pl.DataFrame, name: str, n: int):
            return (df.get_column(name).to_numpy().ravel()
                    if name in df.columns else
                    pl.Series([None]*n).to_numpy())

        y_train_binaria = _col_or_empty(train_data, "clase_binaria", n_tr)
        w_train          = (_col_or_empty(train_data, "clase_peso", n_tr)).astype(float) if "clase_peso" in train_data.columns else pl.Series([1.0]*n_tr).to_numpy()

        y_test_binaria  = _col_or_empty(test_data, "clase_binaria", n_te)
        y_test_class     = (_col_or_empty(test_data, "clase_ternaria", n_te))

    except Exception as e:
        logger.error(f"Error en split train/test: {e}")
        raise

    response = {
        'X_train_pl': X_train_pl,
        'X_test_pl':  X_test_pl,
        'X_pred_pl':  X_pred_pl,
        'y_train_binaria': y_train_binaria,
        'y_test_binaria':  y_test_binaria,
        'w_train':          w_train,
        'y_test_class':     y_test_class,
    }
    logger.info(f"Shapes -> X_train: {X_train_pl.shape}, X_test: {X_test_pl.shape}, X_pred: {X_pred_pl.shape}")
    return response
