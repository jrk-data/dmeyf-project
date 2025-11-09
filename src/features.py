import duckdb
#from pydbus import connect
import re
from typing import Any
from pathlib import Path
from logging import getLogger
from typing import List, Tuple
import polars as pl
import src.config as config

from google.cloud import bigquery,bigquery_storage


logger = getLogger(__name__)







def get_numeric_columns_pl(
    df: pl.DataFrame,
    exclude_cols: Tuple[str, ...] = ()
) -> List[str]:
    """
    Devuelve las columnas numéricas del DataFrame `df`
    excluyendo las especificadas en `exclude_cols`.
    Equivalente a la versión DuckDB pero operando directamente sobre Polars.
    """
    # tipos numéricos soportados
    numeric_types = {
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64, pl.Decimal
    }
    # recorrer el schema del df
    numeric_cols = [
        name
        for name, dtype in df.schema.items()
        if dtype in numeric_types and name not in exclude_cols
    ]
    return numeric_cols






def feature_engineering_lag(df: pl.DataFrame, columnas: List[str], cant_lag: int = 1) -> pl.DataFrame:
    sql = "SELECT "
    for i, attr in enumerate(columnas):
        if attr in df.columns:
            sql += f"{attr}, lag({attr}, {i}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) AS {attr}_lag_{i}, "
        else:
            logger.warning(f"El atributo {attr} no existe en el DataFrame")

    # Completar la consulta
    sql += " * FROM df"
    logger.debug(f"Consulta SQL: {sql}")

    # Ejecutar la consulta
    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df = con.execute(sql).pl()
    con.close()
    logger.info(f"Se han calculado lags de {cant_lag} orden para {len(columnas)} columnas.")
    return df

def feature_engineering_delta(
    df: pl.DataFrame,
    columnas: List[str],
    cant_lag: int = 1,
) -> pl.DataFrame:
    """
    Agrega columnas de deltas para cada columna numérica indicada:
    delta_{col}_{k} = col - LAG(col, k)  (por cliente y ordenado por foto_mes)

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame con al menos: 'numero_de_cliente' y 'foto_mes'
    columnas : List[str]
        Columnas a las que calcular deltas (se ignoran las que no existan/no sean numéricas)
    cant_lag : int
        Cantidad de lags a usar (k = 1..cant_lag)

    Returns
    -------
    pl.DataFrame
        DataFrame original + columnas delta_*
    """
    if cant_lag < 1:
        return df

    # Filtrar columnas existentes y numéricas
    numeric_types = {pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                     pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                     pl.Float32, pl.Float64, pl.Decimal}
    cols_ok = []
    for c in columnas:
        if c not in df.columns:
            logger.warning(f"El atributo {c} no existe en el DataFrame")
            continue
        if type(df.schema.get(c)) not in numeric_types:
            logger.warning(f"El atributo {c} no es numérico, se omite")
            continue
        cols_ok.append(c)

    if not cols_ok:
        logger.warning("No hay columnas válidas para calcular deltas.")
        return df

    # Armar SELECT: tomamos todas las columnas (*) y agregamos las deltas
    select_parts = ["*"]
    for attr in cols_ok:
        for k in range(1, cant_lag + 1):
            select_parts.append(
                f"({attr} - lag({attr}, {k}) "
                f"OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes)) "
                f"AS delta_{attr}_{k}"
            )

    sql = "SELECT " + ", ".join(select_parts) + " FROM df"
    logger.debug(f"Consulta SQL (deltas): {sql}")

    # Ejecutar
    con = duckdb.connect(database=":memory:")
    try:
        # DuckDB acepta Polars directamente
        con.register("df", df)
        out = con.execute(sql).pl()
    finally:
        con.close()
        logger.info(f"Se han calculado {cant_lag} deltas para {len(cols_ok)} campos.")
    return out



def creation_lags(columnas: List[str], cant_lag: int = 1):
    """
    Crea/actualiza la tabla c02_lags con lags de 1..cant_lag para cada columna,
    particionada por foto_mes y clusterizada por numero_de_cliente.
    NO genera lag_0 ni duplica columnas.
    """
    client = bigquery.Client(project=config.BQ_PROJECT)
    meses = config.MES_TRAIN + config.MES_TEST + config.MES_PRED

    # LAGs: 1..cant_lag (nunca 0)
    lag_exprs = []
    for col in columnas:
        for k in range(1, cant_lag + 1):
            lag_exprs.append(
                f" LAG(a.{col}, {k}) OVER (PARTITION BY a.numero_de_cliente ORDER BY a.foto_mes) AS {col}_lag_{k}"
            )

    select_items = [
        ", ".join(lag_exprs) if lag_exprs else None,
    ]
    # limpiar Nones y unir sin coma final
    select_list = ",\n  ".join([s for s in select_items if s])

    sql = f"""
    CREATE OR REPLACE TABLE `{config.BQ_PROJECT}.{config.BQ_DATASET}.c02_lags`
    PARTITION BY RANGE_BUCKET(foto_mes, GENERATE_ARRAY(201901, 202108, 1))
    CLUSTER BY numero_de_cliente
    AS
    SELECT
    a.*,
      {select_list}
    FROM `{config.BQ_PROJECT}.{config.BQ_DATASET}.c02_products` AS a
    """

    job_cfg = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("meses", "INT64", list(meses))]
    )
    job = client.query(sql, job_config=job_cfg)
    job.result()

def creation_deltas(columnas, cant_lag):
    client = bigquery.Client(project=config.BQ_PROJECT)
    table_fqn = f"`{config.BQ_PROJECT}.{config.BQ_DATASET}.c02_lags`"
    table_delta = f"`{config.BQ_PROJECT}.{config.BQ_DATASET}.c02_delta`"
    # expresiones delta (solo numéricas)
    delta_exprs = []
    for col in columnas:
        for k in range(1, cant_lag+1):
            delta_exprs.append(f"SAFE_CAST({col} AS FLOAT64) - SAFE_CAST({col}_lag_{k} AS FLOAT64) AS {col}_{k}")
            # opcional %:
            # delta_exprs.append(f"SAFE_DIVIDE(SAFE_CAST({col} AS FLOAT64) - SAFE_CAST({col}_lag{k} AS FLOAT64), NULLIF(SAFE_CAST({col}_lag{k} AS FLOAT64), 0)) AS {col}_d{k}_pct")

    delta_sql = ",\n  ".join(delta_exprs)

    sql = f"""
    CREATE OR REPLACE TABLE {table_delta}
    PARTITION BY RANGE_BUCKET(foto_mes, GENERATE_ARRAY(201901, 202108, 1))
    CLUSTER BY numero_de_cliente
    AS
    SELECT
      src.*,
      {delta_sql}
    FROM {table_fqn} AS src
    """
    job = client.query(sql)
    job.result()


# ------ SELECT DE DATASET CON FEATURES


def _select_table_schema(project, dataset, table):
    client = bigquery.Client(project=project)
    t = client.get_table(f"{project}.{dataset}.{table}")
    cols = [f.name for f in t.schema]
    return cols

def _filter_lags_deltas(cols, k):
    filtradas = []
    for c in cols:
        # Si es lag o delta hasta 5
        if '_lag_' not in c and '_delta_' not in c:
            filtradas.append(c)
        elif re.search(rf'_lag_[1-{k}]$', c) or re.search(rf'_delta_[1-{k}]$', c):
            filtradas.append(c)
    return filtradas

def select_data_lags_deltas(tabla, mes_train, mes_test_lista,mes_pred, k):
    'Selecciona los campos de lags y deltas para un k y todos los campos que no son lags o deltas'
    mes_test =  mes_test_lista[0]
    logger.info(f"mes_test: {mes_test}")
    logger.info(f"mes_pred: {mes_pred}")
    meses = [mes_train, mes_test, mes_pred]
    logger.info(f"meses: {meses}")

    schema_table = _select_table_schema(config.BQ_PROJECT, config.BQ_DATASET, tabla)

    columns = _filter_lags_deltas(schema_table, k)

    client = bigquery.Client(project=config.BQ_PROJECT)
    bqstorage_client = bigquery_storage.BigQueryReadClient()

    meses =  ", ".join(str(int(m)) for m in meses)

    query = f"""SELECT {', '.join(columns)} FROM `{config.BQ_PROJECT}.{config.BQ_DATASET}.{tabla}`
    where foto_mes in UNNEST ([{meses}])"""

    job = client.query(query)

    # Uso Storage API para traer Arrow más rápido
    arrow_table = job.result().to_arrow(bqstorage_client=bqstorage_client)
    df_pl = pl.from_arrow(arrow_table)
    return df_pl