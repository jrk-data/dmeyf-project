import duckdb
#from pydbus import connect

from src.config import DB_PATH
from typing import Any
from pathlib import Path
from logging import getLogger
from typing import List, Tuple
import polars as pl

from google.cloud import bigquery


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


