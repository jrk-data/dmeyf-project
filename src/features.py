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


def create_intra_month_features_bq(
    project_id: str,
    dataset_id: str,
    source_table: str,
    output_table: str
) -> None:
    """
    Crea la tabla con Feature Engineering intra-mes en BigQuery.

    Args:
        project_id: ID del proyecto de Google Cloud.
        dataset_id: ID del dataset.
        source_table: Nombre de la tabla de entrada (cruda o con targets).
        output_table: Nombre de la tabla de salida.
    """

    logger.info(f"Iniciando Feature Engineering intra-mes para '{output_table}'...")

    try:
        client = bigquery.Client(project=project_id)
        source_ref = f"`{project_id}.{dataset_id}.{source_table}`"
        output_ref = f"`{project_id}.{dataset_id}.{output_table}`"

        query = f"""
        CREATE OR REPLACE TABLE {output_ref}
        PARTITION BY RANGE_BUCKET(foto_mes, GENERATE_ARRAY(201901, 202208, 1))
        CLUSTER BY foto_mes, numero_de_cliente
        AS
        SELECT
            t1.* EXCEPT(clase_ternaria),
            t1.clase_ternaria, -- Aseguramos que la columna clase_ternaria esté al final

            -- kmes (Mes del año)
            MOD(t1.foto_mes, 100) AS kmes,

            -- ctrx_quarter_normalizado (Normalización de ctrx_quarter por antigüedad)
            CASE
                WHEN t1.cliente_antiguedad = 1 THEN t1.ctrx_quarter * 5.0
                WHEN t1.cliente_antiguedad = 2 THEN t1.ctrx_quarter * 2.0
                WHEN t1.cliente_antiguedad = 3 THEN t1.ctrx_quarter * 1.2
                ELSE t1.ctrx_quarter -- Valor por defecto o ctrx_quarter original
            END AS ctrx_quarter_normalizado,

            -- mpayroll_sobre_edad
            CASE
                WHEN t1.cliente_edad IS NULL OR t1.cliente_edad = 0 THEN NULL
                ELSE t1.mpayroll / t1.cliente_edad
            END AS mpayroll_sobre_edad

        FROM {source_ref} AS t1;
        """

        job = client.query(query)
        job.result()
        logger.info(f"✅ Feature Engineering intra-mes completado. Tabla guardada en '{output_table}'.")

    except Exception as e:
        logger.error(f"❌ Error al ejecutar el Feature Engineering intra-mes en BigQuery: {e}")
        raise



def create_historical_features_bq(
    project_id: str,
    dataset_id: str,
    source_table: str,
    output_table: str,
    cols_to_engineer: list, # Lista de columnas para las que calcular historial
    window_size: int = 6,
) -> None:

    logger.info(f"Iniciando Feature Engineering histórico ({window_size} meses) con CTEs para evitar anidación...")

    try:
        client = bigquery.Client(project=project_id)
        source_ref = f"`{project_id}.{dataset_id}.{source_table}`"
        output_ref = f"`{project_id}.{dataset_id}.{output_table}`"

        # --- CONSTRUCCIÓN DINÁMICA DE EXPRESIONES (Sin la cláusula 'OVER') ---
        lag_exprs = []
        hist_exprs = []

        # 1. Definición de la especificación de la ventana (para re-uso)
        window_spec_name = "w"
        window_spec_sql = f"""
            WINDOW {window_spec_name} AS (
                PARTITION BY numero_de_cliente
                ORDER BY foto_mes
                ROWS BETWEEN {window_size - 1} PRECEDING AND CURRENT ROW
            )
        """

        # 2. Loop para generar todas las expresiones (usando alias 't2')
        for col in cols_to_engineer:

            # --- Variables limpias del CTE BaseFeatures (t2) ---
            col_base = f"t2_{col}_clean"
            col_rn = "t2_row_index"

            # --- 1. Lags ---
            col_lag1 = f"LAG({col_base}, 1) OVER (PARTITION BY t2.numero_de_cliente ORDER BY t2.foto_mes)"
            col_lag2 = f"LAG({col_base}, 2) OVER (PARTITION BY t2.numero_de_cliente ORDER BY t2.foto_mes)"

            lag_exprs.append(f"{col_lag1} AS {col}_lag1")
            lag_exprs.append(f"{col_lag2} AS {col}_lag2")

            # --- 2. Deltas (Resta) ---
            lag_exprs.append(f"({col_base} - {col_lag1}) AS {col}_delta1")
            lag_exprs.append(f"({col_base} - {col_lag2}) AS {col}_delta2")

            # --- 3. Tendencia (COVAR_POP / VAR_POP) ---
            # Aplicamos la ventana NOMBRADA ({window_spec_name}) a cada función de agregación
            hist_exprs.append(f"""
                (
                    COVAR_POP({col_base}, {col_rn}) OVER {window_spec_name}
                    /
                    NULLIF(VAR_POP({col_rn}) OVER {window_spec_name}, 0)
                ) AS {col}_tend{window_size}
            """)

            # --- 4. Promedio, Min, Max (AVG, MIN, MAX) ---
            col_avg = f"AVG({col_base}) OVER {window_spec_name}"
            col_min = f"MIN({col_base}) OVER {window_spec_name}"
            col_max = f"MAX({col_base}) OVER {window_spec_name}"

            hist_exprs.append(f"{col_avg} AS {col}_avg{window_size}")
            # Corregir los nombres de alias de min/max (antes eran _avg{window_size} duplicados)
            hist_exprs.append(f"{col_min} AS {col}_min{window_size}")
            hist_exprs.append(f"{col_max} AS {col}_max{window_size}")

            # --- 5. Ratios (División) ---
            hist_exprs.append(f"({col_base} / NULLIF({col_avg}, 0)) AS {col}_ratioavg{window_size}")
            hist_exprs.append(f"({col_base} / NULLIF({col_max}, 0)) AS {col}_ratiomax{window_size}")

        all_new_features = lag_exprs + hist_exprs

        # --- QUERY FINAL CON CTEs ---
        # BaseFeatures: Limpieza de tipos y cálculo del índice (ROW_NUMBER)
        # HistoricalFeatures: Cálculos de ventana (LAG, AVG, COVAR)

        cols_for_base_cte = [f"CAST(t1.{col} AS FLOAT64) AS t2_{col}_clean" for col in cols_to_engineer]

        query = f"""
        CREATE OR REPLACE TABLE {output_ref}
        PARTITION BY RANGE_BUCKET(foto_mes, GENERATE_ARRAY(201801, 203001, 1))
        CLUSTER BY foto_mes, numero_de_cliente
        AS
        WITH BaseFeatures AS (
            SELECT
                -- 1. Seleccionamos TODAS las columnas originales (solo una vez)
                t1.*,
                -- 2. Calculamos el índice para la regresión (X)
                ROW_NUMBER() OVER(PARTITION BY t1.numero_de_cliente ORDER BY t1.foto_mes) AS t2_row_index,
                -- 3. Creamos las versiones limpias/casteadas de las features (Y)
                {', '.join(cols_for_base_cte)}
            FROM {source_ref} AS t1
        ),
        HistoricalFeatures AS (
            SELECT
                -- 1. Seleccionamos todas las columnas base y eliminamos las auxiliares
                t2.* EXCEPT({', '.join([f"t2_{col}_clean" for col in cols_to_engineer])}, t2_row_index),

                -- 2. Agregamos las features históricas calculadas
                {', '.join(all_new_features)}
            FROM BaseFeatures AS t2
            {window_spec_sql}
        )
        SELECT * FROM HistoricalFeatures;
        """
        # DEBUG: Imprimir la query completa para revisión antes de ejecutar
        # print(query)

        job = client.query(query)
        job.result()
        logger.info(f"✅ Feature Engineering histórico completado. Tabla guardada en '{output_table}'.")

    except Exception as e:
        logger.error(f"❌ Error al ejecutar el Feature Engineering histórico en BigQuery: {e}")
        raise


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






def creation_lags(table_source, columnas: List[str],  cant_lag: int = 1):
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
    PARTITION BY RANGE_BUCKET(foto_mes, GENERATE_ARRAY(201901, 202208, 1))
    CLUSTER BY numero_de_cliente
    AS
    SELECT
    a.*,
      {select_list}
    FROM `{config.BQ_PROJECT}.{config.BQ_DATASET}.{table_source}` AS a
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
            delta_exprs.append(f"SAFE_CAST({col} AS FLOAT64) - SAFE_CAST({col}_lag_{k} AS FLOAT64) AS {col}_delta_{k}")
            # opcional %:
            # delta_exprs.append(f"SAFE_DIVIDE(SAFE_CAST({col} AS FLOAT64) - SAFE_CAST({col}_lag{k} AS FLOAT64), NULLIF(SAFE_CAST({col}_lag{k} AS FLOAT64), 0)) AS {col}_d{k}_pct")

    delta_sql = ",\n  ".join(delta_exprs)

    sql = f"""
    CREATE OR REPLACE TABLE {table_delta}
    PARTITION BY RANGE_BUCKET(foto_mes, GENERATE_ARRAY(201901, 202208, 1))
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
        if '_lag' not in c and '_delta' not in c:
            filtradas.append(c)
        elif re.search(rf'_lag[1-{k}]$', c) or re.search(rf'_delta[1-{k}]$', c):
            filtradas.append(c)
    return filtradas

def _build_momentum_alter_update_script(
    table_id: str,
    feature_names: list[str],
    max_delta: int = 3,
    column_type: str = "FLOAT64",
) -> str:
    """
    Genera un script de BigQuery que:
      1) Agrega columnas *_momentum_ponderado a la tabla.
      2) Actualiza esas columnas con el cálculo de momentum ponderado
         en base a los deltas *_delta_1..*_delta_max_delta.

    Args:
        table_id: ID completo de la tabla en BigQuery, ej: "proyecto.dataset.c02_delta".
        feature_names: lista de features base, ej: ["edad", "altura"].
        max_delta: cantidad de deltas disponibles (ej. 3 → _delta_1.._delta_3).
        column_type: tipo de dato de la nueva columna (por defecto FLOAT64).

    Returns:
        str: script SQL listo para ejecutar en BigQuery.
    """

    alter_statements = []
    update_assignments = []

    for feat in feature_names:
        # nombre de la nueva columna
        alias = f"{feat}_momentum_ponderado"

        # construyo la expresión de momentum ponderado
        terms = []
        # pesos: max_delta para delta_1, ..., 1 para delta_max_delta
        for k in range(1, max_delta + 1):
            weight = max_delta - k + 1
            col_name = f"{feat}_delta_{k}"
            terms.append(f"{weight} * {col_name}")

        expr = " + ".join(terms)

        # ALTER TABLE para agregar la columna (si no existe)
        alter_statements.append(
            f"ALTER TABLE `{table_id}` "
            f"ADD COLUMN IF NOT EXISTS {alias} {column_type};"
        )

        # assignment para el UPDATE
        update_assignments.append(f"  {alias} = {expr}")

    # Bloque de ALTERs (uno por columna)
    alter_block = "\n".join(alter_statements)

    # Bloque de UPDATE (un único UPDATE seteando todas)
    update_block = ",\n".join(update_assignments)

    script = f"""
{alter_block}

UPDATE `{table_id}`
SET
{update_block} where 1 = 1;
"""
    return script.strip()

def create_momentums_deltas():
    query = _build_momentum_alter_update_script(table_id=f"{config.BQ_PROJECT}.{config.BQ_DATASET}.c02_delta", feature_names= config.pr.PSI_2021_FEATURES)

    try:
        client = bigquery.Client(project=config.BQ_PROJECT)
        # print("Ejecutando consulta en BigQuery...")
        query_job = client.query(query)
        results = query_job.result()
    except Exception as e:
        logger.error(f"Error en la consulta a BigQuery: {e}")
        pass


def select_data_lags_deltas(tabla, meses_a_cargar: list, k: int):
    # 'meses_a_cargar' debe ser una lista de enteros únicos

    logger.info(f"Meses cargados para features: {meses_a_cargar}")

    schema_table = _select_table_schema(config.BQ_PROJECT, config.BQ_DATASET, tabla)
    columns = _filter_lags_deltas(schema_table, k)
    client = bigquery.Client(project=config.BQ_PROJECT)
    bqstorage_client = bigquery_storage.BigQueryReadClient()

    # Aseguramos que la lista sea de strings para el UNNEST
    meses_str = ", ".join(str(int(m)) for m in meses_a_cargar)

    query = f"""SELECT {', '.join(columns)} FROM `{config.BQ_PROJECT}.{config.BQ_DATASET}.{tabla}`
    where foto_mes in UNNEST ([{meses_str}])"""

    job = client.query(query)

    # Uso Storage API para traer Arrow más rápido
    arrow_table = job.result().to_arrow(bqstorage_client=bqstorage_client)
    df_pl = pl.from_arrow(arrow_table)
    return df_pl