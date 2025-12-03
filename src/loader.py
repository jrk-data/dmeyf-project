import logging
from google.cloud import bigquery, bigquery_storage # storage para que lea mas rápido
import polars as pl
import pyarrow as pa
import src.config as config
import duckdb
# Para autenticarse en Google Cloud
from google.auth import default as google_auth_default
from google.auth.transport.requests import Request as AuthRequest
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)



# ######## UTILIDAD ###########
def _aut_google():
    # 1. --- OBTENER TOKEN DE AUTENTICACIÓN ---
    try:
        credentials, project = google_auth_default()
        credentials.refresh(AuthRequest())
        gcs_bearer_token = credentials.token
        logger.info("Token de Google Cloud obtenido exitosamente.")
    except Exception as e:
        logger.error(f"❌ No se pudo obtener el token de GCS/ADC: {e}")
        raise
    return gcs_bearer_token

# ################# LOGICA DUCKDB BIGQUERY ######################################
def load_gcs_to_bigquery_via_duckdb(
    project_id: str,
    dataset_id: str,
    table_id: str = "c03",
    gcs_file_path: str = config.DATA_PATH_C02,
    temp_local_db: str = ":memory:" # Usa ':memory:' para base de datos en memoria
) -> None:
    """
    Lee un archivo Gzip CSV desde GCS usando DuckDB,
    y luego carga el DataFrame resultante en una tabla de BigQuery.

    Args:
        project_id: ID del proyecto de Google Cloud.
        dataset_id: ID del dataset de destino en BQ.
        table_id: Nombre de la tabla de destino en BQ (por defecto, 'c03').
        gcs_file_path: Ruta completa del archivo GCS.
        temp_local_db: Ruta a la base de datos DuckDB. Usa ':memory:' para RAM.
    """
    # Se definen credenciales de google para que se pueda conectar desde la vm a BigQuery usando duckdb
    gcs_bearer_token = _aut_google()
    logger.info(f"Iniciando proceso: GCS ({gcs_file_path}) -> DuckDB -> BigQuery ({dataset_id}.{table_id})")

    # 1. Conectar DuckDB y leer GCS
    try:
        # Se requiere instalar el módulo httpfs para leer GCS/S3/HTTPs
        con = duckdb.connect(database=temp_local_db, read_only=False)
        con.sql("INSTALL httpfs;")
        con.sql("LOAD httpfs;")
        logger.info("DuckDB conectado y extensión 'httpfs' cargada.")


        #    --- SOLUCIÓN: CREAR SECRETO GCS CON EL TOKEN ---
        # DuckDB utilizará este secreto para todas las peticiones a GCS
        con.sql(f"CREATE SECRET gcs_secret (TYPE GCS, bearer_token '{gcs_bearer_token}');")
        logger.info("DuckDB conectado, extensión 'httpfs' cargada y secreto GCS creado.")


        # Ejecutar la consulta para leer el archivo GCS y obtener el resultado como un DataFrame de Pandas
        query = f"SELECT * FROM read_csv_auto('{gcs_file_path}');"
        df_duckdb = con.sql(query).df()

        logger.info(f"✅ Lectura de GCS a DataFrame completada. Filas cargadas: {len(df_duckdb)}")

        # Cerrar la conexión DuckDB
        con.close()

    except Exception as e:
        logger.error(f"❌ Error durante la lectura con DuckDB: {e}")
        # Asegúrate de que las credenciales de GCS sean válidas para la extensión httpfs
        raise

    # 2. Cargar DataFrame a BigQuery
    try:
        client = bigquery.Client(project=project_id)



        table_ref = client.dataset(dataset_id).table(table_id)

        # Configuración de carga: escribir sobre la tabla si ya existe
        job_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        )

        # Cargar el DataFrame de Pandas a BigQuery
        job = client.load_table_from_dataframe(
            df_duckdb, table_ref, job_config=job_config
        )

        job.result()

        logger.info(f"✅ DataFrame cargado exitosamente a BigQuery en la tabla '{table_id}' en el dataset '{dataset_id}'.")

    except Exception as e:
        logger.error(f"❌ Error durante la carga a BigQuery: {e}")
        raise


def consolidate_tables_bq(
        project_id: str,
        dataset_id: str,
        final_table_id: str = "c03_consolidated",
        source_table_ids: List[str] = ["c03_historico", "c03_agosto2021"],
        type_overrides: Optional[Dict[str, str]] = None,
) -> None:
    """
    Combina (UNION ALL) múltiples tablas de BigQuery, forzando la coherencia
    de tipos de datos usando CAST para evitar el error 400.
    """

    logger.info(f"Iniciando consolidación de tablas en {final_table_id}...")

    try:
        client = bigquery.Client(project=project_id)
        final_table_ref = f"`{project_id}.{dataset_id}.{final_table_id}`"

        union_clauses = []

        # 1. Obtener nombres de columnas de una tabla maestra (la primera en la lista)
        sample_table_id = source_table_ids[0]
        sample_table = client.get_table(f"{project_id}.{dataset_id}.{sample_table_id}")
        all_cols = [field.name for field in sample_table.schema]

        if type_overrides is None:
            # Si no hay overrides, volvemos a la lógica simple (SELECT *)
            logger.info("No se especificaron overrides; usando SELECT * para todas las tablas.")
            select_statement = "*"
        else:
            # 2. Construir la SELECT statement con CAST para todas las columnas
            select_parts = []
            for col in all_cols:
                if col in type_overrides:
                    target_type = type_overrides[col]
                    # Usamos SAFE_CAST para convertir a INT64/FLOAT64 y evitar errores si hay basura
                    select_parts.append(f"SAFE_CAST({col} AS {target_type}) AS {col}")
                else:
                    select_parts.append(col)

            select_statement = ",\n        ".join(select_parts)
            logger.warning(f"Usando CAST/SAFE_CAST en {len(type_overrides)} columnas problemáticas.")

        # 3. Construir la consulta UNION ALL para cada tabla fuente
        for table_id in source_table_ids:
            table_ref = f"`{project_id}.{dataset_id}.{table_id}`"

            # La SELECT statement (con o sin CAST) se aplica a cada tabla
            union_clauses.append(f"SELECT\n        {select_statement}\n        FROM {table_ref}")

        # 4. Consulta SQL final
        union_all_query = "\nUNION ALL\n".join(union_clauses)

        query = f"""
        CREATE OR REPLACE TABLE {final_table_ref}
        AS
        {union_all_query};
        """

        job = client.query(query)
        job.result()

        count_query = f"SELECT count(*) FROM {final_table_ref}"
        count_result = client.query(count_query).result().next()[0]

        logger.info(f"✅ Tablas consolidadas exitosamente en '{final_table_id}'. Total de filas: {count_result:,}")

    except Exception as e:
        logger.error(f"❌ Error durante la consolidación de tablas en BigQuery: {e}")
        raise





def create_churn_targets_bq(
    project_id: str,
    dataset_id: str,
    source_table: str,
    target_table: str = "targets",
) -> None:
    """
    Crea la tabla de targets (clase_ternaria) en BigQuery utilizando la lógica
    de gap temporal (BAJA+1, BAJA+2) del código R/data.table.

    Args:
        project_id: ID del proyecto de Google Cloud.
        dataset_id: ID del dataset.
        source_table: Nombre de la tabla de datos crudos o features.
        target_table: Nombre de la tabla donde se guardarán los targets.
    """

    logger.info(f"Iniciando creación de tabla de targets '{target_table}'...")

    try:
        client = bigquery.Client(project=project_id)
        source_ref = f"`{project_id}.{dataset_id}.{source_table}`"
        target_ref = f"`{project_id}.{dataset_id}.{target_table}`"

        # SQL para replicar la lógica de Target Engineering (R/data.table)
        query = f"""
        CREATE OR REPLACE TABLE {target_ref}
        PARTITION BY RANGE_BUCKET(foto_mes, GENERATE_ARRAY(201901, 202208, 1))
        CLUSTER BY foto_mes, numero_de_cliente
        AS
        WITH PreCalculations AS (
            SELECT
                foto_mes,
                numero_de_cliente,
                -- 1. Calcula el periodo serializado (ej. 202401 -> 24289)
                CAST(FLOOR(t1.foto_mes / 100) AS INT64) * 12 + MOD(t1.foto_mes, 100) AS periodo0,
                -- 2. Calcula los leads (periodo1 y periodo2)
                LEAD(CAST(FLOOR(t1.foto_mes / 100) AS INT64) * 12 + MOD(t1.foto_mes, 100), 1)
                    OVER (PARTITION BY t1.numero_de_cliente ORDER BY t1.foto_mes) AS periodo1,
                LEAD(CAST(FLOOR(t1.foto_mes / 100) AS INT64) * 12 + MOD(t1.foto_mes, 100), 2)
                    OVER (PARTITION BY t1.numero_de_cliente ORDER BY t1.foto_mes) AS periodo2
            FROM {source_ref} AS t1
        ),
        MaxPeriods AS (
            SELECT
                MAX(periodo0) AS periodo_ultimo,
                MAX(periodo0) - 1 AS periodo_anteultimo
            FROM PreCalculations
        )
        SELECT
            t1.foto_mes,
            t1.numero_de_cliente,
            t1.periodo0,
            t1.periodo1,
            t1.periodo2,
            -- 3. Aplica la lógica de la clase ternaria (siguiendo precedencia)
            CASE
                -- BAJA+2: Antes del penúltimo mes, hay continuidad en M+1, pero falta M+2 (gap > 2)
                WHEN t1.periodo0 < mp.periodo_anteultimo AND
                     (t1.periodo0 + 1 = t1.periodo1) AND
                     (t1.periodo2 IS NULL OR t1.periodo0 + 2 < t1.periodo2)
                THEN 'BAJA+2'

                -- BAJA+1: Antes del último mes, falta M+1 (periodo1 es NULL o hay un gap)
                WHEN t1.periodo0 < mp.periodo_ultimo AND
                     (t1.periodo1 IS NULL OR t1.periodo0 + 1 < t1.periodo1)
                THEN 'BAJA+1'

                -- CONTINUA: Por defecto para registros antes del penúltimo mes que no son Baja
                WHEN t1.periodo0 < mp.periodo_anteultimo
                THEN 'CONTINUA'

                ELSE NULL -- Registros del último/penúltimo mes sin clasificación de Baja
            END AS clase_ternaria
        FROM PreCalculations AS t1
        CROSS JOIN MaxPeriods AS mp;
        """

        job = client.query(query)
        job.result()
        logger.info(f"✅ Tabla de targets '{target_table}' creada exitosamente.")

    except Exception as e:
        logger.error(f"❌ Error al crear la tabla de targets en BigQuery: {e}")
        raise

# ################### Lógica para competencia 2 y 3##############################

def select_c02_polars():
    '''
    Esta función lee el csv de competencia 2 y lo convierte a polars.
    Hace una lectura rápida de los primeors 10 registros del csv para crear el schema y pasarlo luego a BigQuery.
    '''
    # tomo una muestra para obtener schema
    logger.info("Creando schema para competencia_02...")
    try:
        df = pl.read_csv(
        config.DATA_PATH_C02,
        n_rows=10,
        schema_overrides={"mprestamos_prendarios": pl.Float64},
        infer_schema_length=1000,
        )
        schema_actual = df.schema

        # schema_modificado = {
        #     col: (pl.Float64 if (col.startswith("m") or "_m" in col) and col != 'foto_mes' and dtype == pl.Int64 else dtype)
        #     for col, dtype in schema_actual.items()
        # }
        schema_modificado = {
            col: (pl.Float64 if (col.startswith("m") or "_m" in col) and col != 'foto_mes' and dtype == pl.Int64 else dtype)
            for col, dtype in schema_actual.items()
        }

    except Exception as e:
        logger.error(f'Error al crear schema: {e}')
        raise
    try:
        # Leo todo el csv con el schema que le paso a polars
        df = (
            pl.read_csv(
                config.DATA_PATH_C02,
                schema_overrides=schema_modificado,
                infer_schema_length=0  # desactiva inferencia de tipos
            )
        )

        return df

    except Exception as e:
        logger.error(f'Error al leer csv: {e}')
        raise

def create_bq_table_c02(df, PROJECT, DATASET, TABLE):
    '''
    Esta función crea una tabla en BigQuery a partir de un DataFrame de Polars.
       '''
    TABLE_ID = f"{PROJECT}.{DATASET}.{TABLE}"
    logger.info(f"Creando tabla {TABLE} en BigQuery...")
    # df_pl es tu DataFrame de Polars ya procesado
    # df_pl: pl.DataFrame = ...

    # 1) Convertir Polars → pandas usando Arrow (mejor preservación de tipos y nulos)
    df_pd = df.to_pandas(use_pyarrow_extension_array=True)

    # 2) Cliente BQ
    client = bigquery.Client(project=PROJECT)

    # 3) Crear dataset si no existe
    dataset_ref = bigquery.Dataset(f"{PROJECT}.{DATASET}")
    client.create_dataset(dataset_ref, exists_ok=True)

    # 4) Configurar el load job
    #    a) Partición por RANGO ENTERO en 'foto_mes' (YYYYMM). Ajustá rango según tus datos.
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,  # o WRITE_APPEND
        autodetect=True,  # BigQuery infiere esquema desde pandas/pyarrow; podés definir esquema explícito si querés
        range_partitioning=bigquery.RangePartitioning(
            field="foto_mes",
            range_=bigquery.PartitionRange(start=201901, end=202208, interval=1),
        ),
        clustering_fields=["foto_mes","numero_de_cliente"],  
    )

    # 5) Cargar DataFrame
    load_job = client.load_table_from_dataframe(df_pd, TABLE_ID, job_config=job_config)
    load_job.result()  # espera a que termine

    logger.info(f"Carga de tabla {TABLE} completada.")
    #tbl = client.get_table(TABLE_ID)

    #return tbl

def create_targets_3(PROJECT, DATASET, TABLE, TARGET_TABLE,targets):
    logger.info(f"Creando tabla de {TARGET_TABLE}...")
    try:
        client = bigquery.Client(project=PROJECT)
        query = f"""
        create or replace table `{PROJECT}.{DATASET}.{TARGET_TABLE}`
        PARTITION BY RANGE_BUCKET(foto_mes, GENERATE_ARRAY(201901, 202208, 1))
        CLUSTER BY foto_mes, numero_de_cliente AS
        with usuarios_ultimo_a_primer_es as(
          select
          foto_mes
          , numero_de_cliente
          , row_number() over (partition by numero_de_cliente order by foto_mes desc) as row_number
          from `{PROJECT}.{DATASET}.{TABLE}`
        ) select
        foto_mes
        ,numero_de_cliente
        , case
        when row_number = 1 and foto_mes < 202108 then 'BAJA+1'
        when row_number = 2 and foto_mes < 202107 then 'BAJA+2'
        when row_number >= 3 then 'CONTINUA'
        else null
        end as clase_ternaria
        from usuarios_ultimo_a_primer_es; 
        """
        client.query(query)
        logger.info("Se ha creado la tabla de clases ternarias")
    except Exception as e:
        logger.error(e)




def tabla_productos_por_cliente(PROJECT, DATASET, TABLE, TARGET_TABLE):
    try:
        client = bigquery.Client(project=PROJECT)
        table_ref = client.dataset(DATASET).table(TABLE)
        table = client.get_table(table_ref)

        # Detectar columnas con "master" (case-insensitive)
        logger.info("Detectando columnas con Master...")
        cols_master = [f.name for f in table.schema if "master" in f.name.lower()]
        logger.info("Detectando columnas con Visa...")
        cols_visa = [f.name for f in table.schema if "visa" in f.name.lower()]

        logger.info("Detectando resto de columnas...")

        cols_general = [
            f.name for f in table.schema
            if "visa" not in f.name.lower()
               and "master" not in f.name.lower()
               and f.name not in ("numero_de_cliente", "foto_mes", "clase_ternaria")
        ]


        # Construir la expresión SQL
        expr_sum_master = " + ".join(
            [f"IF(SAFE_CAST(a.{col} AS FLOAT64) IS NOT NULL AND SAFE_CAST(a.{col} AS FLOAT64) != 0, 1, 0)"
             for col in cols_master]
        )
        expr_sum_visa = " + ".join(
            [f"IF(SAFE_CAST(a.{col} AS FLOAT64) IS NOT NULL AND SAFE_CAST(a.{col} AS FLOAT64) != 0, 1, 0)"
             for col in cols_visa]
        )

        expr_sum_general = " + ".join(
            [
                f"IF(SAFE_CAST(a.{col} AS FLOAT64) IS NOT NULL "
                f"AND SAFE_CAST(a.{col} AS FLOAT64) != 0, 1, 0)"
                for col in cols_general
            ]
        )

        # convierto la lista en numeros unidos por , y primeor los paso a str para que no rompa
        drop_meses = ', '.join(map(str,config.MONTHS_DROP_LOAD))

        feature_table = config.BQ_TABLE_PRODUCTS
        logger.info(f"Creando tabla de {feature_table}...")
        query = f"""
        CREATE OR REPLACE TABLE `{PROJECT}.{DATASET}.{feature_table}`
        PARTITION BY RANGE_BUCKET(foto_mes, GENERATE_ARRAY(201901, 202208, 1))
        CLUSTER BY foto_mes, numero_de_cliente AS
        SELECT
          a.*,
          {expr_sum_master} AS q_producto_master,
          {expr_sum_visa} AS q_producto_visa,
          {expr_sum_general} AS q_producto_general,
            b.clase_ternaria
        FROM `{PROJECT}.{DATASET}.{TABLE}` a
        LEFT JOIN `{PROJECT}.{DATASET}.{TARGET_TABLE}` b on a.foto_mes = b.foto_mes and a.numero_de_cliente = b.numero_de_cliente
        where a.foto_mes not in ({drop_meses})
        ;
        """
        logger.info(query)
        client.query(query)
        logger.info(f"Se ha creado la tabla de {feature_table}")
    except Exception as e:
        logger.info(f"No se creo la tabla de q_productos_por_cliente_mes. error: {e}")
        logger.error(e)

def select_data_c03( MESES):
    try:
        PROJECT= config.BQ_PROJECT
        DATASET = config.BQ_DATASET
        TABLE = 'c03'
        # extraigo meses de la lista y concateno en string
        if isinstance(MESES, (list, tuple)):
            MESES = ", ".join(map(str, MESES))

        client = bigquery.Client(project=PROJECT)
        bqstorage_client = bigquery_storage.BigQueryReadClient()

        query = f"""
            SELECT 
                a.*
            FROM `{PROJECT}.{DATASET}.{TABLE}` AS a
            WHERE a.foto_mes IN ({MESES})
        """

        logger.info(f"Query para select_data_c03: {query}")
        # Ejecutar la query y traer resultados como ArrowTable (más eficiente)
        job = client.query(query)

        # Uso Storage API para traer Arrow más rápido
        arrow_table = job.result().to_arrow(bqstorage_client=bqstorage_client)


        # Convertir ArrowTable → Polars DataFrame
        df_pl = pl.from_arrow(arrow_table)

        ######### Contención de features que se transforman de  string a float ###############
        df_pl = df_pl.with_columns([
            pl.col("tmobile_app").cast(pl.Float64, strict=False),
            pl.col("cmobile_app_trx").cast(pl.Float64, strict=False),
            pl.col("Master_Finiciomora").cast(pl.Float64, strict=False),
            pl.col("Visa_Finiciomora").cast(pl.Float64, strict=False),
        ])
        ######### Contención de features que se transforman de  string a float ##########

        return df_pl

    except Exception as e:
        logger.error(f"Error ejecutando select_data_c0: {e}")
        return pl.DataFrame()  # devolver DF vacío en caso de error




