import duckdb
import os

from fontTools.misc.cython import returns

from src.config import DB_PATH
from pathlib import Path
import logging
from google.cloud import bigquery, bigquery_storage # storage para que lea mas rápido
import polars as pl
import pyarrow as pa



logger = logging.getLogger(__name__)


#
# def create_dataset_c01(DB_PATH,CSV_COMP = CSV_COMP):
#     CSV_COMP = CSV_COMP
#     logger.info(f"Creando base de datos en\n {DB_PATH} \n {CSV_COMP}")
#     try:
#
#         con = duckdb.connect(str(DB_PATH))
#
#
#         query_competencia_01 = f'''
#         CREATE OR REPLACE TABLE competencia_01 AS
#         SELECT * FROM read_csv_auto('{CSV_COMP}');
#
#         ALTER TABLE competencia_01
#         ADD COLUMN IF NOT EXISTS clase_ternaria VARCHAR DEFAULT NULL;
#         '''
#         con.sql(query_competencia_01)
#
#         query_crear_categorias = '''
#         create or replace table clases_ternarias as
#         with usuarios_ultimo_a_primer_es as(
#           select
#           foto_mes
#           , numero_de_cliente
#           , row_number() over (partition by numero_de_cliente order by foto_mes desc) as row_number
#           from competencia_01
#         ) select
#         foto_mes
#         ,numero_de_cliente
#         , case
#         when row_number = 1 and foto_mes < 202106 then 'BAJA+1'
#         when row_number = 2 and foto_mes < 202105 then 'BAJA+2'
#         when row_number >= 3 then 'CONTINUA'
#         else null
#         end as clase_ternaria
#         from usuarios_ultimo_a_primer_es;
#         '''
#
#         query_update_competencia_01 = '''
#         update competencia_01
#         set clase_ternaria = clases_ternarias.clase_ternaria
#         from clases_ternarias
#         where competencia_01.numero_de_cliente = clases_ternarias.numero_de_cliente and competencia_01.foto_mes = clases_ternarias.foto_mes;
#         '''
#
#         con.sql(query_crear_categorias)
#
#         con.sql(query_update_competencia_01)
#
#         con.close()
#     except Exception as e:
#         logger.error(e)
#         con.close()
#     finally:
#         con.close()
#
#
# def select_c01(DB_PATH):
#     try:
#         logger.info("Cargando base de datos competencia_01...")
#         con = duckdb.connect(str(DB_PATH))
#         query = con.sql("SELECT * FROM competencia_01").pl()
#
#         # para chequear que levanta el dataset
#         logger.info(query.head(5))
#
#         con.close()
#
#     except Exception as e:
#         con.close()
#         logger.error(e)
#     finally:
#         con.close()
#         logger.info("Se ha cargado la base de datos")
#     return query



# ################### Lógica para competencia 2 ##############################

def select_c02_polars(path_csv_competencia_2):
    '''
    Esta función lee el csv de competencia 2 y lo convierte a polars.
    Hace una lectura rápida de los primeors 10 registros del csv para crear el schema y pasarlo luego a BigQuery.
    '''
    # tomo una muestra para obtener schema
    logger.info("Creando schema para competencia_02...")
    try:
        df = pl.read_csv(
        "gs://joaquinrk_data_bukito3/datasets/competencia_02_crudo.csv.gz",
        n_rows=10,
        schema_overrides={"mprestamos_prendarios": pl.Float64},
        infer_schema_length=1000,
        )
        schema_actual = df.schema

        schema_modificado = {
            col: (pl.Float64 if (col.startswith("m") or "_m" in col) and col != 'foto_mes' and dtype == pl.Int64 else dtype)
            for col, dtype in schema_actual.items()
        }
    except Exception as e:
        logger.error(f'Error al crear schema: {e}')
        raise
    try:
        # Leo todo el csv con el schema que le paso a polars
        df = pl.read_csv(
            path_csv_competencia_2,
            schema_overrides=schema_modificado,
            infer_schema_length=0  # desactiva inferencia de tipos
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
            range_=bigquery.PartitionRange(start=201901, end=202108, interval=1),
        ),
        clustering_fields=["foto_mes","numero_de_cliente"],  
    )

    # 5) Cargar DataFrame
    load_job = client.load_table_from_dataframe(df_pd, TABLE_ID, job_config=job_config)
    load_job.result()  # espera a que termine

    #tbl = client.get_table(TABLE_ID)

    #return tbl

def create_targets_c02(PROJECT, DATASET, TABLE, TARGET_TABLE):
    logger.info("Creando tabla de clases ternarias...")
    try:
        client = bigquery.Client(project=PROJECT)
        query = f"""
        create or replace table `{PROJECT}.{DATASET}.{TARGET_TABLE}`
        PARTITION BY RANGE_BUCKET(foto_mes, GENERATE_ARRAY(201901, 202108, 1))
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

def select_data_with_targets_c02(PROJECT, DATASET, TABLE, TARGET_TABLE):
    logger.info("Creando tabla de clases ternarias...")
    try:
        client = bigquery.Client(project=PROJECT)
        query = f"""
        create or replace table `{PROJECT}.{DATASET}.{TARGET_TABLE}`
        PARTITION BY RANGE_BUCKET(foto_mes, GENERATE_ARRAY(201901, 202108, 1))
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
        cols_master = [f.name for f in table.schema if "master" in f.name.lower()]
        cols_visa = [f.name for f in table.schema if "visa" in f.name.lower()]
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

        FEATURE_TABLE = "q_productos_por_cliente_mes"

        query = f"""
        CREATE OR REPLACE TABLE `{PROJECT}.{DATASET}.{FEATURE_TABLE}`
        PARTITION BY RANGE_BUCKET(foto_mes, GENERATE_ARRAY(201901, 202108, 1))
        CLUSTER BY foto_mes, numero_de_cliente AS
        SELECT
          a.foto_mes,
          a.numero_de_cliente,
          {expr_sum_master} AS q_producto_master,
          {expr_sum_visa} AS q_producto_visa,
          {expr_sum_general} AS q_producto_general,
          CAST(
            CONCAT(
              SUBSTR(CAST(a.foto_mes AS STRING), 1, 4), '-',
              SUBSTR(CAST(a.foto_mes AS STRING), 5, 2), '-01'
            ) AS DATE
          ) AS foto_mes_date,
            b.clase_ternaria
        FROM `{PROJECT}.{DATASET}.{TABLE}` a
        INNER JOIN `{PROJECT}.{DATASET}.{TARGET_TABLE}` b on a.foto_mes = b.foto_mes and a.numero_de_cliente = b.numero_de_cliente
        ;
        """
        client.query(query)
        logger.info("Se ha creado la tabla de q_productos_por_cliente_mes")
    except Exception as e:
        logger.error(e)


def select_data_c02(PROJECT, DATASET, TABLE,  MESES):
    try:

        # extraigo meses de la lista y concateno en string
        if isinstance(MESES, (list, tuple)):
            MESES = ", ".join(map(str, MESES))

        client = bigquery.Client(project=PROJECT)
        bqstorage_client = bigquery_storage.BigQueryReadClient()

        query = f"""
            SELECT 
                a.*, 
                b.* except(foto_mes, numero_de_cliente,foto_mes_date)
            FROM `{PROJECT}.{DATASET}.{TABLE}` AS a
            INNER JOIN `{PROJECT}.{DATASET}.q_productos_por_cliente_mes` AS b -- esta tabla ya tiene clase_ternaria
            on a.foto_mes = b.foto_mes AND a.numero_de_cliente = b.numero_de_cliente
            WHERE a.foto_mes IN ({MESES})
        """
        logger.info(f"Query para select_data_c02: {query}")
        # Ejecutar la query y traer resultados como ArrowTable (más eficiente)
        job = client.query(query)

        # Uso Storage API para traer Arrow más rápido
        arrow_table = job.result().to_arrow(bqstorage_client=bqstorage_client)

        ######### Contención de features que se transforman de  string a float ###############
        # Definí un schema explícito
        schema = pa.schema([
            ("tmobile_app", pa.float64()),
            ("cmobile_app_trx", pa.float64()),
            ("Master_Finiciomora", pa.float64()),
            ("Visa_Finiciomora", pa.float64()),
        ])

        # Re-casteá el Arrow Table antes de pasarlo a Polars
        arrow_table = arrow_table.cast(schema, safe=False)
        ######### Contención de features que se transforman de  string a float ##########

        # Convertir ArrowTable → Polars DataFrame
        df_pl = pl.from_arrow(arrow_table)

        ########### CODIGO DEBUGEO TIPO DE DATO ########################
        # LOGS PARA VER TIPOS DE DATOS
        type_counts = {}
        for dtype in df_pl.schema.values():
            dtype_str = str(dtype)
            type_counts[dtype_str] = type_counts.get(dtype_str, 0) + 1

        logger.info(f"Conteo de tipos Polars: {type_counts}")

        str_cols = [c for c, t in df_pl.schema.items() if str(t) in ("Utf8", "String")]
        logger.info(f"Columnas String: {str_cols}")

        # ver nulos y ejemplos rápidos
        perfil_str = df_pl.select(
            *[pl.struct(
                col=pl.lit(c),
                n_null=pl.col(c).null_count(),
                n_unique=pl.col(c).n_unique(),
                sample=pl.col(c).drop_nulls().head(5)
            ) for c in str_cols]
        ).to_dicts()
        logger.info(f"Perfil columnas String: {perfil_str}")




    ############## fin debuggeo #######################################

        return df_pl

    except Exception as e:
        logger.error(f"Error ejecutando select_data_c02: {e}")
        return pl.DataFrame()  # devolver DF vacío en caso de error

