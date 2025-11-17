import src.zlgbm as exp
from google.cloud import bigquery, bigquery_storage
import pandas as pd
import polars as pl
import logging
import re
import src.config as config
from src.loader import (select_data_c02, select_c02_polars, create_bq_table_c02, create_targets_c02,tabla_productos_por_cliente)
from src.features import (get_numeric_columns_pl, create_ipc_adjusted_table, creation_lags, creation_deltas)
from src.preprocessing import create_binary_target_column
from src.zlgbm import generar_rango_meses
import src.zlgbm as zlgbm

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =====================
# Config BigQuery
# =====================
PROJECT = "dmecoyfin-250928192534125"
DATASET = "dmeyf"
TABLE = "c02_delta"     # o tu tabla final con features
ID_COL = "numero_de_cliente"

client = bigquery.Client(project=PROJECT)
bqstorage_client = bigquery_storage.BigQueryReadClient()


def _select_table_schema(project, dataset, table):
    client = bigquery.Client(project=project)
    t = client.get_table(f"{project}.{dataset}.{table}")
    cols = [f.name for f in t.schema]
    return cols

def _columns_filter(cols, pattern):
    '''
    Se ingresa todas las columnas que se quiere eliminar.
    Devuelve array con columnas a seleccionar.
    '''

    if not pattern:
        return cols

    combined_pattern = "(" + "|".join(re.escape(p) for p in pattern) + ")"

    columns_to_keep = list(filter(lambda c: not re.search(combined_pattern, c), cols))

    return columns_to_keep

def _filter_lags_deltas(cols, k):
    filtradas = []
    for c in cols:
        # Si es lag o delta hasta 5
        if '_lag_' not in c and '_delta_' not in c:
            filtradas.append(c)
        elif re.search(rf'_lag_[1-{k}]$', c) or re.search(rf'_delta_[1-{k}]$', c):
            filtradas.append(c)
    return filtradas

def select_data_lags_deltas(tabla, columnas_excluir,meses, k):
    'Selecciona los campos de lags y deltas para un k y todos los campos que no son lags o deltas'
    logger.info(f"meses: {meses}")

    schema_table = _select_table_schema(config.BQ_PROJECT, config.BQ_DATASET, tabla)
    schema_table = _columns_filter(schema_table, columnas_excluir)

    columns = _filter_lags_deltas(schema_table, k)

    client = bigquery.Client(project=config.BQ_PROJECT)
    bqstorage_client = bigquery_storage.BigQueryReadClient()

    meses =  ", ".join(str(int(m)) for m in meses)

    query = f"""SELECT {', '.join(columns)} FROM `{config.BQ_PROJECT}.{config.BQ_DATASET}.{tabla}`
    where foto_mes in ({meses})"""
    #logger.info(f"Query: {query}")
    job = client.query(query)

    # Uso Storage API para traer Arrow más rápido
    arrow_table = job.result().to_arrow(bqstorage_client=bqstorage_client)
    df_pl = pl.from_arrow(arrow_table)
    return df_pl






# =============================
# Función de ejecución general
# =============================
def ejecutar_experimento(nombre, meses_train, mes_test1, mes_test2, mes_final):
    logger.info(f"\n{'=' * 50}")
    logger.info(f"Iniciando experimento: {nombre}")
    logger.info(f"{'=' * 50}")

    # Registrar configuración
    logger.info("Configuración del experimento:")
    logger.info(f"- Meses de entrenamiento: {meses_train}")
    logger.info(f"- Mes de prueba 1: {mes_test1}")
    logger.info(f"- Mes de prueba 2: {mes_test2}")
    logger.info(f"- Mes final: {mes_final}")
    logger.info(f"- Feature Engineering Lags: {exp.FEATURE_ENGINEERING_LAGS}")
    logger.info(f"- Proyecto BigQuery: {PROJECT}")
    logger.info(f"- Dataset: {DATASET}")
    logger.info(f"- Tabla: {TABLE}")

    # 1. Definir meses necesarios para la query
    meses_total = sorted(list(set(meses_train + [mes_test1, mes_test2, mes_final])))
    logger.info(f"Meses totales a procesar: {meses_total}")

    nombre_experimento = f"{nombre}_{mes_test1}_{mes_test2}_{mes_final}"

    if config.START_POINT == 'DATA' or config.CREAR_NUEVA_BASE:
        logger.info("Creando nueva base de datos...")

        # Selecciono datos crudos
        data = select_c02_polars(config.DATA_PATH)

        # Creo tabla en BQ a partir de datos Crudos
        create_bq_table_c02(data, config.BQ_PROJECT, config.BQ_DATASET, config.BQ_TABLE)

        # Creo targets
        create_targets_c02(config.BQ_PROJECT, config.BQ_DATASET, config.BQ_TABLE, config.BQ_TABLE_TARGETS)

        # Creo q_productos_cliente_mes
        # Acá filtro los meses que no van a entrar
        tabla_productos_por_cliente(config.BQ_PROJECT, config.BQ_DATASET, config.BQ_TABLE,
                                    config.BQ_TABLE_TARGETS)  # uso c02 y targets para joinear t crear c02_productos

        # ----------- Obtengo algunos datos para obtener tipos de columnas -------------
        data = select_data_c02([202102])
        # Columnas a excluir
        exclude_cols = ["numero_de_cliente", "foto_mes", "clase_binaria", "clase_binaria1", "clase_binaria2", "clase_peso"]
        # Creo array con columnas numéricas
        numeric_cols = get_numeric_columns_pl(data, exclude_cols=exclude_cols)

        # Creo tabla c02_ipc
        logger.info("Creando tabla c02_ipc...")
        create_ipc_adjusted_table()

        # Creo tabla con lags
        logger.info(f"Creando lags n= {config.NUN_WINDOW_LOAD}...")
        creation_lags(numeric_cols, config.NUN_WINDOW_LOAD)

        # Creo tabla con deltas
        logger.info("Creando deltas...")
        creation_deltas(numeric_cols, config.NUN_WINDOW_LOAD)

        # Binarizando target
        logger.info("Binarizando target...")
        table_with_deltas = 'c02_delta'
        create_binary_target_column(config.BQ_PROJECT, config.BQ_DATASET, table_with_deltas)

    # Selecciono los datos de los meses que se van a trabajar
    # data = select_data_c02(config.BQ_PROJECT, config.BQ_DATASET, table_with_deltas, meses)
    else:
        logger.info("Usando base de datos existente...")



    try:
        # 2. Cargar datos de BigQuery
        logger.info("Iniciando carga de datos desde BigQuery...")
        df = select_data_lags_deltas('c02_delta',config.COLUMNAS_EXCLUIR,meses_total,2)
        logger.info(f"Datos cargados exitosamente. Shape: {df.shape}")

        # 3. Configurar workflow
        exp.FOTO_MES_TRAIN_INICIO = min(meses_train)
        exp.FOTO_MES_TRAIN_FIN = max(meses_train)
        exp.FOTO_MES_TEST_1 = mes_test1
        exp.FOTO_MES_TEST_2 = mes_test2
        exp.FOTO_MES_FINAL = mes_final
        exp.FEATURE_ENGINEERING_LAGS = False
        exp.df_inicial = df

        # 4. Ejecutar workflow completo
        logger.info("Iniciando ejecución del workflow...")
        # Ejecutar workflow completo usando df y meses explícitos
        pred_final, df_testing, df_resultados, p1, p2, path = exp.main(
            nombre_experimento = nombre,
            df=df,
            meses_train=meses_train,
            mes_test1=mes_test1,
            mes_test2=mes_test2,
            mes_final=mes_final,
        )
        logger.info(f"Experimento finalizado exitosamente")
        logger.info(f"Outputs guardados en: {path}")

        return path

    except Exception as e:
        logging.error(f"Error durante la ejecución del experimento: {str(e)}", exc_info=True)
        raise

def ejecutar_solo_final(nombre_experimento,
                        df,
                        meses_train,
                        mes_final):
    """
    Ejecuta SOLO la etapa final:
       - preprocess
       - train final
       - predict final
       - submissions final
    """
    logger.info("=" * 80)
    logger.info(f"🎯 Iniciando SOLO FINAL: {nombre_experimento}")
    logger.info("=" * 80)

    try:
        pred_final, df_resultados, exp_path = zlgbm.main_solo_final(
            nombre_experimento=nombre_experimento,
            df=df,
            meses_train=meses_train,
            mes_final=mes_final,
        )

        logger.info(f"✔ SOLO FINAL '{nombre_experimento}' COMPLETADO")
        logger.info(f"📁 Ruta del experimento: {exp_path}")

        return pred_final, df_resultados, exp_path

    except Exception as e:
        logger.error("❌ Error durante SOLO FINAL")
        logger.exception(e)
        raise


# =============================
# EXPERIMENTO 1
# =============================
# meses_train_exp1 = [
#     201901,201902,201903,201904,
#     202001,202002,202003,202004,
#     202101,202102
# ]
#
# ejecutar_experimento(
#     "EXPERIMENTO_ESTACIONAL_ZLGBM",
#     meses_train_exp1,
#     mes_test1=202104,
#     mes_test2=202106,
#     mes_final=202108
# )

# =============================
# EXPERIMENTO 2
# # =============================#
# meses_train_exp2 = [m for m in generar_rango_meses(201901, 202103) if m != 202006]
#
# #meses_train_exp2 = [202001]
#
# ejecutar_experimento(
#     "EXP_NO_ESTACIONAL_ZLGBM_IPC",
#     meses_train_exp2,
#     mes_test1=202104,
#     mes_test2=202106,
#     mes_final=202108
#)

# ================================================================
# 3. SOLO FINAL: ENTRENAR CON MÁS MESES Y PREDECIR 202108
# ================================================================

# Queremos entrenar con 202001–202106 y predecir 202108
meses_train_final = list(range(202001, 202107))  # 202001..202106


df = select_data_lags_deltas('c02_delta',config.COLUMNAS_EXCLUIR,meses_train_final,2)

ejecutar_solo_final(
    nombre_experimento="SOLO_FINAL_202001_202106",
    df=df,
    meses_train=meses_train_final,
    mes_final=202108
)

