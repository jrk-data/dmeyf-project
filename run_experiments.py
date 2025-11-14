import src.zlgbm as exp
from google.cloud import bigquery, bigquery_storage
import pandas as pd
import logging

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


def cargar_bigquery(meses):
    query = f"""
    SELECT *
    FROM `{PROJECT}.{DATASET}.{TABLE}`
    WHERE foto_mes IN ({",".join(map(str, meses))})
    """
    logger.info(f"Cargando datos desde BigQuery: {query}")
    df = (
        client.query(query)
        .result()
        .to_dataframe(bqstorage_client=bqstorage_client)
    )
    logger.info(f"Datos cargados exitosamente. Shape: {df.shape}")
    return df


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

    try:
        # 2. Cargar datos de BigQuery
        logger.info("Iniciando carga de datos desde BigQuery...")
        df = cargar_bigquery(meses_total)
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
        pred_final, df_testing, df_resultados, p1, p2, path = exp.main()

        logger.info(f"Experimento finalizado exitosamente")
        logger.info(f"Outputs guardados en: {path}")

        return path

    except Exception as e:
        logging.error(f"Error durante la ejecución del experimento: {str(e)}", exc_info=True)
        raise


# =============================
# EXPERIMENTO 1
# =============================
meses_train_exp1 = [
    201901,201902,201903,201904,
    202001,202002,202003,202004,
    202101,202102
]

ejecutar_experimento(
    "EXPERIMENTO_ESTACIONAL_ZLGBM",
    meses_train_exp1,
    mes_test1=202104,
    mes_test2=202106,
    mes_final=202108
)


# =============================
# EXPERIMENTO 2
# =============================
meses_train_exp2 = list(range(202004, 202103))

ejecutar_experimento(
    "EXPERIMENTO_NO_ESTACIONAL_ZLGBM",
    meses_train_exp2,
    mes_test1=202004,
    mes_test2=202006,
    mes_final=202008
)
