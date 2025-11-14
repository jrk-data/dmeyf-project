import src.zlgbm as exp
from google.cloud import bigquery, bigquery_storage
import pandas as pd

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
    df = (
        client.query(query)
        .result()
        .to_dataframe(bqstorage_client=bqstorage_client)
    )
    return df


# =============================
# Función de ejecución general
# =============================
def ejecutar_experimento(nombre, meses_train, mes_test1, mes_test2, mes_final):
    print(f"\n\n==============================")
    print(f"   Ejecutando {nombre}")
    print(f"==============================")

    # 1. Definir meses necesarios para la query
    meses_total = sorted(list(set(meses_train + [mes_test1, mes_test2, mes_final])))

    # 2. Cargar datos de BigQuery
    df = cargar_bigquery(meses_total)

    # 3. Configurar workflow
    exp.FOTO_MES_TRAIN_INICIO = min(meses_train)
    exp.FOTO_MES_TRAIN_FIN = max(meses_train)
    exp.FOTO_MES_TEST_1 = mes_test1
    exp.FOTO_MES_TEST_2 = mes_test2
    exp.FOTO_MES_FINAL = mes_final

    # No quiero lags – ya los genero en BQ
    exp.FEATURE_ENGINEERING_LAGS = False

    # Pasamos el dataframe manualmente
    exp.df_inicial = df

    # 4. Ejecutar workflow completo
    pred_final, df_testing, df_resultados, p1, p2, path = exp.main()

    print(f"\n✔ Finalizado: {nombre}")
    print(f"   Outputs en: {path}")
    return path


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
