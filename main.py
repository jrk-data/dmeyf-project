import logging
from datetime import datetime
from src.loader import create_dataset_c01, select_c01
from src.features import (get_numeric_columns_pl,
                          feature_engineering_lag,
                          feature_engineering_delta)
from src.optimization import (binary_target,
                              split_train_data,
                              run_study
                              )
from src.config import (STUDY_NAME, DATA_PATH
                        , SEEDS, MES_TRAIN,
                        MES_VALIDACION, MES_TEST,
                        GANANCIA_ACIERTO, COSTO_ESTIMULO, DB_PATH,
                        STUDY_NAME_OPTUNA, STORAGE_OPTUNA)
import os
from datetime import datetime
import sys


# Set logs
os.makedirs("logs", exist_ok=True)
name_log = f"log_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"

fmt = "%(asctime)s - %(name)s - %(levelname)s - L%(lineno)d - %(message)s"

file_handler = logging.FileHandler(f"logs/{name_log}", mode="w", encoding="utf-8")
stream_handler = logging.StreamHandler(sys.stdout)

file_handler.setFormatter(logging.Formatter(fmt))
stream_handler.setFormatter(logging.Formatter(fmt))

# force=True borra handlers previos (útil si corrés varias veces)
logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, stream_handler], force=True)
logger = logging.getLogger(__name__)


CREAR_NUEVA_BASE = False
# =====================

logger = logging.getLogger(__name__)

### Manejo de Configuración en YAML ###
logger.info("Configuración cargada desde YAML")
logger.info(f"STUDY_NAME: {STUDY_NAME}")
logger.info(f"DATA_PATH: {DATA_PATH}")
logger.info(f"DB_PATH: {DB_PATH}")
logger.info(f"SEMILLA: {SEEDS[0]}")
logger.info(f"MES_TRAIN: {MES_TRAIN}")
logger.info(f"MES_VALIDACION: {MES_VALIDACION}")
logger.info(f"MES_TEST: {MES_TEST}")
logger.info(f"GANANCIA_ACIERTO: {GANANCIA_ACIERTO}")
logger.info(f"COSTO_ESTIMULO: {COSTO_ESTIMULO}")


def main():
    logger.info(f"Entrenando con SEMILLA={SEEDS[0]}, TRAIN={MES_TRAIN}, VALID={MES_VALIDACION}")
    if CREAR_NUEVA_BASE:
        logger.info("Creando nueva base de datos...")
        create_dataset_c01(DB_PATH)
        #data = competencia_01_features(DB_PATH)

    else:
        logger.info("Usando base de datos existente...")

        # Cargo dataset
        logger.info("Cargando dataset comision_01...")
        data = select_c01(DB_PATH)

        #############################
        #### FEATURE ENGINEERING ####
        ############################
        # Creación Lags
        logger.info(f"#### INICIO FEATURE ENGINEERING ###")
        logger.info("Creando Lags...")
        numeric_cols = get_numeric_columns_pl(data, exclude_cols=["numero_de_cliente","foto_mes"])
        data = feature_engineering_lag(
            data,
            numeric_cols,
            cant_lag=1,
        )

        # Creación DEltas
        logger.info("Creando Deltas...")
        data = feature_engineering_delta(data, numeric_cols,1)
        logger.info(f"Data shape: {data.shape}")

        logger.info(f"#### FIN FEATURE ENGINEERING ###")
        ################################
        #### FIN FEATURE ENGINEERING ####
        ################################

        logger.info("Binarizando target...")
        data = binary_target(data)

        ####################################
        #### OPTIMIZACIÓN HIPERPARAMETROS ####
        ####################################

        response_split = split_train_data(data,MES_TRAIN,MES_TEST)


        X_train = response_split["X_train_pl"]
        y_train_b1 = response_split["y_train_binaria1"]
        y_train_b2 = response_split["y_train_binaria2"]
        w_train = response_split["w_train"]
        y_test_class = response_split["y_test_class"]

        # Seteamos path de BBDD para el study

        storage_optuna = STORAGE_OPTUNA
        study_name_optuna = STUDY_NAME_OPTUNA
        logger.info(f"Seteando path de BBDD: {storage_optuna} - {study_name_optuna}")


        # Transformo de polars a pandas para usar study
        X_train = X_train.to_pandas()

        #y_train_b2= y_train_b2.to_pandas()
        #w_train = w_train.to_pandas()

        logger.info("Iniciando estudio...")
        run_study(X_train[:1000],
                              y_train_b2[:1000]
                              , SEEDS[0]
                              ,w_train[:1000]
                              ,None
                              ,storage_optuna
                              ,study_name_optuna)


    print("Fin Corrida")
    #print(data.columns)
if __name__ == '__main__':
    main()