import logging
from datetime import datetime

from colorlog import exception

from src.loader import create_dataset_c01, select_c01
from src.features import (get_numeric_columns_pl,
                          feature_engineering_lag,
                          feature_engineering_delta)
from src.optimization import (binary_target,
                              split_train_data,
                              run_study
                              )
from src.config import (CREAR_NUEVA_BASE, DATA_PATH
                        , SEEDS, MES_TRAIN,
                        MES_VALIDACION, MES_TEST,
                        GANANCIA_ACIERTO, COSTO_ESTIMULO, DB_PATH,
                        STUDY_NAME_OPTUNA, STORAGE_OPTUNA, OPTIMIZAR
                        , DIR_MODELS, MES_PRED, START_POINT)

from src.train_test import (train_model
                            , calculo_curvas_ganancia
                            ,pred_ensamble_modelos)

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


CREAR_NUEVA_BASE = CREAR_NUEVA_BASE
# =====================

logger = logging.getLogger(__name__)

### Manejo de Configuración en YAML ###
logger.info("Configuración cargada desde YAML")
logger.info(f"STUDY_NAME: {STUDY_NAME_OPTUNA}")
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
    logger.info(f"Iniciando Pipeline desde el punto: {START_POINT}")

    try:
        # ---------------------------------------------------------------------------------
        # 1. CREACIÓN/CARGA DE DATOS (START_POINT == 'DATA')
        # ---------------------------------------------------------------------------------
        if START_POINT == 'DATA':
            logger.info("Creando nueva base de datos...")
            # NOTA: Usamos CREAR_NUEVA_BASE aquí si es necesario, aunque START_POINT es más limpio.
            create_dataset_c01(DB_PATH)

        logger.info("Usando base de datos existente...")
        # Cargo dataset base
        logger.info("Cargando dataset comision_01...")
        data = select_c01(DB_PATH)  # <-- 'data' es necesario para los siguientes pasos

        # Binarizar target
        logger.info("Binarizando target...")
        data = binary_target(data)

        # ---------------------------------------------------------------------------------
        # 2. FEATURE ENGINEERING (START_POINT == 'FEATURES')
        # ---------------------------------------------------------------------------------
        if START_POINT in ['FEATURES', 'OPTUNA', 'TRAIN', 'PREDICT']:
            logger.info(f"#### INICIO FEATURE ENGINEERING ###")
            logger.info("Creando Lags...")
            numeric_cols = get_numeric_columns_pl(data, exclude_cols=["numero_de_cliente", "foto_mes"])
            data = feature_engineering_lag(
                data,
                numeric_cols,
                cant_lag=1,
            )

            logger.info("Creando Deltas...")
            data = feature_engineering_delta(data, numeric_cols, 1)
            logger.info(f"Data shape: {data.shape}")
            logger.info(f"#### FIN FEATURE ENGINEERING ###")

        # El resto del pipeline necesita que los datos estén spliteados
        response_split = split_train_data(data, MES_TRAIN, MES_TEST, MES_PRED)

        X_train = response_split["X_train_pl"].to_pandas()
        y_train_b2 = response_split["y_train_binaria2"]
        w_train = response_split["w_train"]
        y_test_class = response_split["y_test_class"]
        X_test = response_split["X_test_pl"].to_pandas()  # Convertido a Pandas

        X_pred = response_split["X_pred_pl"].to_pandas()
        # ---------------------------------------------------------------------------------
        # 3. OPTIMIZACIÓN HIPERPARÁMETROS (START_POINT == 'OPTUNA')
        # ---------------------------------------------------------------------------------
        study = None
        storage_optuna = STORAGE_OPTUNA
        study_name_optuna = STUDY_NAME_OPTUNA

        if START_POINT in ['OPTUNA', 'TRAIN', 'PREDICT']:
            logger.info(f"Seteando path de BBDD Optuna: {storage_optuna} - {study_name_optuna}")
            logger.info("Iniciando estudio...")

            # 'run_study' maneja la carga o creación del estudio Optuna
            study = run_study(X_train,
                              y_train_b2
                              , SEEDS[0]
                              , w_train
                              , None
                              , storage_optuna
                              , study_name_optuna
                              , optimizar=OPTIMIZAR)
            logger.info("#### FIN OPTIMIZACIÓN HIPERPARAMETROS ####")

        # ---------------------------------------------------------------------------------
        # 4. ENTRENAMIENTO Y CÁLCULO CURVAS (START_POINT == 'TRAIN')
        # ---------------------------------------------------------------------------------
        top_k_model = 5

        if START_POINT in ['TRAIN', 'PREDICT']:

            # Si no pasamos por OPTUNA, necesitamos el objeto study cargado
            if START_POINT == 'TRAIN' and study is None:
                logger.info("Cargando estudio Optuna existente para entrenamiento...")
                # Aquí deberías tener una función para CARGAR el study si OPTIMIZAR es False
                # OJO: La función run_study ya maneja la carga si OPTIMIZAR=False
                study = run_study(X_train, y_train_b2, SEEDS[0], w_train, None, storage_optuna, study_name_optuna,
                                  optimizar=False)

            logger.info("Entrenando modelos Top-K...")
            train_model(study, X_train, y_train_b2, w_train, top_k_model)

            logger.info("Calculando curvas de ganancia...")
            y_predicciones, curvas, mejores_cortes_normalizado = calculo_curvas_ganancia(X_test, y_test_class,
                                                                                         DIR_MODELS, study_name_optuna)
            print(mejores_cortes_normalizado)

        # ---------------------------------------------------------------------------------
        # 5. PREDICCIÓN FINAL/ENSEMBLE (START_POINT == 'PREDICT')
        # ---------------------------------------------------------------------------------
        if START_POINT == 'PREDICT':
            logger.info("Realizando predicción final (Ensemble)...")
            # El número 6 es el parámetro 'k' que tenías en main.py para pred_ensamble_modelos
            pred_ensamble_modelos(X_pred, DIR_MODELS, study_name_optuna, 6)
    except  Exception as e:
        logger.error(f'Se cortó ejecución por un error:\n {e}')

    logger.info("Fin Corrida")
    #print(data.columns)
if __name__ == '__main__':
    main()