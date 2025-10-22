import logging
from datetime import datetime
from src.loader import create_dataset_c01, select_c01
from src.features import (get_numeric_columns_pl,
                          feature_engineering_lag,
                          feature_engineering_delta)
from src.optimization import (binary_target)
from src.config import (STUDY_NAME, DATA_PATH
                        , SEMILLA, MES_TRAIN,
                        MES_VALIDACION, MES_TEST,
                        GANANCIA_ACIERTO, COSTO_ESTIMULO, DB_PATH)
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
logger.info(f"SEMILLA: {SEMILLA}")
logger.info(f"MES_TRAIN: {MES_TRAIN}")
logger.info(f"MES_VALIDACION: {MES_VALIDACION}")
logger.info(f"MES_TEST: {MES_TEST}")
logger.info(f"GANANCIA_ACIERTO: {GANANCIA_ACIERTO}")
logger.info(f"COSTO_ESTIMULO: {COSTO_ESTIMULO}")


def main():
    logger.info(f"Entrenando con SEMILLA={SEMILLA}, TRAIN={MES_TRAIN}, VALID={MES_VALIDACION}")
    if CREAR_NUEVA_BASE:
        logger.info("Creando nueva base de datos...")
        create_dataset_c01(DB_PATH)
        #data = competencia_01_features(DB_PATH)

    else:
        logger.info("Usando base de datos existente...")

        # Cargo dataset
        logger.info("Cargando dataset comision_01...")
        data = select_c01(DB_PATH)

        #### FEATURE ENGINEERING

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
        #### FIN FEATURE ENGINEERING

        logger.info("Binarizando target...")
        data = binary_target(data)

    print("Fin Corrida")
    print(data.columns)
if __name__ == '__main__':
    main()