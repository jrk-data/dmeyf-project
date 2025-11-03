import yaml
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


#BASE = Path(__file__).resolve().parents[1]

# DB_PATH = BASE / "data" / "churn.duckdb"
# CSV_COMP = (BASE / "data" / "competencia_01_crudo.csv").as_posix()
#CSV_DIC  = (BASE / "data" / "DiccionarioDatos_2025 - Diccionario.csv").as_posix()


# Ruta del archivo de configuracion (al nivel raíz del proyecto)
PATH_CONFIG = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")

try:
    with open(PATH_CONFIG, "r", encoding="utf-8") as f:
        _cfgGeneral = yaml.safe_load(f)

    _cfg = _cfgGeneral["competencia01"]
    _cfg2 = _cfgGeneral["competencia02"]
    # Generales
    CREAR_NUEVA_BASE = _cfgGeneral.get("NUEVA_BASE", False)
    DB_PATH = _cfgGeneral.get("DB_PATH", "data/churn.duckdb")
    CSV_COMP = _cfgGeneral.get("CSV_COMP", "data/competencia_01_crudo.csv")
    STUDY_NAME = _cfgGeneral.get("STUDY_NAME", "default_study")
    STORAGE = _cfgGeneral.get("STORAGE", None)
    N_TRIALS = int(_cfgGeneral.get("N_TRIALS", 50))
    N_STARTUP_TRIALS = int(_cfgGeneral.get("N_STARTUP_TRIALS", 20))
    NFOLD = int(_cfgGeneral.get("NFOLD", 5))
    EARLY_STOPPING_ROUNDS = int(_cfgGeneral.get("EARLY_STOPPING_ROUNDS", 0))  # 0 = no usar
    LOGS_PATH = _cfgGeneral.get("LOGS_PATH", "logs/")

    STORAGE_OPTUNA = _cfgGeneral.get("STORAGE_OPTUNA", None)
    STUDY_NAME_OPTUNA = _cfgGeneral.get("STUDY_NAME_OPTUNA",None)

    # Busca la variable optimizar, por defecto queda en False
    OPTIMIZAR = _cfgGeneral.get("OPTIMIZAR", False)


    # PATH MODELO
    DIR_MODELS = _cfgGeneral.get("DIR_MODELS", ".src/models/default/")

    DB_MODELS_TRAIN_PATH = _cfgGeneral.get("DB_MODELS_TRAIN_PATH", "data/models_train_test.duckdb")

    START_POINT = _cfgGeneral.get("START_POINT", "FEATURES")

    # Competencia 01
    DATA_PATH = _cfg.get("DATA_PATH", "data/competencia_01.csv")
    SEEDS = _cfg.get("SEEDS")
    MES_TRAIN = _cfg.get("MONTH_TRAIN", [202102])
    MES_VALIDACION = _cfg.get("MONTH_VALIDATION", [202103])
    MES_TEST = _cfg.get("MONTH_TEST", [202104])
    MES_PRED = _cfg.get("MONTH_PRED", [202105])
    TARGET = _cfg.get("TARGET", "target")
    ID_COL = _cfg.get("ID_COL", "id")

    GANANCIA_ACIERTO = _cfg.get("GANANCIA_ACIERTO", None)
    COSTO_ESTIMULO = _cfg.get("COSTO_ESTIMULO", None)


    # Competencia 02
    BQ_PROJECT = _cfg2.get("BQ_PROJECT", None)
    BQ_DATASET = _cfg2.get("BQ_DATASET", None)
    BQ_TABLE = _cfg2.get("BQ_TABLE", None)
    BQ_TABLE_TARGETS = _cfg2.get("BQ_TABLE_TARGETS", None)


except Exception as e:
    logger.error(f"Error al cargar el archivo de configuracion: {e}")
    raise
