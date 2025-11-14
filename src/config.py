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
        exp = _cfgGeneral["experiment"]
        bq = _cfgGeneral["bigquery"]
        opt = _cfgGeneral["optimization"]
        flag = _cfgGeneral["flags"]
        pr = _cfgGeneral["preprocessing"]
        seeds = _cfgGeneral["seeds"]

    # ---- Semillas ----
    SEEDS = seeds.get("SEEDS")
    SEED = seeds.get("SEED")

    # ---- Experimento ----
    CREAR_NUEVA_BASE = flag.get("NUEVA_BASE", False)
    STUDY_NAME = exp.get("STUDY_NAME", "default_study")
    STUDY_NAME_OPTUNA = exp.get("STUDY_NAME_OPTUNA",None)

    TEST_BY_TRAIN = exp.get("TEST_BY_TRAIN", {})  # dict[str->int]
    PREDICT_SCENARIOS = exp.get("PREDICT_SCENARIOS", [])  # list[dict]

    # ---- Optimization ----
    N_TRIALS = int(opt.get("N_TRIALS", 50))
    N_STARTUP_TRIALS = int(opt.get("N_STARTUP_TRIALS", 20))
    NFOLD = int(opt.get("NFOLD", 5))
    EARLY_STOPPING_ROUNDS = int(opt.get("EARLY_STOPPING_ROUNDS", 0))  # 0 = no usar
    N_BOOSTS = 1000
    # --- Preprocessing ----

    SUB_SAMPLE = pr.get("SUB_SAMPLE", None) # si es None, no se hace sub-sampling
    NUN_WINDOW_LOAD = pr.get("NUN_WINDOW_LOAD", 5)
    NUN_WINDOW =  pr.get("NUN_WINDOW", 3)

    # --- Flags ----
    # Busca la variable optimizar, por defecto queda en False
    OPTIMIZAR = flag.get("RUN_OPTIMIZATION", False)
    RUN_CALC_CURVAS = flag.get("RUN_CALC_CURVAS", False)
    TOP_K_MODEL = flag.get("TOP_K_MODEL", 5)




    START_POINT = flag.get("START_POINT", "FEATURES")



    # ---- Competencia 02 ----
    MES_TRAIN = exp.get("MONTH_TRAIN", [202102])
    MES_VALIDACION = exp.get("MONTH_VALIDATION", [202103])
    MES_TEST = exp.get("MONTH_TEST", [202104])
    MES_PRED = exp.get("MONTH_PRED", [202106])
    TARGET = _cfg2.get("TARGET", "target")
    ID_COL = _cfg2.get("ID_COL", "id")

    MONTHS_DROP_LOAD = _cfg2.get("MONTHS_DROP_LOAD", [202006])

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


def setup_environment(is_vm_environment):
    """
    Función que recibe un booleano y ejecuta el IF/ELSE
    correspondiente.
    """
    global DB_PATH, LOGS_PATH, OUTPUT_PATH, STORAGE_OPTUNA, DIR_MODELS, DATA_PATH, DB_MODELS_TRAIN_PATH

    if is_vm_environment:
        paths = _cfgGeneral["vm"]
    else:
        paths = _cfgGeneral["local"]
    # Defino los paths
    DB_PATH = paths.get("DB_PATH", "data/churn.duckdb")
    LOGS_PATH = paths.get("LOGS_PATH", "logs/")
    OUTPUT_PATH = paths.get("OUTPUT_PATH", "output/")
    STORAGE_OPTUNA = paths.get("STORAGE_OPTUNA", None)
    DIR_MODELS = paths.get("DIR_MODELS", "src/models/default/")
    DATA_PATH = paths.get("DATA_PATH", "data/competencia_01.csv")
    DB_MODELS_TRAIN_PATH = paths.get("DB_MODELS_TRAIN_PATH", "data/models_train_test.duckdb")


#  MONTH_TRAIN: [   201901,201902,201903,201904,201905,201906,201907,
#                  201908,201909,201910,201911,201912,202001,202002,
#                  202003,202004,202005,202006, 202007, 202008, 202009,
#                  202010, 202011, 202012, 202101,202102 ]