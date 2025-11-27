import yaml
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Rutas base
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # Raíz del proyecto
PATH_CONFIG_GLOBAL = os.path.join(BASE_DIR, "config.yaml")

try:
    # 1. Cargar Configuración Global (Infraestructura)
    with open(PATH_CONFIG_GLOBAL, "r", encoding="utf-8") as f:
        _cfgGlobal = yaml.safe_load(f)

    # 2. Detectar qué experimento correr
    exp_file_rel_path = _cfgGlobal.get("EXPERIMENT_FILE")
    if not exp_file_rel_path:
        raise ValueError("El archivo config.yaml no tiene definida la variable 'EXPERIMENT_FILE'")

    path_experiment = os.path.join(BASE_DIR, exp_file_rel_path)
    logger.info(f"🔄 Cargando configuración del experimento: {exp_file_rel_path}")

    # 3. Cargar Configuración del Experimento
    with open(path_experiment, "r", encoding="utf-8") as f:
        _cfgExp = yaml.safe_load(f)

    # 4. Fusionar diccionarios (El experimento extiende a la global)
    # Creamos un diccionario unificado
    _cfgGeneral = _cfgGlobal.copy()
    _cfgGeneral.update(_cfgExp)

    # --- A PARTIR DE ACÁ TU CÓDIGO SIGUE IGUAL ---
    # Ya que _cfgGeneral ahora tiene TODAS las claves (las de config.yaml y las de exp_XX.yaml)

    _cfg = _cfgGeneral.get("competencia01", {})  # Usa .get para evitar error si no está definido
    _cfg3 = _cfgGeneral["competencia03"]
    exp = _cfgGeneral["experiment"]
    bq = _cfgGeneral["bigquery"]
    opt = _cfgGeneral["optimization"]
    flag = _cfgGeneral["flags"]
    pr = _cfgGeneral["preprocessing"]
    seeds = _cfgGeneral["seeds"]

    # DATA CRUDA
    DATA_PATH_C02 = _cfgGeneral.get("DATA_PATH_C02", "data/competencia_01.csv")
    DATA_PATH_C03 = _cfgGeneral.get("DATA_PATH_C03", "data/competencia_01.csv")
    PATH_FEATURES_SELECTION = _cfgGeneral.get("PATH_FEATURES_SELECTION", "data/features_selection.txt")
    CARGAR_HISTORIA_COMPLETA = _cfgGeneral.get("CARGAR_HISTORIA_COMPLETA", False)
    # ---- Semillas ----
    SEEDS = seeds.get("SEEDS")
    SEED = seeds.get("SEED")

    # ---- Experimento ----
    CREAR_NUEVA_BASE = flag.get("NUEVA_BASE", False)
    STUDY_NAME = exp.get("STUDY_NAME", "default_study")
    STUDY_NAME_OPTUNA = exp.get("STUDY_NAME_OPTUNA",None)

    PREDICT_SCENARIOS = exp.get("PREDICT_SCENARIOS", [])  # list[dict]

    MESES_JUNTOS = exp.get("MESES_JUNTOS", False)

    CONSOLIDATED_PATH = _cfgGeneral.get("CONSOLIDATED_PATH", "CONSOLIDATED")

    # ---- Optimization ----
    N_TRIALS = int(opt.get("N_TRIALS", 50))
    N_STARTUP_TRIALS = int(opt.get("N_STARTUP_TRIALS", 20))
    NFOLD = int(opt.get("NFOLD", 5))
    EARLY_STOPPING_ROUNDS = int(opt.get("EARLY_STOPPING_ROUNDS", 0))  # 0 = no usar
    N_BOOSTS = 1000
    FIXED_PARAMS_REF = opt.get("fixed_params", {})
    SEARCHABLE_PARAMS_REF = opt.get("searchable_params", {})

    # --- Preprocessing ----
    PSI_2021_FEATURES = pr.get("PSI_2021_FEATURES", None)
    SUB_SAMPLE = pr.get("SUB_SAMPLE", None) # si es None, no se hace sub-sampling
    NUN_WINDOW_LOAD = pr.get("NUN_WINDOW_LOAD", 5)
    NUN_WINDOW =  pr.get("NUN_WINDOW", 2)

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
    TARGET = _cfg3.get("TARGET", "target")
    ID_COL = _cfg3.get("ID_COL", "id")

    MONTHS_DROP_LOAD = _cfg3.get("MONTHS_DROP_LOAD", [202006])

    GANANCIA_ACIERTO = _cfg.get("GANANCIA_ACIERTO", None)
    COSTO_ESTIMULO = _cfg.get("COSTO_ESTIMULO", None)


    # Competencia 03
    BQ_PROJECT = _cfg3.get("BQ_PROJECT", None)
    BQ_DATASET = _cfg3.get("BQ_DATASET", None)
    BQ_TABLE = _cfg3.get("BQ_TABLE", None)
    BQ_TABLE_TARGETS = _cfg3.get("BQ_TABLE_TARGETS", None)
    BQ_TABLE_PRODUCTS = bq.get("BQ_TABLE_PRODUCTS", None)
    BQ_TABLE_FEATURES = _cfg3.get("BQ_TABLE_FEATURES", None)
    BQ_TABLE_FEATURES_HISTORICAL = _cfg3.get("BQ_TABLE_FEATURES_HISTORICAL", None)

    COLUMNAS_EXCLUIR = pr.get("COLUMNAS_EXCLUIR", [])


except Exception as e:
    logger.error(f"Error al cargar el archivo de configuracion: {e}")
    raise


def setup_environment(is_vm_environment):
    """
    Función que recibe un booleano y ejecuta el IF/ELSE
    correspondiente.
    """
    global DB_PATH, LOGS_PATH, OUTPUT_PATH, STORAGE_OPTUNA, DIR_MODELS, DATA_PATH, DB_MODELS_TRAIN_PATH, PATH_FEATURES_SELECTION

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
    DATA_PATH_C02 = paths.get("DATA_PATH_C02", "data/competencia_01.csv")
    DATA_PATH_C03 = paths.get("DATA_PATH_C03", "data/competencia_01.csv")
    DB_MODELS_TRAIN_PATH = paths.get("DB_MODELS_TRAIN_PATH", "data/models_train_test.duckdb")
    DATA_PATH_FEATURES = paths.get("DATA_PATH_FEATURES", "data/features_train_test.csv")
    PATH_FEATURES_SELECTION = paths.get("PATH_FEATURES_SELECTION", "data/features_selection.csv")

#  MONTH_TRAIN: [   201901,201902,201903,201904,201905,201906,201907,
#                  201908,201909,201910,201911,201912,202001,202002,
#                  202003,202004,202005,202006, 202007, 202008, 202009,
#                  202010, 202011, 202012, 202101,202102 ]