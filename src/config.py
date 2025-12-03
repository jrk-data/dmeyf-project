import yaml
import os
import logging

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

    SELECTED_RANKS = flag.get("SELECTED_RANKS", [])



    START_POINT = flag.get("START_POINT", "FEATURES")


    # ---- Competencia 02 ----
    MES_TRAIN = exp.get("MONTH_TRAIN", [202102])
    MES_VALIDACION = exp.get("MONTH_VALIDATION", [202103])
    MES_TEST = exp.get("MONTH_TEST", [202104])
    MES_PRED = exp.get("MONTH_PRED", [202106])

    K_ENVIO_PRED: exp.get("K_ENVIO_PRED", 20000)

    TARGET = _cfg3.get("TARGET", "target")
    ID_COL = _cfg3.get("ID_COL", "id")

    MONTHS_DROP_LOAD = _cfg3.get("MONTHS_DROP_LOAD", [202006])

    GANANCIA_ACIERTO = _cfg3.get("GANANCIA_ACIERTO", None)
    COSTO_ESTIMULO = _cfg3.get("COSTO_ESTIMULO", None)


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
    Configura los paths globales. Modifica DIR_MODELS para incluir
    una subcarpeta basada en el último mes de entrenamiento definido
    en el archivo del experimento (ej: corrida_02_b.yaml).
    """
    global DB_PATH, LOGS_PATH, OUTPUT_PATH, STORAGE_OPTUNA, DIR_MODELS, DATA_PATH, DB_MODELS_TRAIN_PATH
    global DATA_PATH_C02, DATA_PATH_C03, DATA_PATH_FEATURES, PATH_FEATURES_SELECTION

    # 1. Seleccionar paths base (VM o Local) desde config.yaml
    if is_vm_environment:
        paths = _cfgGeneral["vm"]
        logger.info("🔧 Entorno: VM")
    else:
        paths = _cfgGeneral["local"]
        logger.info("🔧 Entorno: LOCAL")

    # 2. Asignar variables estándar
    DB_PATH = paths.get("DB_PATH", "data/churn.duckdb")
    LOGS_PATH = paths.get("LOGS_PATH", "logs/")
    OUTPUT_PATH = paths.get("OUTPUT_PATH", "output/")
    STORAGE_OPTUNA = paths.get("STORAGE_OPTUNA", None)

    DATA_PATH_C02 = paths.get("DATA_PATH_C02", "data/competencia_01.csv")
    DATA_PATH_C03 = paths.get("DATA_PATH_C03", "data/competencia_01.csv")
    DB_MODELS_TRAIN_PATH = paths.get("DB_MODELS_TRAIN_PATH", "data/models_train_test.duckdb")
    DATA_PATH_FEATURES = paths.get("DATA_PATH_FEATURES", "data/features_train_test.csv")
    PATH_FEATURES_SELECTION = paths.get("PATH_FEATURES_SELECTION", "data/features_selection.csv")

    # -------------------------------------------------------------------------
    # 3. LÓGICA DINÁMICA PARA DIR_MODELS
    # -------------------------------------------------------------------------
    # Path base definido en config.yaml (ej: /home/.../models/)
    base_dir_models = paths.get("DIR_MODELS", "src/models/default/")

    try:
        # Recuperamos la sección 'experiment' que viene de corrida_02_b.yaml
        exp_config = _cfgGeneral.get("experiment", {})
        raw_train_months = exp_config.get("MONTH_TRAIN", [])

        if raw_train_months:
            # Tomamos el último mes para el nombre de la carpeta
            ultimo_mes = sorted(list(raw_train_months))[-1]
            suffix = f"meses_entrenados_hasta_{ultimo_mes}"

            # Construimos la ruta completa de forma segura
            DIR_MODELS = os.path.join(base_dir_models, suffix)
            logger.info(f"📂 DIR_MODELS dinámico: {DIR_MODELS}")
        else:
            # Si la lista está vacía, usamos el base
            DIR_MODELS = base_dir_models
            logger.warning("⚠️ MONTH_TRAIN vacío en el experimento. Usando ruta base.")

    except Exception as e:
        logger.error(f"❌ Error construyendo ruta dinámica: {e}")
        DIR_MODELS = base_dir_models

    # Limpieza final de la ruta (quita barras dobles si las hubiera)
    DIR_MODELS = os.path.normpath(DIR_MODELS)