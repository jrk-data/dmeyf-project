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

    # Generales
    DB_PATH = _cfgGeneral.get("DB_PATH", "data/churn.duckdb")
    CSV_COMP = _cfgGeneral.get("CSV_COMP", "data/competencia_01_crudo.csv")
    STUDY_NAME = _cfgGeneral.get("STUDY_NAME", "default_study")
    STORAGE = _cfgGeneral.get("STORAGE", None)
    N_TRIALS = int(_cfgGeneral.get("N_TRIALS", 50))
    N_STARTUP_TRIALS = int(_cfgGeneral.get("N_STARTUP_TRIALS", 20))
    NFOLD = int(_cfgGeneral.get("NFOLD", 5))
    EARLY_STOPPING_ROUNDS = int(_cfgGeneral.get("EARLY_STOPPING_ROUNDS", 0))  # 0 = no usar

    # Competencia
    DATA_PATH = _cfg.get("DATA_PATH", "data/competencia_01.csv")
    SEEDS = _cfg.get("SEEDS")
    MES_TRAIN = _cfg.get("MES_TRAIN", "202102")
    MES_VALIDACION = _cfg.get("MES_VALIDACION", "202103")
    MES_TEST = _cfg.get("MES_TEST", "202104")
    TARGET = _cfg.get("TARGET", "target")
    ID_COL = _cfg.get("ID_COL", "id")

    GANANCIA_ACIERTO = _cfg.get("GANANCIA_ACIERTO", None)
    COSTO_ESTIMULO = _cfg.get("COSTO_ESTIMULO", None)

except Exception as e:
    logger.error(f"Error al cargar el archivo de configuracion: {e}")
    raise
