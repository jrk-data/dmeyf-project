# === Bootstrap mínimo de logging a consola (no depende de config) ===
import argparse
import sys
import logging
from datetime import datetime
from pathlib import Path

# Consola básica para errores tempranos
logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
    format="%(asctime)s - %(name)s - %(levelname)s - L%(lineno)d - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)
logger.info("Bootstrap de logger a consola listo")

# --- Parseo de argumentos bien temprano ---
parser = argparse.ArgumentParser(description="Script principal con opción de entorno VM.")
parser.add_argument(
    "-vm", "--virtual-machine",
    action="store_true",
    help="Establece que el script se ejecuta en la máquina virtual."
)
args, unknown = parser.parse_known_args()
logger.info(f"Detectado flag -vm: {args.virtual_machine}")

# --- Import tardío de config y seteo de entorno ---
import src.config as config  # importa el módulo completo
config.setup_environment(args.virtual_machine)

# --- Ahora que config tiene valores correctos, agregamos file handler ---
LOGS_PATH = Path(config.LOGS_PATH)
LOGS_PATH.mkdir(parents=True, exist_ok=True)
name_log = f"log_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"

file_handler = logging.FileHandler(LOGS_PATH / name_log, mode="w", encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - L%(lineno)d - %(message)s"))

root_logger = logging.getLogger()
root_logger.addHandler(file_handler)

logger.info("Logger con archivo inicializado")
logger.info("Configuración cargada:")
logger.info(f"CREAR_NUEVA_BASE: {config.CREAR_NUEVA_BASE}")
logger.info(f"STUDY_NAME: {config.STUDY_NAME_OPTUNA}")
logger.info(f"DATA_PATH: {config.DATA_PATH}")
logger.info(f"DB_PATH: {config.DB_PATH}")
logger.info(f"SEEDS: {getattr(config, 'SEEDS', None)} | SEED: {getattr(config, 'SEED', None)}")
logger.info(f"MES_TRAIN: {config.MES_TRAIN}")
logger.info(f"MES_VALIDACION: {config.MES_VALIDACION}")
logger.info(f"MES_TEST: {config.MES_TEST}")
logger.info(f"GANANCIA_ACIERTO: {config.GANANCIA_ACIERTO}")
logger.info(f"COSTO_ESTIMULO: {config.COSTO_ESTIMULO}")
logger.info(f"STORAGE_OPTUNA: {config.STORAGE_OPTUNA}")
logger.info(f"DIR_MODELS: {config.DIR_MODELS}")
logger.info(f"RUN_CALC_CURVAS: {config.RUN_CALC_CURVAS} | SUB_SAMPLE: {config.SUB_SAMPLE}")

# --- Import tardío del resto (evita leer config antes de tiempo) ---
from src.loader import (
    select_c02_polars, create_bq_table_c02, create_targets_c02,
    select_data_c02, tabla_productos_por_cliente
)
from src.features import (
    get_numeric_columns_pl, feature_engineering_lag, feature_engineering_delta, creation_lags, creation_deltas, select_data_lags_deltas
)
from src.optimization import (run_study, run_study_cv, create_seed)
from src.preprocessing import binary_target, split_train_data, create_binary_target_column
from src.train_test import train_model, calculo_curvas_ganancia, pred_ensamble_modelos


def main():
    logger.info("Iniciando Corrida")
    semillas_bay = create_seed(n_semillas=6)

    try:
        # ---------------------------------------------------------------------------------
        # 1. CREACIÓN/CARGA DE DATOS (START_POINT == 'DATA')
        # ---------------------------------------------------------------------------------
        if config.START_POINT == 'DATA':
            logger.info("Creando nueva base de datos...")

            # Selecciono datos crudos
            data = select_c02_polars(config.DATA_PATH)

            # Creo tabla en BQ a partir de datos Crudos
            create_bq_table_c02(data, config.BQ_PROJECT, config.BQ_DATASET, config.BQ_TABLE)

            # Creo targets
            create_targets_c02(config.BQ_PROJECT, config.BQ_DATASET, config.BQ_TABLE, config.BQ_TABLE_TARGETS)

            # Creo q_productos_cliente_mes
            tabla_productos_por_cliente(config.BQ_PROJECT, config.BQ_DATASET, config.BQ_TABLE, 'c02_q_productos')

            # ----------- Obtengo algunos datos para obtener tipos de columnas -------------
            data = select_data_c02(config.BQ_PROJECT, config.BQ_DATASET, 'c02_products', [202102])
            # Columnas a excluir
            exclude_cols = ["numero_de_cliente", "foto_mes", "clase_binaria1", "clase_binaria2", "clase_peso"]
            # Creo array con columnas numéricas
            numeric_cols = get_numeric_columns_pl(data, exclude_cols=exclude_cols)

            # Creo tabla con lags
            creation_lags(numeric_cols, 5)

            # Creo tabla con deltas
            creation_deltas(numeric_cols, 5)

            # Binarizando target
            logger.info("Binarizando target...")
            table_with_deltas = 'c02_delta'
            create_binary_target_column(config.BQ_PROJECT,config.BQ_DATASET,table_with_deltas)

        # Meses a usar
        meses = config.MES_TRAIN + config.MES_TEST + config.MES_PRED

        table_with_deltas = 'c02_delta'


        # Binarizar target


        # Selecciono los datos de los meses que se van a trabajar
        #data = select_data_c02(config.BQ_PROJECT, config.BQ_DATASET, table_with_deltas, meses)

        logger.info("Usando base de datos existente...")
        logger.info("Cargando dataset...")


        # ---------------------------------------------------------------------------------
        # 2. FEATURE ENGINEERING (START_POINT == 'FEATURES')
        # las features ya están creadas, acá selecciono las que quiero utilizar
        # ---------------------------------------------------------------------------------
        if config.START_POINT in ['FEATURES', 'OPTUNA', 'TRAIN', 'PREDICT']:
            logger.info("#### INICIO FEATURE ENGINEERING ###")
            logger.info("Creando Lags...")
            #numeric_cols = get_numeric_columns_pl(data, exclude_cols=["numero_de_cliente", "foto_mes","clase_binaria1","clase_binaria2",'clase_peso'])

            # ---- Creación de tabla con lags ----
            #creation_lags(meses, numeric_cols, 5)
            # SELECT LAGS - DELTAS
            #data = select_data_lags_deltas(k=4)

            #data = feature_engineering_lag(data, numeric_cols, cant_lag=3)
            #logger.info("Creando Deltas...")
            #data = feature_engineering_delta(data, numeric_cols, 3)
            logger.info("#### FIN FEATURE ENGINEERING ###")

        # ---------------------------------------------------------------------------------
        # 2.5. SPLITS POR MES
        # ---------------------------------------------------------------------------------
        meses_train_separados = {}
        for mes_train in config.MES_TRAIN:

            logger.info(f"Splitting data for mes {mes_train}...")
            # selecciono mes prediccion


            # paso mes predicción al select
            data = select_data_lags_deltas(table_with_deltas,mes_train,config.MES_TEST,config.MES_PRED,k=3)
            logger.info(f"Data shape: {data.shape}")
            logger.info(f"Inicio de split_train_data")
            resp = split_train_data(
                data, mes_train, config.MES_TEST, config.MES_PRED, getattr(config, "SEED", None), config.SUB_SAMPLE
            )
            logger.info(f"Fin de split_train_data")


            meses_train_separados[mes_train] = {
                'X_train': resp["X_train_pl"].to_pandas(),
                'y_train_binaria': resp["y_train_binaria"],
                'w_train': resp["w_train"],
                'y_test_class': resp["y_test_class"],
                'X_test': resp["X_test_pl"].to_pandas(),
                'X_pred': resp["X_pred_pl"].to_pandas(),
            }

        # ---------------------------------------------------------------------------------
        # 3. OPTIMIZACIÓN HIPERPARÁMETROS (por mes)
        # ---------------------------------------------------------------------------------
        storage_optuna = config.STORAGE_OPTUNA
        base_study_name = config.STUDY_NAME_OPTUNA
        studies_by_month = {}
        if config.START_POINT in ['OPTUNA', 'TRAIN', 'PREDICT']:
            logger.info(f"Seteando path de BBDD Optuna: {storage_optuna} - base_name={base_study_name}")
            logger.info("Iniciando estudios por mes...")

            # mantengo el cv en la bayesiana porque entrno por mes
            for mes, bundle in meses_train_separados.items():
                study_name = f"{base_study_name}_{mes}"
                study = run_study_cv(
                    X_train=bundle['X_train'],
                    y_train=bundle['y_train_binaria'],
                    #semillas = semillas_bay,
                    SEED=config.SEEDS[0],
                    w_train=bundle['w_train'],
                    matching_categorical_features=None,
                    storage_optuna=storage_optuna,
                    study_name_optuna=study_name,
                    optimizar=config.OPTIMIZAR,  # True: optimiza; False: sólo carga
                )
                studies_by_month[mes] = study
                logger.info(f"#### FIN OPTIMIZACIÓN HIPERPARÁMETROS MES {mes} ####")

        # ---------------------------------------------------------------------------------
        # 4. ENTRENAMIENTO Y CÁLCULO DE CURVAS (por mes)
        # ---------------------------------------------------------------------------------
        top_k_model = config.TOP_K_MODEL
        models_root = config.DIR_MODELS

        if config.START_POINT in ['TRAIN', 'PREDICT']:
            logger.info("Entrenando modelos Top-K y calculando curvas por mes...")

            if config.RUN_CALC_CURVAS:
                for mes, bundle in meses_train_separados.items():
                    study_name = f"{base_study_name}_{mes}"
                    study = studies_by_month.get(mes)
                    if study is None:
                        study = run_study_cv(
                            X_train=bundle['X_train'],
                            y_train=bundle['y_train_binaria'],
                            SEED=config.SEEDS[0],
                            w_train=bundle['w_train'],
                            matching_categorical_features=None,
                            storage_optuna=storage_optuna,
                            study_name_optuna=study_name,
                            optimizar=False,  # sólo cargar resultados
                        )
                        studies_by_month[mes] = study

                    logger.info(f"[{study_name}] Entrenando Top-{top_k_model}...")
                    _meta = train_model(
                        study=study,
                        X_train=bundle['X_train'],
                        y_train=bundle['y_train_binaria'],
                        weights=bundle['w_train'],
                        k=top_k_model,
                        base_study_name=base_study_name,  # <--- nuevo
                        mes=str(mes),
                        #experimento=study_name,
                        save_root=models_root,
                        seeds=config.SEEDS,
                        logger=logger,
                    )

                    logger.info(f"[{study_name}] Calculando curvas de ganancia...")
                    models_dir_mes = Path(models_root) / base_study_name / str(mes)
                    y_predicciones, curvas, mejores_cortes_normalizado = calculo_curvas_ganancia(
                        Xif=bundle['X_test'],
                        y_test_class=bundle['y_test_class'],
                        dir_model_opt=str(models_dir_mes),
                        experimento_key=study_name,
                        resumen_csv_name="resumen_ganancias.csv",
                    )
                    logger.info(f"[{study_name}] mejores cortes: {mejores_cortes_normalizado}")

        # ---------------------------------------------------------------------------------
        # 5. PREDICCIÓN FINAL / ENSEMBLE
        # ---------------------------------------------------------------------------------
        if config.START_POINT == 'PREDICT':
            mes_ref = max(meses_train_separados.keys())
            experimento = f"{base_study_name}_{mes_ref}"
            models_dir = Path(config.DIR_MODELS) / base_study_name / str(mes_ref)

            _df_pred = pred_ensamble_modelos(
                Xif=meses_train_separados[mes_ref]['X_pred'],
                dir_model_opt=str(models_dir),
                experimento=experimento,
                output_path=config.OUTPUT_PATH,
                resumen_csv_name="resumen_ganancias.csv",
                k=6
            )

    except Exception as e:
        logger.error(f"Se cortó ejecución por un error:\n {e}")

    logger.info("Fin Corrida")


if __name__ == "__main__":
    logger.info(f"Entrenando con SEED={getattr(config, 'SEED', None)} | TRAIN={config.MES_TRAIN} | VALID={config.MES_VALIDACION}")
    logger.info(f"Iniciando Pipeline desde el punto: {config.START_POINT}")
    main()
