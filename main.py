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
        if config.START_POINT == 'DATA' or config.CREAR_NUEVA_BASE:
            logger.info("Creando nueva base de datos...")

            # Selecciono datos crudos
            data = select_c02_polars(config.DATA_PATH)

            # Creo tabla en BQ a partir de datos Crudos
            create_bq_table_c02(data, config.BQ_PROJECT, config.BQ_DATASET, config.BQ_TABLE)

            # Creo targets
            create_targets_c02(config.BQ_PROJECT, config.BQ_DATASET, config.BQ_TABLE, config.BQ_TABLE_TARGETS)

            # Creo q_productos_cliente_mes
            # Acá filtro los meses que no van a entrar
            tabla_productos_por_cliente(config.BQ_PROJECT, config.BQ_DATASET, config.BQ_TABLE, 'c02_q_productos')

            # ----------- Obtengo algunos datos para obtener tipos de columnas -------------
            data = select_data_c02(config.BQ_PROJECT, config.BQ_DATASET, 'c02_products', [202102])
            # Columnas a excluir
            exclude_cols = ["numero_de_cliente", "foto_mes", "clase_binaria1", "clase_binaria2", "clase_peso"]
            # Creo array con columnas numéricas
            numeric_cols = get_numeric_columns_pl(data, exclude_cols=exclude_cols)

            # Creo tabla con lags
            logger.info(f"Creando lags n= {config.NUN_WINDOW_LOAD}...")
            creation_lags(numeric_cols, config.NUN_WINDOW_LOAD)

            # Creo tabla con deltas
            logger.info("Creando deltas...")
            creation_deltas(numeric_cols, config.NUN_WINDOW_LOAD)

            # Binarizando target
            logger.info("Binarizando target...")
            table_with_deltas = 'c02_delta'
            create_binary_target_column(config.BQ_PROJECT,config.BQ_DATASET,table_with_deltas)

        # Selecciono los datos de los meses que se van a trabajar
        #data = select_data_c02(config.BQ_PROJECT, config.BQ_DATASET, table_with_deltas, meses)
        else:
            logger.info("Usando base de datos existente...")


        # ---------------------------------------------------------------------------------
        # 2. FEATURE ENGINEERING (START_POINT == 'FEATURES')
        # las features ya están creadas, acá selecciono las que quiero utilizar
        # ---------------------------------------------------------------------------------
        if config.START_POINT in ['FEATURES', 'OPTUNA', 'TRAIN', 'PREDICT']:
            logger.info("#### INICIO FEATURE ENGINEERING ###")
            logger.info("Creando Lags...")

            logger.info("#### FIN FEATURE ENGINEERING ###")

        # ---------------------------------------------------------------------------------
        # 2.5. SPLITS POR MES
        # ---------------------------------------------------------------------------------

        def _as_list(x):
            return x if isinstance(x, (list, tuple, set)) else [x]

        meses_train_separados = {}
        for mes_train in config.MES_TRAIN:
            logger.info(f"Splitting data for mes {mes_train}...")
            # selecciono mes prediccion
            mes_test_cfg = config.TEST_BY_TRAIN.get(mes_train, None)
            mes_test = mes_test_cfg if mes_test_cfg is not None else config.MES_TEST

            # 2) SCALARS para SELECT
            mes_train_s = int(mes_train)
            mes_test_s = int(mes_test[0] if isinstance(mes_test, (list, tuple, set)) else mes_test)
            mes_pred_s = int(config.MES_PRED[0] if isinstance(config.MES_PRED, (list, tuple, set)) else config.MES_PRED)

            if mes_test is None:
                mes_test = config.MES_TEST  # mantiene compatibilidad (puede ser lista o int)

            table_with_deltas = 'c02_delta'
            # paso mes predicción al select
            data = select_data_lags_deltas(
                table_with_deltas,
                mes_train_s,  # SCALAR
                mes_test_s,  # SCALAR
                mes_pred_s,
                k=config.NUN_WINDOW
            )
            logger.info(f"Data shape: {data.shape}")
            logger.info(f"Inicio de split_train_data")
            resp = split_train_data(
                data,
                MES_TRAIN=[mes_train_s],
                MES_TEST=[mes_test_s],
                MES_PRED=[mes_pred_s],
                SEED=config.SEED,
                SUB_SAMPLE=config.SUB_SAMPLE
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
            scenarios = getattr(config, "PREDICT_SCENARIOS", [])
            if not scenarios:

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
            else:
                logger.info("Se recorren los escenarios de entrenamiento")
                for sc in scenarios:
                    logger.info(f"Escenario: {sc}")
                    name = sc["name"]
                    pred_month = sc["pred_month"]
                    groups = sc["train_for_pred"]

                    logger.info(f"[PREDICT] Ejecutando escenario: {name} -> pred_month={pred_month}")

                    # 1) Construir X_pred del mes objetivo (SCALARS para SELECT, LISTAS para SPLIT)
                    pred_s = int(pred_month)
                    table_with_deltas = 'c02_delta'

                    data_pred = select_data_lags_deltas(
                        table_with_deltas,
                        pred_s,  # mes_train (dummy)
                        pred_s,  # mes_test  (dummy)
                        pred_s,  # mes_pred  (el que importa)
                        k=config.NUN_WINDOW
                    )

                    resp_pred = split_train_data(
                        data_pred,
                        MES_TRAIN=[pred_s],
                        MES_TEST=[pred_s],
                        MES_PRED=[pred_s],
                        SEED=config.SEED,
                        SUB_SAMPLE=config.SUB_SAMPLE
                    )

                    X_pred = resp_pred["X_pred_pl"].to_pandas()

                    # 2) Armar lista de experimentos (carpetas + nombre de experimento)
                    exp_list = []
                    for g in groups:
                        for src_mes in g["use_experiments_from"]:
                            experimento = f"{base_study_name}_{src_mes}"
                            dir_model_opt = Path(config.DIR_MODELS) / base_study_name / str(src_mes)
                            exp_list.append({"dir": str(dir_model_opt), "experimento": experimento})

                    # 3) Ensamble multi-experimento
                    from src.train_test import pred_ensamble_desde_experimentos
                    _ = pred_ensamble_desde_experimentos(
                        Xif=X_pred,
                        experiments=exp_list,
                        k=config.TOP_K_MODEL,
                        output_path=config.OUTPUT_PATH,
                        output_basename=f"{name}_{pred_month}",
                        resumen_csv_name="resumen_ganancias.csv"
                    )



    except Exception as e:
        logger.error(f"Se cortó ejecución por un error:\n {e}")

    logger.info("Fin Corrida")


if __name__ == "__main__":
    logger.info(f"Entrenando con SEED={getattr(config, 'SEED', None)} | TRAIN={config.MES_TRAIN} | VALID={config.MES_VALIDACION}")
    logger.info(f"Iniciando Pipeline desde el punto: {config.START_POINT}")
    main()
