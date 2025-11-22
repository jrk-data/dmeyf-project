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
logger.info(f"DATA_PATH_C02: {config.DATA_PATH_C02}")
logger.info(f"DATA_PATH_C03: {config.DATA_PATH_C03}")
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
from src.loader import ( load_gcs_to_bigquery_via_duckdb, create_churn_targets_bq,  consolidate_tables_bq,
    select_c03_polars, create_bq_table_c03,
    select_data_c03, tabla_productos_por_cliente
)
from src.features import (
    get_numeric_columns_pl, feature_engineering_lag, feature_engineering_delta, creation_lags, 
    create_intra_month_features_bq, create_historical_features_bq,
    creation_deltas, select_data_lags_deltas, 
)
from src.optimization import (run_study)
from src.create_seeds import create_seed
from src.preprocessing import binary_target, split_train_data, create_binary_target_column
from src.train_test import train_model, calculo_curvas_ganancia, pred_ensamble_modelos,pred_ensamble_desde_experimentos
from src.predict import prepare_prediction_dataframe
from src.reporting import generar_reporte_html_ganancia
from google.cloud import bigquery
import numpy as np


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
            PROJECT_ID = config.BQ_PROJECT
            DATASET_ID = config.BQ_DATASET  # Cambia por tu dataset
            TABLE_ID_02 = "c02"
            TABLE_ID_03 = "ult_mes"  # Nombre de la tabla de destino
            FINAL_TABLE_ID = "c03"

            try:
                logger.info(f"Ejecutando carga de datos desde GCS a BigQuery para tabla {TABLE_ID_02}...")
                logger.info("tarda unos 10 minutos..")

                # Creo tabla en BQ a partir de datos Crudos
                # load_gcs_to_bigquery_via_duckdb(
                #     project_id=PROJECT_ID,
                #     dataset_id=DATASET_ID,
                #     table_id=TABLE_ID_02
                # )
                logger.info(f'Tabla {TABLE_ID_02} creada con éxito. ')
            except Exception as e:
                logger.error(f"La ejecución falló: {e}.\n Asegúrate de tener credenciales válidas y de que la extensión httpfs esté instalada en DuckDB.")
                pass

            try:
                logger.info(f"Ejecutando carga de datos desde GCS a BigQuery para tabla {TABLE_ID_03}...")
                logger.info("tarda unos 10 minutos..")

                # Creo tabla en BQ a partir de datos Crudos
                load_gcs_to_bigquery_via_duckdb(
                    project_id=PROJECT_ID,
                    dataset_id=DATASET_ID,
                    table_id=TABLE_ID_03,
                    gcs_file_path=config.DATA_PATH_C03
                )
            except Exception as e:
                logger.error(f"La ejecución falló: {e}.\n Asegúrate de tener credenciales válidas y de que la extensión httpfs esté instalada en DuckDB.")
                pass

            try:

                COLUMNS_TO_FIX = {
                    "tmobile_app": "FLOAT64",
                    "cmobile_app_trx": "FLOAT64"
                }


                consolidate_tables_bq(
                    project_id=PROJECT_ID,
                    dataset_id=DATASET_ID,
                    final_table_id=FINAL_TABLE_ID,
                    source_table_ids=[TABLE_ID_02, TABLE_ID_03],
                    type_overrides=COLUMNS_TO_FIX
                )
            except Exception:
                pass

            #data = select_c02_polars(config.DATA_PATH)

            # Creo tabla en BQ a partir de datos Crudos
            #create_bq_table_c02(data, config.BQ_PROJECT, config.BQ_DATASET, config.BQ_TABLE)

            # Creo targets
            create_churn_targets_bq(config.BQ_PROJECT, config.BQ_DATASET, FINAL_TABLE_ID, "targets")
            #create_targets_c02(config.BQ_PROJECT, config.BQ_DATASET, config.BQ_TABLE, config.BQ_TABLE_TARGETS)

            # Creo q_productos_cliente_mes
            # Acá filtro los meses que no van a entrar
            tabla_productos_por_cliente(config.BQ_PROJECT, config.BQ_DATASET, config.BQ_TABLE, config.BQ_TABLE_TARGETS)
            #uso c03 y targets para joinear t crear c03_productos

            # ----------- Obtengo algunos datos para obtener tipos de columnas -------------
            data = select_data_c03([202102])
            # Columnas a excluir
            exclude_cols = ["numero_de_cliente", "foto_mes", "clase_binaria1", "clase_binaria2", "clase_peso"]

            # Crep tabla con FE intra mes
            create_intra_month_features_bq(config.BQ_PROJECT, config.BQ_DATASET, 'c03_products',
                                           config.BQ_TABLE_FEATURES)

            # CAMPOS QUE VAN A USARSE EN LAG Y DELTAS
            LAG_VARS = ['Master_cconsumos', 'Master_fultimo_cierre', 'Master_mconsumospesos',
                        'Master_mfinanciacion_limite',
                        'Master_mlimitecompra', 'Master_mpagominimo', 'Master_mpagospesos', 'Master_msaldopesos',
                        'Master_msaldototal',
                        'Visa_Fvencimiento', 'Visa_cconsumos', 'Visa_mconsumosdolares', 'Visa_mconsumospesos',
                        'Visa_mconsumototal', 'Visa_mfinanciacion_limite', 'Visa_mlimitecompra', 'Visa_mpagado',
                        'Visa_mpagominimo', 'Visa_mpagospesos', 'Visa_msaldopesos', 'Visa_msaldototal', 'Visa_status',
                        'ccaja_ahorro',
                        'ccaja_seguridad', 'ccajas_consultas', 'ccajas_otras', 'ccajas_transacciones',
                        'ccallcenter_transacciones',
                        'ccomisiones_mantenimiento', 'ccomisiones_otras', 'ccuenta_debitos_automaticos',
                        'cdescubierto_preacordado', 'cextraccion_autoservicio', 'chomebanking_transacciones',
                        'cmobile_app_trx', 'cpagomiscuentas',
                        'cpayroll_trx', 'cprestamos_personales', 'cproductos', 'ctarjeta_debito', 'ctarjeta_visa',
                        'ctarjeta_visa_debitos_automaticos',
                        'ctarjeta_visa_transacciones', 'ctransferencias_emitidas', 'ctransferencias_recibidas',
                        'ctrx_quarter', 'mactivos_margen', 'mautoservicio', 'mcaja_ahorro', 'mcaja_ahorro_dolares',
                        'mcomisiones', 'mcomisiones_mantenimiento', 'mcomisiones_otras', 'mcuenta_corriente',
                        'mcuenta_debitos_automaticos', 'mcuentas_saldo', 'mextraccion_autoservicio', 'mpagomiscuentas',
                        'mpasivos_margen',
                        'mpayroll', 'mplazo_fijo_dolares', 'mprestamos_personales', 'mrentabilidad',
                        'mrentabilidad_annual',
                        'mtarjeta_master_consumo', 'mtarjeta_visa_consumo', 'mtransferencias_emitidas',
                        'mtransferencias_recibidas', 'mttarjeta_visa_debitos_automaticos', 'tcallcenter',
                        'thomebanking', 'tmobile_app']

            # Listo aparte las features creadas
            features_nuevas = ["q_producto_master", "q_producto_visa", "q_producto_general", "ctrx_quarter_normalizado",
                               "mpayroll_sobre_edad"]

            # Agrego las features creadas
            LAG_VARS += features_nuevas

            # Dejo valores únicos
            LAG_VARS = list(set(LAG_VARS))

            create_historical_features_bq(
                config.BQ_PROJECT,
                config.BQ_DATASET,
                config.BQ_TABLE_FEATURES,
                config.BQ_TABLE_FEATURES_HISTORICAL,
                cols_to_engineer=LAG_VARS,
                window_size=6
            )

            # Creo tabla con lags
            # logger.info(f"Creando lags n= {config.NUN_WINDOW_LOAD}...")
            # creation_lags(tabla_features, numeric_cols, config.NUN_WINDOW_LOAD)
            # 
            # # Creo tabla con deltas
            # logger.info("Creando deltas...")
            # creation_deltas(numeric_cols, config.NUN_WINDOW_LOAD)

            # Binarizando target
            logger.info("Binarizando target...")
            table_delta_features_historical = config.BQ_TABLE_FEATURES_HISTORICAL
            create_binary_target_column(config.BQ_PROJECT,config.BQ_DATASET,table_delta_features_historical)

        # Selecciono los datos de los meses que se van a trabajar
        #data = select_data_c02(config.BQ_PROJECT, config.BQ_DATASET, table_delta_features_historical, meses)
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

            # Definir los meses a usar (Asumimos MES_TRAIN y MES_TEST ya son listas en config)
            MES_TRAIN_LIST = config.MES_TRAIN

            # Usaremos MES_VALIDACION como el conjunto de TEST para Optuna/Curvas, y lo convertimos a lista si no lo es
            if isinstance(config.MES_VALIDACION, (list, tuple, set)):
                MES_TEST_LIST = [int(m) for m in config.MES_VALIDACION]
            else:
                MES_TEST_LIST = [int(config.MES_VALIDACION)]

            if isinstance(config.MES_PRED, (list, tuple, set)):
                MES_PRED_LIST = [int(list(config.MES_PRED)[0])]
            else:
                MES_PRED_LIST = [int(config.MES_PRED)]

            # Nombre base del experimento
            EXPERIMENT_NAME = config.STUDY_NAME_OPTUNA

            # 1. Cargar todos los meses necesarios para el split
            meses_a_cargar = [int(m) for m in MES_TRAIN_LIST] + MES_TEST_LIST + MES_PRED_LIST

            table_delta_features_historical = config.BQ_TABLE_FEATURES_HISTORICAL

            logger.info(f"Cargando {len(meses_a_cargar)} meses para el split consolidado...")
            data = select_data_lags_deltas(
                table_delta_features_historical,
                config.MES_TEST,
                config.MES_PRED,
                meses_a_cargar,
                k=config.NUN_WINDOW
            )
            logger.info(f"Data shape cargada: {data.shape}")

            # 2. Hacer split único y consolidado
            FULL_SPLIT = split_train_data(
                data,
                MES_TRAIN=MES_TRAIN_LIST,  # Lista de meses a entrenar
                MES_TEST=MES_TEST_LIST,  # Lista de meses a testear (concatenados)
                MES_PRED=MES_PRED_LIST,
                SEED=config.SEED,
                SUB_SAMPLE=config.SUB_SAMPLE
            )
            logger.info(f"Split Consolidado listo. Train: {len(FULL_SPLIT['X_train_pl'])} filas.")

        # ---------------------------------------------------------------------------------
        # 3. OPTIMIZACIÓN HIPERPARÁMETROS (por mes)
        # ---------------------------------------------------------------------------------

            storage_optuna = config.STORAGE_OPTUNA
            studies_by_month = {}

            if config.START_POINT in ['OPTUNA', 'TRAIN']:
                logger.info("#### INICIO OPTIMIZACIÓN HIPERPARÁMETROS (Consolidado) ####")

                # Generar las 100 semillas para el Semillerío
                n_semillas_semillerio = 100
                semillas_semillerio = create_seed(n_semillas=n_semillas_semillerio)

                study_name = EXPERIMENT_NAME + "_CONSOLIDATED"

                study = run_study(
                    X_train=FULL_SPLIT['X_train_pl'].to_pandas(),
                    y_train=FULL_SPLIT['y_train_binaria'],
                    semillas=semillas_semillerio,  # PASAMOS LAS 100 SEMILLAS
                    SEED=config.SEEDS[0],
                    w_train=FULL_SPLIT['w_train'],
                    matching_categorical_features=None,
                    storage_optuna=storage_optuna,
                    study_name_optuna=study_name,
                    optimizar=config.OPTIMIZAR,
                )
                studies_by_month["CONSOLIDATED"] = study
                logger.info("#### FIN OPTIMIZACIÓN HIPERPARÁMETROS ####")
                # ---------------------------------------------------------------------------------
                # 4. ENTRENAMIENTO Y CÁLCULO DE CURVAS (Generación del Top-K Ensamblado)
                # ---------------------------------------------------------------------------------
                top_k_model = config.TOP_K_MODEL
                models_root = config.DIR_MODELS
                semillas_final_train = config.SEEDS

                if config.START_POINT in ['TRAIN', 'PREDICT']:
                    logger.info("Entrenando modelos Top-K y calculando curvas...")

                    if config.RUN_CALC_CURVAS:
                        study_name = EXPERIMENT_NAME + "_CONSOLIDATED"
                        study = studies_by_month.get("CONSOLIDATED")
                        # Cargar el study optimizado

                        # Lógica de carga si study es None (si START_POINT='TRAIN' directamente)
                        if study is None:
                            # Se necesita un split dummy para que run_study cargue (usamos el FULL_SPLIT cargado)
                            # Asumiendo que run_study puede cargar el study sin datos, si optimizar=False
                            study = run_study(
                                X_train=FULL_SPLIT['X_train_pl'].to_pandas(),
                                # Datos dummy para evitar error de DataFrame vacío
                                y_train=FULL_SPLIT['y_train_binaria'],
                                SEED=config.SEEDS[0],
                                w_train=FULL_SPLIT['w_train'],
                                storage_optuna=storage_optuna,
                                study_name_optuna=study_name,
                                optimizar=False,
                                semillas=[],
                            )
                            studies_by_month["CONSOLIDATED"] = study

                        logger.info(
                            f"[{study_name}] Entrenando Top-{top_k_model} (x{len(semillas_final_train)} semillas)...")

                        # 1. ENTRENAMIENTO DEL ENSEMBLE FINAL
                        train_model(
                            study=study,
                            X_train=FULL_SPLIT['X_train_pl'].to_pandas(),
                            y_train=FULL_SPLIT['y_train_binaria'],
                            weights=FULL_SPLIT['w_train'],
                            k=top_k_model,
                            base_study_name=config.STUDY_NAME_OPTUNA,
                            mes="CONSOLIDATED",  # Carpeta única
                            save_root=models_root,
                            seeds=semillas_final_train,
                            logger=logger,
                        )

                        # 2. CÁLCULO DE CURVAS DE GANANCIA SOBRE TEST CONCATENADO
                        logger.info(
                            f"[{study_name}] Calculando curvas de ganancia sobre TEST CONSOLIDADO (Meses: {MES_TEST_LIST})...")

                        models_dir_mes = Path(models_root) / config.STUDY_NAME_OPTUNA / "CONSOLIDATED"

                        # Los datos de test ya contienen los múltiples meses concatenados
                        _, _, mejores_cortes_normalizado = calculo_curvas_ganancia(
                            Xif=FULL_SPLIT['X_test_pl'].to_pandas(),
                            y_test_class=FULL_SPLIT['y_test_class'],
                            dir_model_opt=str(models_dir_mes),
                            experimento_key=study_name,
                            resumen_csv_name="resumen_ganancias.csv",
                        )
                        # La función print de logger debe estar fuera del cálculo del dictionary get para evitar errores si el dict falla
                        ganancia_max_promedio = mejores_cortes_normalizado.get('PROMEDIO', {}).get('ganancia', 'N/A')
                        logger.info(f"[{study_name}] Ganancia Máxima del Ensamblaje en Test: {ganancia_max_promedio}")

                # ---------------------------------------------------------------------------------
                # 5. PREDICCIÓN FINAL / ENSEMBLE
                # ---------------------------------------------------------------------------------
        if config.START_POINT == 'PREDICT':
            scenarios = getattr(config, "PREDICT_SCENARIOS", [])

            base_study_name = config.STUDY_NAME_OPTUNA
            # Caso simple: sin escenarios → usar último mes de train como experimento base
            if not scenarios:
                mes_ref = max(int(m) for m in config.MES_TRAIN)
                experimento = f"{base_study_name}_{mes_ref}"
                models_dir = Path(config.DIR_MODELS) / base_study_name / str(mes_ref)

                # Mes a predecir tomado de config.MES_PRED
                if isinstance(config.MES_PRED, (list, tuple, set)):
                    pred_s = int(list(config.MES_PRED)[0])
                else:
                    pred_s = int(config.MES_PRED)

                logger.info(f"[PREDICT] Escenario simple -> train_ref={mes_ref}, pred_month={pred_s}")

                X_pred = prepare_prediction_dataframe(
                    table_name=config.BQ_TABLE_FEATURES_HISTORICAL,
                    mes_pred=pred_s,
                    k=config.NUN_WINDOW
                )

                _ = pred_ensamble_modelos(
                    Xif=X_pred,
                    dir_model_opt=str(models_dir),
                    experimento=experimento,
                    output_path=config.OUTPUT_PATH,
                    resumen_csv_name="resumen_ganancias.csv",
                    k=config.TOP_K_MODEL
                )

            # Caso con escenarios definidos en config.PREDICT_SCENARIOS
            else:
                logger.info("Se recorren los escenarios de entrenamiento")
                for sc in scenarios:
                    logger.info(f"Escenario: {sc}")
                    name = sc["name"]
                    pred_month = sc["pred_month"]
                    groups = sc["train_for_pred"]

                    logger.info(f"[PREDICT] Ejecutando escenario: {name} -> pred_month={pred_month}")

                    pred_s = int(pred_month)

                    # 1) Construir X_pred del mes objetivo
                    X_pred = prepare_prediction_dataframe(
                        table_name=config.BQ_TABLE_FEATURES_HISTORICAL,
                        mes_pred=pred_s,
                        k=config.NUN_WINDOW
                    )

                    # 2) Armar lista de experimentos (carpetas + nombre de experimento)
                    exp_list = []
                    for g in groups:
                        for src_mes in g["use_experiments_from"]:
                            experimento = f"{base_study_name}_{src_mes}"
                            dir_model_opt = Path(config.DIR_MODELS) / base_study_name / str(src_mes)
                            exp_list.append({"dir": str(dir_model_opt), "experimento": experimento})

                    # 3) Ensamble multi-experimento
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
