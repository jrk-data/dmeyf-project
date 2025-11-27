# === Bootstrap mínimo de logging a consola (no depende de config) ===
import argparse
import sys
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
import gc

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
logger.info(f"START_POINT: {config.START_POINT}")
logger.info(f"STUDY_NAME: {config.STUDY_NAME_OPTUNA}")

# --- Import tardío del resto (evita leer config antes de tiempo) ---
from src.loader import (load_gcs_to_bigquery_via_duckdb, create_churn_targets_bq, consolidate_tables_bq,
                        create_bq_table_c03, select_data_c03, tabla_productos_por_cliente
                        )
from src.features import (
    create_intra_month_features_bq, create_historical_features_bq,
    select_data_lags_deltas
)
from src.optimization import (run_study)
from src.create_seeds import create_seed
from src.preprocessing import split_train_data, create_binary_target_column
from src.train_test import train_model, calculo_curvas_ganancia, pred_ensamble_modelos, pred_ensamble_desde_experimentos
from src.predict import prepare_prediction_dataframe
from src.feature_selection import perform_canaritos_selection  # <--- IMPORTANTE: Nuevo módulo
from google.cloud import bigquery
import polars as pl


def main():
    logger.info("Iniciando Corrida")

    try:
        # ---------------------------------------------------------------------------------
        # 1. CREACIÓN/CARGA DE DATOS (START_POINT == 'DATA')
        # ---------------------------------------------------------------------------------
        if config.START_POINT == 'DATA' or config.CREAR_NUEVA_BASE:
            logger.info("Creando nueva base de datos...")

            # ... (Lógica de creación de tablas igual que antes) ...
            PROJECT_ID = config.BQ_PROJECT
            DATASET_ID = config.BQ_DATASET
            FINAL_TABLE_ID = "c03"

            # Carga C03
            try:
                load_gcs_to_bigquery_via_duckdb(
                    project_id=PROJECT_ID, dataset_id=DATASET_ID, table_id="ult_mes",
                    gcs_file_path=config.DATA_PATH_C03
                )
            except Exception:
                pass

            # Consolidación (si aplica)
            try:
                COLUMNS_TO_FIX = {"tmobile_app": "FLOAT64", "cmobile_app_trx": "FLOAT64"}
                consolidate_tables_bq(
                    project_id=PROJECT_ID, dataset_id=DATASET_ID,
                    final_table_id=FINAL_TABLE_ID,
                    source_table_ids=["c02", "ult_mes"],
                    type_overrides=COLUMNS_TO_FIX
                )
            except Exception:
                pass

            # Creación de Features en BQ
            create_churn_targets_bq(config.BQ_PROJECT, config.BQ_DATASET, FINAL_TABLE_ID, "targets")
            tabla_productos_por_cliente(config.BQ_PROJECT, config.BQ_DATASET, config.BQ_TABLE, "targets")

            create_intra_month_features_bq(config.BQ_PROJECT, config.BQ_DATASET, 'c03_products',
                                           config.BQ_TABLE_FEATURES)

            # Definición de Features para Historial
            # ... (Lista LAG_VARS igual que tu código original) ...
            LAG_VARS = ['Master_cconsumos', 'Master_fultimo_cierre', 'Master_mconsumospesos',
                        'Master_mfinanciacion_limite', 'Master_mlimitecompra', 'Master_mpagominimo',
                        'Master_mpagospesos', 'Master_msaldopesos', 'Master_msaldototal', 'Visa_Fvencimiento',
                        'Visa_cconsumos', 'Visa_mconsumosdolares', 'Visa_mconsumospesos', 'Visa_mconsumototal',
                        'Visa_mfinanciacion_limite', 'Visa_mlimitecompra', 'Visa_mpagado', 'Visa_mpagominimo',
                        'Visa_mpagospesos', 'Visa_msaldopesos', 'Visa_msaldototal', 'Visa_status', 'ccaja_ahorro',
                        'ccaja_seguridad', 'ccajas_consultas', 'ccajas_otras', 'ccajas_transacciones',
                        'ccallcenter_transacciones', 'ccomisiones_mantenimiento', 'ccomisiones_otras',
                        'ccuenta_debitos_automaticos', 'cdescubierto_preacordado', 'cextraccion_autoservicio',
                        'chomebanking_transacciones', 'cmobile_app_trx', 'cpagomiscuentas', 'cpayroll_trx',
                        'cprestamos_personales', 'cproductos', 'ctarjeta_debito', 'ctarjeta_visa',
                        'ctarjeta_visa_debitos_automaticos', 'ctarjeta_visa_transacciones', 'ctransferencias_emitidas',
                        'ctransferencias_recibidas', 'ctrx_quarter', 'mactivos_margen', 'mautoservicio', 'mcaja_ahorro',
                        'mcaja_ahorro_dolares', 'mcomisiones', 'mcomisiones_mantenimiento', 'mcomisiones_otras',
                        'mcuenta_corriente', 'mcuenta_debitos_automaticos', 'mcuentas_saldo',
                        'mextraccion_autoservicio', 'mpagomiscuentas', 'mpasivos_margen', 'mpayroll',
                        'mplazo_fijo_dolares', 'mprestamos_personales', 'mrentabilidad', 'mrentabilidad_annual',
                        'mtarjeta_master_consumo', 'mtarjeta_visa_consumo', 'mtransferencias_emitidas',
                        'mtransferencias_recibidas', 'mttarjeta_visa_debitos_automaticos', 'tcallcenter',
                        'thomebanking', 'tmobile_app']
            features_nuevas = ["q_producto_master", "q_producto_visa", "q_producto_general", "ctrx_quarter_normalizado",
                               "mpayroll_sobre_edad"]
            LAG_VARS += features_nuevas
            LAG_VARS = list(set(LAG_VARS))

            create_historical_features_bq(
                config.BQ_PROJECT, config.BQ_DATASET,
                config.BQ_TABLE_FEATURES, config.BQ_TABLE_FEATURES_HISTORICAL,
                cols_to_engineer=LAG_VARS, window_size=6
            )

            # Binarizar target
            create_binary_target_column(config.BQ_PROJECT, config.BQ_DATASET, config.BQ_TABLE_FEATURES_HISTORICAL)

        # ---------------------------------------------------------------------------------
        # 2. SELECCIÓN DE VARIABLES (START_POINT == 'SELECTION') - NUEVO
        # ---------------------------------------------------------------------------------
        if config.START_POINT == 'SELECTION':
            logger.info("#### INICIO FEATURE SELECTION (CANARITOS) ####")

            # A. Meses definidos para esta etapa
            MESES_SELECCION = [202101, 202102, 202103]
            UNDERSAMPLING_CANARITOS = 0.1

            logger.info(f"Cargando meses {MESES_SELECCION} para selección de variables...")

            df_pl = select_data_lags_deltas(
                config.BQ_TABLE_FEATURES_HISTORICAL,
                MESES_SELECCION,
                k=config.NUN_WINDOW
            )

            # B. Usar split_train_data para hacer el Undersampling (0.1)
            logger.info(f"Aplicando split y undersampling ({UNDERSAMPLING_CANARITOS})...")

            split_data = split_train_data(
                data=df_pl,
                MES_TRAIN=MESES_SELECCION,
                MES_TEST=[],
                MES_PRED=[],
                SEED=config.SEED,
                SUB_SAMPLE=UNDERSAMPLING_CANARITOS
            )

            # C. Ejecutar lógica de Canaritos
            selected_features = perform_canaritos_selection(
                X_train=split_data['X_train_pl'],
                y_train=split_data['y_train_binaria'],
                n_canaritos=20,
                seed=config.SEED
            )

            # D. Guardar resultados (Lógica Exclusiva GCS / Bucket)
            output_file = config.PATH_FEATURES_SELECTION
            logger.info(f"Intentando subir features seleccionadas a: {output_file}")

            try:
                # Importación local para asegurar que la librería esté disponible
                from google.cloud import storage

                # 1. Validación de formato gs://
                if not output_file.startswith("gs://"):
                    raise ValueError(
                        f"La ruta configurada debe ser un bucket (empezar con 'gs://'). Valor actual: {output_file}")

                # 2. Parseo de la ruta: gs://bucket_name/path/to/file.txt
                parts = output_file.replace("gs://", "").split("/", 1)
                if len(parts) < 2:
                    raise ValueError("La ruta del bucket parece incompleta. Debe ser: gs://bucket/carpeta/archivo.txt")

                bucket_name = parts[0]
                blob_name = parts[1]

                # 3. Conexión y subida
                client = storage.Client()  # Toma credenciales de tu entorno (gcloud auth o VM)
                bucket = client.bucket(bucket_name)
                blob = bucket.blob(blob_name)

                # Convertir lista a string
                content_str = "\n".join(selected_features)

                # Subir contenido
                blob.upload_from_string(content_str)

                logger.info(f"✅ Lista guardada exitosamente en el Bucket: {output_file}")
                logger.info(
                    "⚠️ RECUERDA: Copia el contenido de este archivo a 'PSI_2021_FEATURES' en tu config.yaml antes de correr OPTUNA.")

            except Exception as e:
                logger.error(f"❌ Error crítico al guardar en Bucket: {e}")
                raise  # Detenemos la ejecución porque si no se guarda, no sirve seguir.

            # Cortamos ejecución aquí
            logger.info("Fin de proceso SELECTION.")
            return
        # ---------------------------------------------------------------------------------
        # 3. SPLIT POR MES PARA OPTUNA / TRAIN
        # ---------------------------------------------------------------------------------

        # Variable para almacenar el split consolidado
        FULL_SPLIT = None

        if config.START_POINT in ['OPTUNA', 'TRAIN']:
            logger.info("#### PREPARANDO DATOS PARA OPTIMIZACIÓN/ENTRENAMIENTO ####")

            # Definir meses de Test/Pred según config
            if isinstance(config.MES_VALIDACION, (list, tuple, set)):
                MES_TEST_LIST = [int(m) for m in config.MES_VALIDACION]
            else:
                MES_TEST_LIST = [int(config.MES_VALIDACION)]

            if isinstance(config.MES_PRED, (list, tuple, set)):
                MES_PRED_LIST = [int(list(config.MES_PRED)[0])]
            else:
                MES_PRED_LIST = [int(config.MES_PRED)]

            if config.CARGAR_HISTORIA_COMPLETA:
                logger.info("🚀 MODO PRODUCCIÓN: Cargando histórico extendido desde 2019...")
                # Generamos rango dinámico: Desde 201901 hasta el último mes necesario
                all_months_needed = MES_TEST_LIST + MES_PRED_LIST
                max_month = max(all_months_needed) if all_months_needed else 202106

                # Meses desde 201901 hasta max_month
                meses_a_cargar = [m for m in range(201901, max_month + 1) if m % 100 in range(1, 13)]
            else:
                logger.info("🧪 MODO PRUEBA: Cargando estrictamente los meses configurados...")
                # Solo carga lo que pusiste en el YAML (Train + Test + Pred)
                meses_a_cargar = config.MES_TRAIN + MES_TEST_LIST + MES_PRED_LIST

                # Evitar duplicados y cargar
            meses_a_cargar = list(set(meses_a_cargar))
            logger.info(f"Meses a cargar desde BigQuery: {meses_a_cargar}")

            data_full = select_data_lags_deltas(
                config.BQ_TABLE_FEATURES_HISTORICAL,
                meses_a_cargar,
                k=config.NUN_WINDOW
            )

            logger.info("Aplicando Filtros de Negocio:")
            logger.info("1. BAJA+1 / BAJA+2 >= 201901")
            logger.info(f"2. CONTINUA >= 202001 (se aplicará subsample {config.SUB_SAMPLE} luego)")

            # Filtro Polars
            data_filtered = data_full.filter(
                # Registros que son Test o Predicción (pasan directo)
                (pl.col("foto_mes").is_in(MES_TEST_LIST + MES_PRED_LIST))
                |
                # Registros de Entrenamiento: Bajas desde 2019
                ((pl.col("clase_ternaria").is_in(["BAJA+1", "BAJA+2"])) & (pl.col("foto_mes") >= 201901))
                |
                # Registros de Entrenamiento: Continuas solo desde 2020
                ((pl.col("clase_ternaria") == "CONTINUA") & (pl.col("foto_mes") >= 202001))
            )

            logger.info(f"Registros tras filtro de fechas: {data_filtered.height}")

            # --- LIMPIEZA 1: Ya no necesitas data_full ---
            del data_full
            gc.collect()
            logger.info("Memoria liberada: data_full eliminada.")
            # ---------------------------------------------

            # Aplicar Split y Undersampling (0.5)
            logger.info(f"Ejecutando Split con Undersampling {config.SUB_SAMPLE}...")

            FULL_SPLIT = split_train_data(
                data=data_filtered,
                MES_TRAIN=config.MES_TRAIN,  # Lista completa de meses de train
                MES_TEST=MES_TEST_LIST,
                MES_PRED=MES_PRED_LIST,
                SEED=config.SEED,
                SUB_SAMPLE= config.SUB_SAMPLE # <--- Undersampling del 0.5 a los CONTINUA (que ya son >202001)
            )

            logger.info(f"Split Consolidado listo. Train size: {len(FULL_SPLIT['X_train_pl'])}")

            # --- LIMPIEZA 2: Ya no necesitas data_filtered ---
            # FULL_SPLIT ya tiene copias o vistas de los datos necesarios
            del data_filtered
            gc.collect()
            logger.info("Memoria liberada: data_filtered eliminada.")
            # -------------------------------------------------


        # ---------------------------------------------------------------------------------
        # 4. OPTIMIZACIÓN HIPERPARÁMETROS
        # ---------------------------------------------------------------------------------
        storage_optuna = config.STORAGE_OPTUNA
        studies_by_month = {}

        if config.START_POINT in ['OPTUNA', 'TRAIN']:
            logger.info("#### INICIO OPTIMIZACIÓN HIPERPARÁMETROS (Consolidado) ####")

            # Generar semillas
            semillas_semillerio = create_seed(n_semillas=100)
            study_name = config.STUDY_NAME_OPTUNA + "_CONSOLIDATED"

            study = run_study(
                X_train=FULL_SPLIT['X_train_pl'].to_pandas(),
                y_train=FULL_SPLIT['y_train_binaria'].reshape(-1, 1),  # Reshape necesario para LGBM Dataset
                semillas=semillas_semillerio,
                SEED=config.SEEDS[0],
                w_train=FULL_SPLIT['w_train'].reshape(-1, 1),
                matching_categorical_features=None,
                storage_optuna=storage_optuna,
                study_name_optuna=study_name,
                optimizar=config.OPTIMIZAR,
            )
            studies_by_month["CONSOLIDATED"] = study



        # ---------------------------------------------------------------------------------
        # 5. ENTRENAMIENTO Y CURVAS
        # ---------------------------------------------------------------------------------
        if config.START_POINT == 'TRAIN':
            logger.info("Entrenando modelos Top-K y calculando curvas...")

            study = studies_by_month.get("CONSOLIDATED")
            top_k_model = config.TOP_K_MODEL
            semillas_final_train = config.SEEDS
            models_root = config.DIR_MODELS

            # 1. ENTRENAMIENTO
            train_model(
                study=study,
                X_train=FULL_SPLIT['X_train_pl'].to_pandas(),
                y_train=FULL_SPLIT['y_train_binaria'],  # train_model espera array plano
                weights=FULL_SPLIT['w_train'],
                k=top_k_model,
                base_study_name=config.STUDY_NAME_OPTUNA,
                mes="CONSOLIDATED",
                save_root=models_root,
                seeds=semillas_final_train,
                logger=logger,
            )

            # 2. CURVAS
            if config.RUN_CALC_CURVAS:
                logger.info("Calculando curvas de ganancia...")
                models_dir_mes = Path(models_root) / config.STUDY_NAME_OPTUNA / config.CONSOLIDATED_PATH

                calculo_curvas_ganancia(
                    Xif=FULL_SPLIT['X_test_pl'].to_pandas(),
                    y_test_class=FULL_SPLIT['y_test_class'],
                    dir_model_opt=str(models_dir_mes),
                    experimento_key=study.study_name,
                    resumen_csv_name="resumen_ganancias.csv",
                )

        # ---------------------------------------------------------------------------------
        # 6. PREDICCIÓN FINAL
        # ---------------------------------------------------------------------------------
        if config.START_POINT == 'PREDICT':
            # ... (Lógica de predicción se mantiene igual, usando features guardadas en modelo) ...
            scenarios = getattr(config, "PREDICT_SCENARIOS", [])
            base_study_name = config.STUDY_NAME_OPTUNA

            if not scenarios:
                # Caso simple
                if isinstance(config.MES_PRED, (list, tuple, set)):
                    pred_s = int(list(config.MES_PRED)[0])
                else:
                    pred_s = int(config.MES_PRED)

                mes_ref = "CONSOLIDATED"  # Ajuste para apuntar a la carpeta consolidada
                models_dir = Path(config.DIR_MODELS) / base_study_name / mes_ref
                experimento = f"{base_study_name}_{mes_ref}"

                logger.info(f"[PREDICT] Escenario simple -> Folder={models_dir}, pred_month={pred_s}")

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
            else:
                # Caso con escenarios
                for sc in scenarios:
                    name = sc["name"]
                    pred_month = sc["pred_month"]
                    groups = sc["train_for_pred"]  # Espera [{"use_experiments_from": ["CONSOLIDATED"]}]

                    X_pred = prepare_prediction_dataframe(
                        table_name=config.BQ_TABLE_FEATURES_HISTORICAL,
                        mes_pred=int(pred_month),
                        k=config.NUN_WINDOW
                    )

                    exp_list = []
                    for g in groups:
                        for src_mes in g["use_experiments_from"]:
                            # Asumiendo src_mes = "CONSOLIDATED"
                            experimento = f"{base_study_name}_{src_mes}"
                            dir_model_opt = Path(config.DIR_MODELS) / base_study_name / str(src_mes)
                            exp_list.append({"dir": str(dir_model_opt), "experimento": experimento})

                    _ = pred_ensamble_desde_experimentos(
                        Xif=X_pred,
                        experiments=exp_list,
                        k=config.TOP_K_MODEL,
                        output_path=config.OUTPUT_PATH,
                        output_basename=f"{name}_{pred_month}",
                        resumen_csv_name="resumen_ganancias.csv"
                    )

    except Exception as e:
        logger.error(f"Se cortó ejecución por un error:\n {e}", exc_info=True)

    logger.info("Fin Corrida")


if __name__ == "__main__":
    logger.info(f"Entrenando con SEED={getattr(config, 'SEED', None)}")
    logger.info(f"Iniciando Pipeline desde: {config.START_POINT}")
    main()