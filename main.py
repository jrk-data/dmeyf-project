# === bootstrap logging lo PRIMERO ===
import os, sys, logging
from datetime import datetime
from pathlib import Path


######################################################################

from src.loader import (
                        select_c02_polars, create_bq_table_c02,create_targets_c02,
                        select_data_c02, tabla_productos_por_cliente)
from src.features import (get_numeric_columns_pl,
                          feature_engineering_lag,
                          feature_engineering_delta)
from src.optimization import (binary_target,
                              split_train_data,
                              run_study
                              )
from src.config import (CREAR_NUEVA_BASE, DATA_PATH, LOGS_PATH, OUTPUT_PATH
                        , SEEDS, MES_TRAIN,
                        MES_VALIDACION, MES_TEST,
                        GANANCIA_ACIERTO, COSTO_ESTIMULO, DB_PATH,
                        STUDY_NAME_OPTUNA, STORAGE_OPTUNA, OPTIMIZAR
                        , DIR_MODELS, MES_PRED, START_POINT, RUN_CALC_CURVAS,
                        # Variables BigQuery
                        BQ_PROJECT, BQ_DATASET, BQ_TABLE, BQ_TABLE_TARGETS, TOP_K_MODEL)

from src.train_test import (train_model
                            , calculo_curvas_ganancia
                            ,pred_ensamble_modelos)


# ---- INSTANCIO LOS LOGS PARA REGISTRAR CUALQUIER ERROR DE IMPORT

# Elegí un path ABSOLUTO para que no dependa del cwd
LOGS_PATH = Path(LOGS_PATH)
LOGS_PATH.mkdir(parents=True, exist_ok=True)

name_log = f"log_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
fmt = "%(asctime)s - %(name)s - %(levelname)s - L%(lineno)d - %(message)s"

file_handler = logging.FileHandler(LOGS_PATH / name_log, mode="w", encoding="utf-8")
stream_handler = logging.StreamHandler(sys.stdout)
file_handler.setFormatter(logging.Formatter(fmt))
stream_handler.setFormatter(logging.Formatter(fmt))

logging.basicConfig(level=logging.INFO, handlers=[file_handler, stream_handler], force=True)
logger = logging.getLogger(__name__)
logger.info("Logger inicializado")


# FLAG para crear BBDD de cero o no
CREAR_NUEVA_BASE = CREAR_NUEVA_BASE
# =====================

logger = logging.getLogger(__name__)

### Manejo de Configuración en YAML ###
logger.info("Configuración cargada desde YAML")
logger.info(f"CREAR_NUEVA_BASE: {CREAR_NUEVA_BASE}")
logger.info(f"STUDY_NAME: {STUDY_NAME_OPTUNA}")
logger.info(f"DATA_PATH: {DATA_PATH}")
logger.info(f"DB_PATH: {DB_PATH}")
logger.info(f"SEMILLA: {SEEDS[0]}")
logger.info(f"MES_TRAIN: {MES_TRAIN}")
logger.info(f"MES_VALIDACION: {MES_VALIDACION}")
logger.info(f"MES_TEST: {MES_TEST}")
logger.info(f"GANANCIA_ACIERTO: {GANANCIA_ACIERTO}")
logger.info(f"COSTO_ESTIMULO: {COSTO_ESTIMULO}")


# === FLAGS DE CONTROL ===
RUN_CALC_CURVAS = RUN_CALC_CURVAS    # ⬅️ poner False para no recalcular curvas
# =========================




def main():
    logger.info(f"Entrenando con SEMILLA={SEEDS[0]}, TRAIN={MES_TRAIN}, VALID={MES_VALIDACION}")
    logger.info(f"Iniciando Pipeline desde el punto: {START_POINT}")

    try:
        # ---------------------------------------------------------------------------------
        # 1. CREACIÓN/CARGA DE DATOS (START_POINT == 'DATA')
        # ---------------------------------------------------------------------------------
        if START_POINT == 'DATA':
            logger.info("Creando nueva base de datos...")
            # NOTA: Usamos CREAR_NUEVA_BASE aquí si es necesario, aunque START_POINT es más limpio.

            # Selecciono datos crudos
            data = select_c02_polars(DATA_PATH)

            # Creo tabla en BQ a partir de datos Crudos
            create_bq_table_c02(data, BQ_PROJECT, BQ_DATASET , BQ_TABLE)

            # Creo targets
            create_targets_c02(BQ_PROJECT, BQ_DATASET , BQ_TABLE, BQ_TABLE_TARGETS)

            # Creo q_productos_cliente_mes
            tabla_productos_por_cliente(BQ_PROJECT, BQ_DATASET , BQ_TABLE, BQ_TABLE_TARGETS)

        # Meses a usar
        meses = MES_TRAIN + MES_TEST + MES_PRED

        # Selecciono los datos de los meses que se van a trabajar
        data = select_data_c02(BQ_PROJECT, BQ_DATASET , BQ_TABLE, meses)



        logger.info("Usando base de datos existente...")
        # Cargo dataset base
        logger.info("Cargando dataset...")
        #data = select_c01(DB_PATH)  # <-- 'data' es necesario para los siguientes pasos

        # Binarizar target
        logger.info("Binarizando target...")
        data = binary_target(data)

        # ---------------------------------------------------------------------------------
        # 2. FEATURE ENGINEERING (START_POINT == 'FEATURES')
        # ---------------------------------------------------------------------------------
        if START_POINT in ['FEATURES', 'OPTUNA', 'TRAIN', 'PREDICT']:
            logger.info(f"#### INICIO FEATURE ENGINEERING ###")
            logger.info("Creando Lags...")
            numeric_cols = get_numeric_columns_pl(data, exclude_cols=["numero_de_cliente", "foto_mes"])
            data = feature_engineering_lag(
                data,
                numeric_cols,
                cant_lag=3,
            )

            logger.info("Creando Deltas...")
            data = feature_engineering_delta(data, numeric_cols, 3)
            logger.info(f"Data shape: {data.shape}")
            logger.info(f"#### FIN FEATURE ENGINEERING ###")

        # ---------------------------------------------------------------------------------
        # 2.5. SPLITS POR MES
        # ---------------------------------------------------------------------------------
        meses_train_separados = {}
        for mes_train in MES_TRAIN:
            resp = split_train_data(data, mes_train, MES_TEST, MES_PRED)
            meses_train_separados[mes_train] = {
                'X_train': resp["X_train_pl"].to_pandas(),
                'y_train_b2': resp["y_train_binaria2"],
                'w_train': resp["w_train"],
                'y_test_class': resp["y_test_class"],
                'X_test': resp["X_test_pl"].to_pandas(),
                'X_pred': resp["X_pred_pl"].to_pandas(),
            }

        # ---------------------------------------------------------------------------------
        # 3. OPTIMIZACIÓN HIPERPARÁMETROS (por mes)
        # ---------------------------------------------------------------------------------
        storage_optuna = STORAGE_OPTUNA
        base_study_name = STUDY_NAME_OPTUNA
        studies_by_month = {}
        if START_POINT in ['OPTUNA', 'TRAIN', 'PREDICT']:
            logger.info(f"Seteando path de BBDD Optuna: {storage_optuna} - base_name={base_study_name}")
            logger.info("Iniciando estudios por mes...")

            for mes, bundle in meses_train_separados.items():
                study_name = f"{base_study_name}_{mes}"
                study = run_study(
                    X_train=bundle['X_train'],
                    y_train=bundle['y_train_b2'],
                    SEED=SEEDS[0],
                    w_train=bundle['w_train'],
                    matching_categorical_features=None,
                    storage_optuna=storage_optuna,
                    study_name_optuna=study_name,
                    optimizar=OPTIMIZAR,  # True: optimiza; False: sólo carga
                )
                studies_by_month[mes] = study
                logger.info(f"#### FIN OPTIMIZACIÓN HIPERPARÁMETROS MES {mes} ####")

        # ---------------------------------------------------------------------------------
        # 4. ENTRENAMIENTO Y CÁLCULO DE CURVAS (por mes)
        # ---------------------------------------------------------------------------------
        top_k_model = TOP_K_MODEL # El top de k de modelos que vamos a elegir
        models_root = DIR_MODELS  # raíz de modelos en config

        if START_POINT in ['TRAIN', 'PREDICT']:
            logger.info("Entrenando modelos Top-K y calculando curvas por mes...")

            if RUN_CALC_CURVAS:
                for mes, bundle in meses_train_separados.items():
                    # asegurar tener el study (si no corriste OPTUNA ahora, se carga desde storage)
                    study_name = f"{base_study_name}_{mes}"
                    study = studies_by_month.get(mes)
                    if study is None:
                        study = run_study(
                            X_train=bundle['X_train'],
                            y_train=bundle['y_train_b2'],
                            SEED=SEEDS[0],
                            w_train=bundle['w_train'],
                            matching_categorical_features=None,
                            storage_optuna=storage_optuna,
                            study_name_optuna=study_name,
                            optimizar=False,  # sólo cargar resultados
                        )
                        studies_by_month[mes] = study

                    # ---------- ENTRENAR Top-K por mes ----------
                    logger.info(f"[{study_name}] Entrenando Top-{top_k_model}...")
                    meta = train_model(
                        study=study,
                        X_train=bundle['X_train'],
                        y_train=bundle['y_train_b2'],
                        weights=bundle['w_train'],
                        k=top_k_model,
                        experimento=study_name,    # <- NOMBRE del experimento (por mes)
                        save_root=models_root,     # <- raíz donde guardar modelos
                        seeds=SEEDS,               # <- semillas a entrenar
                        logger=logger,
                    )

                    # ---------- CURVAS por mes ----------
                    logger.info(f"[{study_name}] Calculando curvas de ganancia...")
                    models_dir = str(Path(models_root) / study_name)  # carpeta del estudio/mes
                    y_predicciones, curvas, mejores_cortes_normalizado = calculo_curvas_ganancia(
                        Xif=bundle['X_test'],
                        y_test_class=bundle['y_test_class'],
                        dir_model_opt=models_dir,  # 👈 sin concatenar nada adentro
                        resumen_csv_name="resumen_ganancias.csv",
                    )
                    logger.info(f"[{study_name}] mejores cortes: {mejores_cortes_normalizado}")

        # ---------------------------------------------------------------------------------
        # 5. PREDICCIÓN FINAL / ENSEMBLE (elige el mes de referencia)
        # ---------------------------------------------------------------------------------
        if START_POINT == 'PREDICT':
            # estrategia simple: usar el último mes de MES_TRAIN (o elegí el que quieras)
            mes_ref = max(meses_train_separados.keys())  # o el mes que quieras usar. Todos tienen el mismo X_pred
            experimento = f"{STUDY_NAME_OPTUNA}_{mes_ref}"
            models_dir = Path(DIR_MODELS) / experimento

            df_pred = pred_ensamble_modelos(
                Xif=meses_train_separados[mes_ref]['X_pred'],
                dir_model_opt=str(models_dir),
                output_path=OUTPUT_PATH,
                resumen_csv_name="resumen_ganancias.csv",
                experimento=experimento,
                k=6
            )

    except  Exception as e:
        logger.error(f'Se cortó ejecución por un error:\n {e}')

    logger.info("Fin Corrida")
    #print(data.columns)
if __name__ == '__main__':
    main()