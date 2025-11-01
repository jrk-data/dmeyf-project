import logging
import pandas as pd
import lightgbm as lgb
from pathlib import Path
import json
from src.config import (SEEDS,GANANCIA_ACIERTO,COSTO_ESTIMULO, STUDY_NAME_OPTUNA, DB_MODELS_TRAIN_PATH )
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from lightgbm.basic import LightGBMError
import duckdb
logger = logging.getLogger(__name__)


# --- INSTRUCCIÓN PARA SILENCIAR MATPLOTLIB ---

# 2. Silencia el logger específico de Matplotlib para el gestor de fuentes
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)



def train_model(study,X_train, y_train, weights, k):

    # Seleccionar top-k trials según 'value'
    df_trials = study.trials_dataframe()
    topk_df = (
        df_trials.nlargest(k, "value")
        .reset_index(drop=True)
        .loc[:, ["number", "value", "user_attrs_best_iter"]]
    )

    number_to_trial = {t.number: t for t in study.trials}

    # Datos a entrenar
    train_data = lgb.Dataset(X_train, label=y_train, weight=weights)

    # Path donde se guarda modelo entrenado
    save_dir = Path(f"/home/joacosk/Documents/maestria/Q2/script_project/src/models/{STUDY_NAME_OPTUNA}/")
    # Si el path no existe, lo creamos
    save_dir.mkdir(parents=True, exist_ok=True)

    # Seteo parámetros fijos
    final_params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
    'feature_pre_filter' : False,
    'metric': 'None',
        'max_bin': 31,
        'verbosity': -1,  # Para suprimir la salida
        'n_jobs': -1      # Para usar todos los cores
    }

    # itero el top de los modelos y guardo la posición en el top y el objeto trial
    experimento = STUDY_NAME_OPTUNA

    # array para guardar resumen de modelos y guardar metadata
    resumen_rows= list()

    logger.info(f"Entrenando {k} modelos")
    for top_rank, row in topk_df.iterrows():
        trial_num = int(row["number"])
        trial_val = float(row["value"])
        trial_obj = number_to_trial[trial_num]
        print(trial_val)
        num_boost_round = int(trial_obj.user_attrs.get("best_iter"))
        params = final_params.copy()
        # Obtengo los parámetros del trial
        params.update(trial_obj.params)

        for seed in SEEDS:

            try:
                file = f"train_model_lgb_optimization_top_{top_rank + 1}_seed_{seed}_experimento_{experimento}.txt"
                check_path = save_dir / file

                logger.info(f"Entrenando modelo {file}")

                if Path(check_path).exists():
                    logger.warning(f'Archivo {file} ya existe en directorio')
                    pass
                else:

                    # Agrgo semilla a params
                    params.update({'seed': seed})

                    # entrenamiento del modelo
                    model = lgb.train(
                        params=params,
                        train_set=train_data,
                        num_boost_round=int(num_boost_round)
                    )

                    # Guardado
                    out_path = save_dir / file
                    # si no existe, lo creo
                    #out_path.mkdir(parents=True, exist_ok=True)

                    model.save_model(str(out_path))

                    resumen_rows.append({
                        "top_rank": top_rank + 1,
                        "trial_number": trial_num,
                        "trial_value": trial_val,
                        "best_iter": int(num_boost_round),
                        "seed": int(seed),
                        "model_path": str(out_path),
                        "params": json.dumps(params)
                    })

            except Exception as e:
                logger.error(f"Error al entrenar modelo: {e}")


    logger.info(f"Modelos entrenados y guardados en {save_dir}")
    meta = pd.DataFrame(resumen_rows)
    TABLE_NAME = str(experimento) + '_train'  # Definimos el nombre de la tabla una vez

    try:
        # 1. Usamos 'with' para asegurar el cierre automático de la conexión
        with duckdb.connect(str(DB_MODELS_TRAIN_PATH)) as con:

            # 2. Ejecutamos la creación e inserción secuencialmente.
            #    'CREATE TABLE IF NOT EXISTS' maneja el caso de tabla ya existente.

            # Crear la tabla a partir del esquema de 'meta' si no existe
            con.execute(f"CREATE TABLE IF NOT EXISTS {TABLE_NAME} AS SELECT * FROM meta WHERE 1=0;")
            logger.info(f"Tabla '{TABLE_NAME}' asegurada (creada si no existía).")

            try:
                q_alter = 'ALTER TABLE lgb_optimization_exp1_train ADD PRIMARY KEY ("model_path");'
                con.sql(q_alter)
            except:
                pass

            # Insertar los datos del DataFrame 'meta' en la tabla
            con.execute(f"""INSERT INTO {TABLE_NAME} SELECT * FROM meta ON CONFLICT (model_path) DO NOTHING;""")
            logger.info(f"Datos insertados en la tabla '{TABLE_NAME}'.")

            # La conexión se cierra automáticamente aquí.

    # 3. Bloque 'except' para capturar cualquier error (conexión, creación, inserción)
    except Exception as e:
        # Mejor manejo de errores: captura cualquier excepción y loguea el error real.
        logger.error(f"Error al procesar la base de datos DuckDB para la tabla '{TABLE_NAME}': {e}")
        # Opcional: puedes re-lanzar el error si quieres que el programa se detenga
        # raise e



    # meta_path = save_dir / f"trained_models_metadata_experimento_{experimento}.csv"
    #
    # # Chequeo si ya existe metadata de este modelo. En caso de existir usamos mismo csv para guardar la nueva metadata
    # if Path(meta_path).exists():
    #     logger.warning(f'Archivo {meta} ya existe en directorio')
    #     df = pd.read_csv(meta_path)
    #     meta = pd.concat([df, meta], ignore_index=True)

    #meta.to_csv(meta_path, index=False)

    return meta




def calculo_curvas_ganancia(Xif,
                            y_test_class,
                            dir_model_opt,
                            resumen_csv_name: str = "resumen_ganancias_modelos.csv",
                            ):
    piso_envios = 4000
    techo_envios = 20000  # exclusivo

    # estilo común
    LINEWIDTH = 1.5
    ALPHA_MODELOS = 0.3  # transparencia para líneas que NO son el promedio
    ALPHA_PROM = 1.0  # promedio sin transparencia
    LS_PROM = '--'  # estilo del promedio

    # ----- figura única
    plt.figure(figsize=(10, 6))

    curvas = []
    mejores_cortes = {}  # {nombre_modelo: (k_envios, ganancia_max, thr_opt_prob)}
    probs_ordenadas = []  # lista de arrays con y_pred ordenado desc por modelo (para el promedio)

    y_predicciones = []

    # Concateno dirección de carpeta de modelos con nombre de experimento (que oficia de directorio)

    dir_model_opt = Path(dir_model_opt + STUDY_NAME_OPTUNA)

    logger.info(f"Seteando path para guardar modelos del experimento: {dir_model_opt}")

    try:
        dir_model_opt.mkdir(parents=True, exist_ok=True)
    except:
        pass

    logger.info(f"Obteniendo modelos de {dir_model_opt}")
    if not dir_model_opt.exists():
        logger.error(FileNotFoundError(f"❌ Carpeta no encontrada: {dir_model_opt}"))
        raise


    # 🔍 Buscar todos los modelos
    model_files = sorted([p for p in dir_model_opt.glob("*.txt")] + [p for p in dir_model_opt.glob("*.bin")])
    if not model_files:
        logger.error(FileNotFoundError(f"⚠️ No se encontraron modelos .txt o .bin en {dir_model_opt}"))
        raise

    modelos_validos = []
    for p in model_files:
        if not p.exists() or p.stat().st_size == 0:
            logger.warning(f"Saltando modelo inválido (no existe o vacío): {p}")
            continue
        # Intento abrir para verificar que realmente es un booster
        try:
            _ = lgb.Booster(model_file=str(p))
        except LightGBMError as e:
            logger.warning(f"Saltando modelo inválido (no es booster LGBM): {p} | {e}")
            continue
        modelos_validos.append(p)

    if not modelos_validos:
        raise RuntimeError(f"No hay modelos válidos en {dir_model_opt}")

    # Crear carpeta de salida para los gráficos
    curvas_dir = dir_model_opt / "curvas_de_complejidad"
    curvas_dir.mkdir(parents=True, exist_ok=True)



    # Para el CSV resumen (acumularemos y luego haremos upsert)
    resumen_rows = []

    # ganancia por fila (independiente del modelo)
    ganancia = np.where(y_test_class == "BAJA+2", GANANCIA_ACIERTO, 0) - \
               np.where(y_test_class != "BAJA+2", COSTO_ESTIMULO, 0)

    for model_file in modelos_validos:
        model = lgb.Booster(model_file=f"{model_file}")

        #filtro features utiliadas para entrenar al modelo
        feature_names = model.feature_name()
        Xif_filtered = Xif[feature_names]

        logger.info(f"Prediciendo datos con modelo {model_file}")
        y_pred = model.predict(Xif_filtered)

        #guardo las predicciones en df para poder compararlas entre sí
        df_pred_export = Xif[['numero_de_cliente','foto_mes']].copy()
        df_pred_export['y_pred'] = y_pred
        y_predicciones.append(df_pred_export)


        # ordeno por probabilidad descendente
        idx = np.argsort(y_pred)[::-1]
        y_pred_sorted = y_pred[idx]  # <-- PROBABILIDADES ORDENADAS
        gan_ord = ganancia[idx]  # ganancias alineadas al ranking

        # acumulada y segmento
        gan_cum = np.cumsum(gan_ord)
        curva_segmento = gan_cum[piso_envios:techo_envios]
        curvas.append(curva_segmento)
        probs_ordenadas.append(y_pred_sorted)

        # eje X: cantidad de envíos
        x_envios = np.arange(piso_envios, piso_envios + len(curva_segmento))

        # mejor k (dentro del segmento)
        argmax_local = int(np.argmax(curva_segmento))
        k_mejor = int(piso_envios + argmax_local)
        ganancia_max = float(curva_segmento[argmax_local])

        # Mejor probabilidad de corte
        thr_opt = gan_ord[k_mejor]

        # Umbral de probabilidad en el mejor k (ojo índice 0-based)
        k_idx = max(k_mejor - 1, 0)  # por seguridad si k_mejor==0 (no debería)
        thr_opt = float(y_pred_sorted[k_idx])

        # Se usa el método stem del objeto Path para quedarse con el nombre del archivo sin su extensión
        nombre = model_file.stem
        mejores_cortes[nombre] = (k_mejor, ganancia_max, thr_opt)

        # Para el CSV resumen
        resumen_rows.append({
            "modelo": nombre,
            "k_opt": int(k_mejor),
            "ganancia_max": float(ganancia_max),
            "thr_opt": float(thr_opt),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        # ploteo curva del modelo
        plt.plot(x_envios, curva_segmento,
                 label=nombre,
                 linewidth=LINEWIDTH,
                 alpha=ALPHA_MODELOS)

        # Guardar gráfico
        jpg_path = curvas_dir / f"{nombre}.jpg"
        plt.savefig(jpg_path, dpi=300)
        plt.close()

    # ----- promedio del segmento
    curvas_np = np.vstack(curvas)  # (n_modelos, n_puntos)
    promedio = curvas_np.mean(axis=0)
    x_envios = np.arange(piso_envios, piso_envios + len(promedio))

    x_argmax_local = int(np.argmax(promedio))
    x_k_mejor = int(piso_envios + x_argmax_local)
    x_ganancia_max = float(promedio[x_argmax_local])

    # Umbral promedio en probabilidad en el rank x_k_mejor:
    x_k_idx = max(x_k_mejor - 1, 0)
    # tomamos la probabilidad en ese rank para cada modelo y promediamos
    x_thr_opt = float(np.mean([p_sorted[x_k_idx] for p_sorted in probs_ordenadas]))

    plt.plot(x_envios, promedio,
             linewidth=LINEWIDTH,
             linestyle=LS_PROM,
             alpha=ALPHA_PROM,
             label=f'Promedio (n={len(model_files)})',
             zorder=5)
    plt.axvline(x=x_k_mejor, linestyle=':', linewidth=LINEWIDTH)

    # ----- decorado
    plt.title('Curvas de Ganancia - Modelos LGBM (eje: cantidad de envíos)')
    plt.xlabel('Cantidad de envíos (top-k)')
    plt.ylabel('Ganancia acumulada')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Salida normalizada (incluye probabilidad de corte)
    mejores_cortes_normalizado = {
        nombre: {'k': int(k), 'ganancia': float(g), 'thr_opt': float(thr)}
        for nombre, (k, g, thr) in mejores_cortes.items()
    }

    # Agrego el mejor corte del promedio
    mejores_cortes_normalizado['PROMEDIO'] = {
        'k': int(x_k_mejor),
        'ganancia': float(x_ganancia_max),
        'thr_opt': float(x_thr_opt)
    }
    # ===== Guardar/actualizar CSV resumen =====

    resumen_path = dir_model_opt / resumen_csv_name
    nuevos = pd.DataFrame(resumen_rows)

    # Guardar en BBDD test
    try:
        with duckdb.connect(str(DB_MODELS_TRAIN_PATH)) as con:

            # 1. Crear Vista Temporal (Asegura que DuckDB vea el DF 'nuevos')
            con.execute("CREATE OR REPLACE TEMP VIEW nuevos_data AS SELECT * FROM nuevos;")

            TABLE_NAME = str(resumen_path.name) + '_test'

            # 2. Asegurar la Tabla y la PK
            try:
                # CREAMOS la tabla si no existe, usando el esquema de la vista temporal
                con.execute(f"CREATE TABLE IF NOT EXISTS {TABLE_NAME} AS SELECT * FROM nuevos_data WHERE 1=0;")
                logger.info(f"Tabla '{TABLE_NAME}' asegurada (creada si no existía).")

                # Intentamos añadir la clave primaria (modelo)
                # Ojo: Usamos (modelo) sin comillas dobles
                q_alter = F'ALTER TABLE {TABLE_NAME} ADD PRIMARY KEY (modelo);'
                con.sql(q_alter)
                logger.info(f"Clave primaria (modelo) añadida a la tabla '{TABLE_NAME}'.")

            except Exception as e:
                error_msg = str(e).lower()
                if "already exists for this table" in error_msg:
                    logger.warning(f"La clave primaria en '{TABLE_NAME}' ya existía. Continuando con MERGE.")
                    pass
                else:
                    logger.error(f"Error CRÍTICO al crear/modificar tabla {TABLE_NAME}: {e}")
                    raise

            # 3. USAR MERGE INTO (Más robusto que ON CONFLICT)
            # Objetivo: Si existe, actualizamos el timestamp. Si NO existe (la fila eliminada), la insertamos.
            current_timestamp = datetime.now().strftime("'%Y-%m-%d %H:%M:%S'")

            con.execute(f"""
                    MERGE INTO {TABLE_NAME} AS t
                    USING nuevos_data AS s
                    ON t.modelo = s.modelo
                    WHEN MATCHED THEN
                        -- Si el registro existe, actualizamos el timestamp (y otros campos si quieres)
                        UPDATE SET 
                            ganancia_max = s.ganancia_max,
                            k_opt = s.k_opt,
                            thr_opt = s.thr_opt,
                            timestamp = {current_timestamp} -- Actualizamos el timestamp de la fila existente
                    WHEN NOT MATCHED THEN
                        -- Si el registro no existe (la fila eliminada), la insertamos
                        INSERT *;
                """)
            logger.info(
                f'Se procesaron registros en tabla {TABLE_NAME} usando MERGE. (Fila faltante insertada, existentes actualizadas).')

    except Exception as e:
        logger.error(f"Error general en la operación de DuckDB: {e}")




    if resumen_path.exists():
        prev = pd.read_csv(resumen_path)
        merged = pd.concat([prev, nuevos], ignore_index=True)
        merged.drop_duplicates(subset=["modelo"], keep="last", inplace=True)
        resumen_path = resumen_path.with_suffix(".csv")
        merged.to_csv(resumen_path, index=False)
    else:
        nuevos.to_csv(resumen_path, index=False)

    print(f"\n✅ CSV resumen actualizado: {resumen_path}")
    print(f"✅ Gráficos guardados en: {curvas_dir}")

    return y_predicciones,curvas, mejores_cortes_normalizado


def pred_ensamble_modelos(Xif,
                     dir_model_opt,
                    experimento,
                     k
                     ):
    # Me conecto a duckdb y selecciono el top 5 de modelos
    con = duckdb.connect(str(DB_MODELS_TRAIN_PATH))
    q = f'select * from {experimento}_test order by ganancia_max desc limit {k};'
    df_top_k = con.sql(q).df()
    con.close()
    logger.info(f"Se seleccionaron los 5 mejores modelos de la tabla {experimento}_test")
    logger.info(df_top_k)

    list_of_predictions = []
    # =========================================================================
    # 1. GENERAR PREDICCIÓN BINARIA PARA CADA MODELO Y CADA CLIENTE
    # =========================================================================
    base_dir = Path(dir_model_opt)

    for modelo in df_top_k['modelo']:
        # Obtengo el thr_optimo
        # CORRECCIÓN IMPORTANTE: Usar .iloc[0] para extraer el valor escalar de la selección.
        try:
            thr_opt = df_top_k[df_top_k['modelo'] == modelo]['thr_opt'].iloc[0]
        except IndexError:
            logger.error(f"No se encontró el umbral óptimo ('thr_opt') para el modelo: {modelo}")
            continue

        logger.info(f"Prediciendo con modelo {modelo} (Thr: {thr_opt:.5f})")
        model_path = base_dir / experimento / modelo

        # Aseguramos que el archivo termine en .txt o .bin (ya que 'modelo' es el nombre base)
        if not model_path.exists():
            model_path = model_path.with_suffix('.txt')
            if not model_path.exists():
                logger.error(f"Archivo de modelo no encontrado en: {model_path}")
                continue

        try:
            model = lgb.Booster(model_file=str(model_path))

            # Extraigo nombres de features que se usaron para entrenar el modelo
            feature_names = model.feature_name()
            # Filtro el dataset con las features utilizadas para entrenar el modelo
            Xif_filtered = Xif[feature_names]

            y_pred_prob = model.predict(Xif_filtered)
        except Exception as e:
            logger.error(f"Error al cargar/predecir con modelo {modelo}: {e}")
            continue

        # DataFrame auxiliar para guardar la predicción BINARIA (el voto)
        df_pred_bin = Xif[['numero_de_cliente', 'foto_mes']].copy()

        # Aplicar el umbral óptimo y guardar el voto como una columna separada
        # Usamos np.where para mayor eficiencia que .apply()
        df_pred_bin[f'voto_{modelo}'] = np.where(y_pred_prob >= thr_opt, 1, 0)

        list_of_predictions.append(df_pred_bin)

    # =========================================================================
    # 2. CONSOLIDAR Y APLICAR EL VOTO DE MAYORÍA
    # =========================================================================

    if not list_of_predictions:
        logger.error("No se generó ninguna predicción. Retornando DataFrame vacío.")
        df_final_votos = pd.DataFrame(columns=['numero_de_cliente', 'foto_mes', 'y_pred'])
    else:
        # Hacemos merge secuencial o join de todos los DataFrames en la lista
        # Usamos el primer DF como base
        df_final_votos = list_of_predictions[0]

        for df_pred in list_of_predictions[1:]:
            # Unimos las columnas de votos (manteniendo 'numero_de_cliente' y 'foto_mes')
            df_final_votos = df_final_votos.merge(
                df_pred,
                on=['numero_de_cliente', 'foto_mes'],
                how='left'
            )

        # Identificar todas las columnas que contienen los votos (las que empiezan con 'voto_')
        voto_cols = [col for col in df_final_votos.columns if col.startswith('voto_')]

        # Sumar los votos (cuántos modelos predijeron '1' para ese cliente)
        df_final_votos['votos_positivos'] = df_final_votos[voto_cols].sum(axis=1)

        # Voto de Mayoría: La predicción final es '1' si la suma de votos > (número total de modelos / 2)
        n_modelos = len(voto_cols)
        umbral_mayoria = n_modelos / 2

        df_final_votos['y_pred_baja'] = np.where(
            df_final_votos['votos_positivos'] > umbral_mayoria,
            1,
            0
        )

        # Limpieza: El resultado final debe ser solo con identificadores y la predicción final.
        df_final_votos = df_final_votos[['numero_de_cliente', 'foto_mes', 'y_pred_baja']].drop_duplicates(
            subset=['numero_de_cliente'])
        df_final_votos.rename(columns={'y_pred_baja': 'y_pred'}, inplace=True)

        df_final_votos.to_csv(f'/home/joacosk/Documents/maestria/Q2/script_project/output/{experimento}.csv',index=False)

    logger.info(f"DataFrame final de votos de mayoría creado con {len(df_final_votos)} clientes únicos.")