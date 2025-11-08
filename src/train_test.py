import logging
import pandas as pd
import lightgbm as lgb
from pathlib import Path
import json
import src.config as config
#from src.config import (SEEDS,GANANCIA_ACIERTO,COSTO_ESTIMULO, STUDY_NAME_OPTUNA, DB_MODELS_TRAIN_PATH )
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from lightgbm.basic import LightGBMError
import duckdb
logger = logging.getLogger(__name__)


# --- INSTRUCCIÓN PARA SILENCIAR MATPLOTLIB ---

# 2. Silencia el logger específico de Matplotlib para el gestor de fuentes
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# -- Funcion Helper que sirve para definir nombre de tabla de ganancias

def _resumen_table_name(resumen_csv_name: str) -> str:
    return f"{Path(resumen_csv_name).stem}_test"


def train_model(study, X_train, y_train, weights, k,
                experimento, save_root, seeds, logger):

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

    # PATH DE MODELOS: por experimento/mes
    save_dir = Path(save_root) / str(experimento)
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
    experimento = config.STUDY_NAME_OPTUNA

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

        for seed in seeds:
            try:
                file = f"lgb_top{top_rank + 1}_seed_{int(seed)}.txt"
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
    TABLE_NAME = f"{experimento}_train"  # Definimos el nombre de la tabla una vez

    try:
        # 1. Usamos 'with' para asegurar el cierre automático de la conexión
        with duckdb.connect(str(config.DB_MODELS_TRAIN_PATH)) as con:

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

    dir_model_opt = Path(dir_model_opt)

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
    ganancia = np.where(y_test_class == "BAJA+2", config.GANANCIA_ACIERTO, 0) - \
               np.where(y_test_class != "BAJA+2", config.COSTO_ESTIMULO, 0)

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

        k_idx = max(k_mejor - 1, 0)
        thr_opt = float(y_pred_sorted[k_idx])

        # Se usa el método stem del objeto Path para quedarse con el nombre del archivo sin su extensión
        nombre = model_file.stem
        mejores_cortes[nombre] = (k_mejor, ganancia_max, thr_opt)

        # Para el CSV resumen
        resumen_rows.append({
            "experimento": dir_model_opt.name,  # ej: STUDY_NAME_OPTUNA_202003
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
        with duckdb.connect(str(config.DB_MODELS_TRAIN_PATH)) as con:

            # 1. Crear Vista Temporal (Asegura que DuckDB vea el DF 'nuevos')
            con.execute("CREATE OR REPLACE TEMP VIEW nuevos_data AS SELECT * FROM nuevos;")

            TABLE_NAME = _resumen_table_name(resumen_csv_name)  # usa stem para evitar '.csv' en el nombre

            # 2. Asegurar la Tabla y la PK compuesta (experimento, modelo)
            try:
                con.execute(f"CREATE TABLE IF NOT EXISTS {TABLE_NAME} AS SELECT * FROM nuevos_data WHERE 1=0;")
                logger.info(f"Tabla '{TABLE_NAME}' asegurada (creada si no existía).")
                # PK compuesta
                q_alter = f'ALTER TABLE {TABLE_NAME} ADD PRIMARY KEY (experimento, modelo);'
                con.sql(q_alter)
                logger.info(f"Clave primaria (experimento, modelo) añadida a la tabla '{TABLE_NAME}'.")
            except Exception as e:
                error_msg = str(e).lower()
                if "already exists for this table" in error_msg:
                    logger.warning(f"La clave primaria en '{TABLE_NAME}' ya existía. Continuando con MERGE.")
                else:
                    logger.error(f"Error CRÍTICO al crear/modificar tabla {TABLE_NAME}: {e}")
                    raise

            # 3. MERGE usando (experimento, modelo)
            current_timestamp = datetime.now().strftime("'%Y-%m-%d %H:%M:%S'")

            con.execute(f"""
                MERGE INTO {TABLE_NAME} AS t
                USING nuevos_data AS s
                ON t.experimento = s.experimento AND t.modelo = s.modelo
                WHEN MATCHED THEN
                    UPDATE SET 
                        ganancia_max = s.ganancia_max,
                        k_opt = s.k_opt,
                        thr_opt = s.thr_opt,
                        timestamp = {current_timestamp}
                WHEN NOT MATCHED THEN
                    INSERT *;
            """)
            logger.info(
                f'Se procesaron registros en tabla {TABLE_NAME} usando MERGE. (Fila faltante insertada, existentes actualizadas).')

    except Exception as e:
        logger.error(f"Error general en la operación de DuckDB: {e}")




    if resumen_path.exists():
        prev = pd.read_csv(resumen_path)
        merged = pd.concat([prev, nuevos], ignore_index=True)
        merged.drop_duplicates(subset=["experimento", "modelo"], keep="last", inplace=True)
        resumen_path = resumen_path.with_suffix(".csv")
        merged.to_csv(resumen_path, index=False)
    else:
        nuevos.to_csv(resumen_path, index=False)

    print(f"\n✅ CSV resumen actualizado: {resumen_path}")
    print(f"✅ Gráficos guardados en: {curvas_dir}")

    return y_predicciones,curvas, mejores_cortes_normalizado


def pred_ensamble_modelos(
    Xif: pd.DataFrame,
    dir_model_opt: str | Path,   # p.ej. ".../src/models/STUDY_NAME_OPTUNA_202003"
    experimento: str,            # p.ej. "STUDY_NAME_OPTUNA_202003"
    k: int,
    output_path,
    resumen_csv_name: str = "resumen_ganancias_modelos.csv"
) -> pd.DataFrame:
    """
    Ensambla las predicciones de los top-k modelos (por 'ganancia_max')
    del experimento/mes indicado, aplicando voto de mayoría con el 'thr_opt' por modelo.
    Guarda CSV final de predicciones y retorna el DataFrame.
    """
    base_dir = Path(dir_model_opt)
    base_dir.mkdir(parents=True, exist_ok=True)

    # ===== 1) Top-K modelos desde DuckDB, filtrando por experimento =====
    with duckdb.connect(str(config.DB_MODELS_TRAIN_PATH)) as con:
        table_resumen = _resumen_table_name(resumen_csv_name)
        q = f"""
            SELECT modelo, thr_opt, ganancia_max
            FROM {table_resumen}
            WHERE experimento = ?
            ORDER BY ganancia_max DESC
            LIMIT {int(k)};
        """
        df_top_k = con.execute(q, [experimento]).df()

    if df_top_k.empty:
        logger.error(f"No se encontraron modelos para experimento '{experimento}' en {table_resumen}.")
        return pd.DataFrame(columns=['numero_de_cliente', 'foto_mes', 'y_pred'])

    logger.info(f"[{experimento}] Top-{k} modelos seleccionados:")
    logger.info(df_top_k)

    # ===== 2) Predicción binaria por modelo =====
    lista_votos = []

    for _, row in df_top_k.iterrows():
        modelo = str(row['modelo'])
        thr_opt = float(row['thr_opt'])

        # Los modelos se guardaron como: base_dir / "lgb_top{rank}_seed_{seed}.txt"
        # df_top_k['modelo'] ya trae el nombre de archivo (sin extensión o con .txt/.bin).
        model_path = base_dir / modelo
        if not model_path.suffix:
            # probar .txt y .bin
            cand_txt = model_path.with_suffix('.txt')
            cand_bin = model_path.with_suffix('.bin')
            if cand_txt.exists():
                model_path = cand_txt
            elif cand_bin.exists():
                model_path = cand_bin

        if not model_path.exists():
            logger.error(f"[{experimento}] Modelo no encontrado: {model_path}")
            continue

        try:
            booster = lgb.Booster(model_file=str(model_path))
            feature_names = booster.feature_name()

            # Alinear features del test con las del modelo
            #Xif_filtered = Xif.reindex(columns=feature_names, fill_value=0)
            Xif_filtered = Xif[feature_names]

            y_pred_prob = booster.predict(Xif_filtered)
        except Exception as e:
            logger.error(f"[{experimento}] Error con modelo {modelo}: {e}")
            continue

        df_voto = Xif[['numero_de_cliente', 'foto_mes']].copy()
        df_voto[f'voto_{Path(modelo).stem}'] = (y_pred_prob >= thr_opt).astype(int)
        lista_votos.append(df_voto)

    # ===== 3) Consolidar votos y aplicar mayoría =====
    if not lista_votos:
        logger.error(f"[{experimento}] No se generaron predicciones. Devuelvo vacío.")
        return pd.DataFrame(columns=['numero_de_cliente', 'foto_mes', 'y_pred'])

    df_final = lista_votos[0]
    for df_pred in lista_votos[1:]:
        df_final = df_final.merge(df_pred, on=['numero_de_cliente', 'foto_mes'], how='left')

    voto_cols = [c for c in df_final.columns if c.startswith('voto_')]
    if not voto_cols:
        logger.error(f"[{experimento}] No hay columnas de voto. Devuelvo vacío.")
        return pd.DataFrame(columns=['numero_de_cliente', 'foto_mes', 'y_pred'])

    # Evitar NaNs si algún merge dejó faltantes
    df_final[voto_cols] = df_final[voto_cols].fillna(0).astype(int)

    n_modelos = len(voto_cols)
    umbral_mayoria = n_modelos / 2.0  # mayoría estricta: > n/2

    df_final['votos_positivos'] = df_final[voto_cols].sum(axis=1)
    df_final['y_pred'] = (df_final['votos_positivos'] > umbral_mayoria).astype(int)

    # Output final
    df_final_out = (
        df_final[['numero_de_cliente', 'foto_mes', 'y_pred']]
        .drop_duplicates(subset=['numero_de_cliente'], keep='last')
        .reset_index(drop=True)
    )

    # Guardar CSV
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{experimento}.csv"
    df_final_out.to_csv(out_csv, index=False)
    logger.info(f"[{experimento}] Ensemble guardado en {out_csv} (clientes={len(df_final_out)})")

    return df_final_out
