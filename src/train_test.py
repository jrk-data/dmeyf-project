import logging
import pandas as pd
import polars as pl
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
from src.utils import _coerce_object_cols

logger = logging.getLogger(__name__)


# --- INSTRUCCIÓN PARA SILENCIAR MATPLOTLIB ---

# 2. Silencia el logger específico de Matplotlib para el gestor de fuentes
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# -- Funcion Helper que sirve para definir nombre de tabla de ganancias

def _resumen_table_name(resumen_csv_name: str) -> str:
    return f"{Path(resumen_csv_name).stem}_test"


def train_model(study, X_train, y_train, weights, k,
                base_study_name: str,
                mes,
                save_root, seeds, logger):

    if isinstance(X_train, pl.DataFrame):
        X_train = X_train.to_pandas()

    # 🔧 Arreglo clave:
    X_train = _coerce_object_cols(X_train)

    # Seleccionar top-k trials según 'value'
    df_trials = study.trials_dataframe()

    # Identificar nombre de columna correcto en el DataFrame de Optuna
    col_best_iter = "user_attrs_mean_best_iter"

    # Verificamos si existe, por si acaso se corrió con otra lógica antes
    if col_best_iter not in df_trials.columns:
        # Fallback por si en alguna versión vieja se llamó 'best_iter'
        if "user_attrs_best_iter" in df_trials.columns:
            col_best_iter = "user_attrs_best_iter"
        else:
            logger.error(f"No se encontró columna de iteraciones. Columnas disponibles: {df_trials.columns}")
            raise KeyError("No se encontró 'user_attrs_mean_best_iter' ni 'user_attrs_best_iter'")

    topk_df = (
        df_trials.nlargest(k, "value")
        .reset_index(drop=True)
        .loc[:, ["number", "value", col_best_iter]]
    )

    number_to_trial = {t.number: t for t in study.trials}

    # Datos a entrenar
    train_data = lgb.Dataset(X_train, label=y_train, weight=weights)

    # PATH DE MODELOS: por experimento/mes
    save_dir = Path(save_root) / str(base_study_name) / str(mes)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Seteo parámetros fijos
    final_params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'feature_pre_filter' : False,
        'metric': 'None',
        'max_bin': 31,
        'verbosity': -1,
        'n_jobs': -1
    }

    # array para guardar resumen de modelos y guardar metadata
    resumen_rows= list()

    logger.info(f"Entrenando {k} modelos")
    for top_rank, row in topk_df.iterrows():
        trial_num = int(row["number"])
        trial_val = float(row["value"])
        trial_obj = number_to_trial[trial_num]
        print(f"Trial Value: {trial_val}")

        # --- INICIO CORRECCIÓN -----------------------------------------
        # 1. Intentar obtener 'mean_best_iter' (lógica semillerío)
        raw_best_iter = trial_obj.user_attrs.get("mean_best_iter")

        # 2. Fallback: intentar obtener 'best_iter' (lógica antigua)
        if raw_best_iter is None:
            raw_best_iter = trial_obj.user_attrs.get("best_iter")

        # 3. Validación para evitar TypeError: int() argument must be...
        if raw_best_iter is None:
            logger.error(f"El trial {trial_num} no tiene 'mean_best_iter' ni 'best_iter' en user_attrs.")
            logger.error(f"Keys disponibles: {list(trial_obj.user_attrs.keys())}")
            raise ValueError(f"No se pudo determinar num_boost_round para el trial {trial_num}")

        num_boost_round = int(raw_best_iter)
        # --- FIN CORRECCIÓN --------------------------------------------

        params = final_params.copy()
        # Obtengo los parámetros del trial
        params.update(trial_obj.params)

        for seed in seeds:
            try:
                file = f"lgb_top{top_rank + 1}_seed_{int(seed)}.txt"
                check_path = save_dir / file

                logger.info(f"Entrenando modelo {file} (Rounds: {num_boost_round})")

                if Path(check_path).exists():
                    logger.warning(f'Archivo {file} ya existe en directorio')
                    pass
                else:
                    # Agrego semilla a params
                    params.update({'seed': seed})

                    # entrenamiento del modelo
                    model = lgb.train(
                        params=params,
                        train_set=train_data,
                        num_boost_round=int(num_boost_round)
                    )

                    # Guardado
                    out_path = save_dir / file
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
    TABLE_NAME = f"{base_study_name}_train"

    try:
        with duckdb.connect(str(config.DB_MODELS_TRAIN_PATH)) as con:
            con.execute(f"CREATE TABLE IF NOT EXISTS {TABLE_NAME} AS SELECT * FROM meta WHERE 1=0;")
            try:
                con.sql(f'ALTER TABLE {TABLE_NAME} ADD PRIMARY KEY ("model_path");')
            except:
                pass
            con.execute(f"""INSERT INTO {TABLE_NAME} SELECT * FROM meta ON CONFLICT (model_path) DO NOTHING;""")
            logger.info(f"Datos insertados en la tabla '{TABLE_NAME}'.")
    except Exception as e:
        logger.error(f"Error al procesar la base de datos DuckDB para la tabla '{TABLE_NAME}': {e}")

    return meta



    # meta_path = save_dir / f"trained_models_metadata_experimento_{experimento}.csv"
    #
    # # Chequeo si ya existe metadata de este modelo. En caso de existir usamos mismo csv para guardar la nueva metadata
    # if Path(meta_path).exists():
    #     logger.warning(f'Archivo {meta} ya existe en directorio')
    #     df = pd.read_csv(meta_path)
    #     meta = pd.concat([df, meta], ignore_index=True)

    #meta.to_csv(meta_path, index=False)

    return meta


def calculo_curvas_ganancia(Xif, y_test_class, dir_model_opt,
                            experimento_key: str,
                            resumen_csv_name: str = "resumen_ganancias.csv"):
    piso_envios = 4000
    techo_envios = 20000  # exclusivo

    # estilo común
    LINEWIDTH = 1.5
    ALPHA_MODELOS = 0.5
    ALPHA_PROM = 1.0
    LS_PROM = '-'

    # ----- Arreglando tipos de datos -----
    if isinstance(Xif, pl.DataFrame):
        Xif = Xif.to_pandas()

    # 🔧 Arreglo clave:
    Xif = _coerce_object_cols(Xif)

    # 1. Detectar Meses para Título y Nombre de CARPETA
    meses_titulo = "Desconocido"
    meses_archivo_str = "meses_desconocidos"

    if 'foto_mes' in Xif.columns:
        meses_unicos = sorted(Xif['foto_mes'].unique())
        meses_titulo = ", ".join([str(m) for m in meses_unicos])
        meses_archivo_str = "_".join([str(m) for m in meses_unicos])

    curvas = []
    mejores_cortes = {}
    probs_ordenadas = []
    y_predicciones = []

    dir_model_opt = Path(dir_model_opt)
    try:
        dir_model_opt.mkdir(parents=True, exist_ok=True)
    except:
        pass

    # 🔍 Buscar todos los modelos
    model_files = sorted([p for p in dir_model_opt.glob("*.txt")] + [p for p in dir_model_opt.glob("*.bin")])
    if not model_files:
        raise RuntimeError(f"No hay modelos válidos en {dir_model_opt}")

    # --- Definición explicita de modelos_validos ---
    modelos_validos = []
    for p in model_files:
        if not p.exists() or p.stat().st_size == 0:
            logger.warning(f"Saltando modelo inválido (no existe o vacío): {p}")
            continue
        try:
            _ = lgb.Booster(model_file=str(p))
            modelos_validos.append(p)
        except LightGBMError as e:
            logger.warning(f"Saltando modelo inválido (no es booster LGBM): {p} | {e}")
            continue

    if not modelos_validos:
        raise RuntimeError(f"No quedan modelos válidos en {dir_model_opt}")

    # -----------------------------------------------------------
    # 📂 CREACIÓN DE CARPETA: MESES + NOMBRE_EXPERIMENTO
    # -----------------------------------------------------------
    # Limpiamos el nombre del experimento por si tiene rutas
    nombre_exp_limpio = Path(experimento_key).name

    # Formato: 202105_202107_c03_exp01_baseline
    folder_name = f"{meses_archivo_str}_{nombre_exp_limpio}"

    base_curvas_dir = dir_model_opt / "curvas_de_complejidad"
    target_folder = base_curvas_dir / folder_name

    target_folder.mkdir(parents=True, exist_ok=True)
    logger.info(f"Guardando curvas en carpeta específica: {target_folder}")

    resumen_rows = []

    # ganancia por fila
    ganancia = np.where(y_test_class == "BAJA+2", config.GANANCIA_ACIERTO, 0) - \
               np.where(y_test_class != "BAJA+2", config.COSTO_ESTIMULO, 0)

    # Inicializamos figura para el plot acumulativo (todos juntos)
    plt.figure(figsize=(12, 7))

    for model_file in modelos_validos:
        nombre = model_file.stem  # Ej: lgb_top1_seed_155555

        # --- Lógica de predicción ---
        model = lgb.Booster(model_file=str(model_file))
        feature_names = model.feature_name()
        Xif_filtered = Xif[feature_names]
        y_pred = model.predict(Xif_filtered)

        df_pred_export = Xif[['numero_de_cliente', 'foto_mes']].copy()
        df_pred_export['y_pred'] = y_pred
        y_predicciones.append(df_pred_export)

        # --- Lógica de curva ---
        idx = np.argsort(y_pred)[::-1]
        y_pred_sorted = y_pred[idx]
        gan_ord = ganancia[idx]
        gan_cum = np.cumsum(gan_ord)
        curva_segmento = gan_cum[piso_envios:techo_envios]
        curvas.append(curva_segmento)
        probs_ordenadas.append(y_pred_sorted)

        # eje X
        x_envios = np.arange(piso_envios, piso_envios + len(curva_segmento))

        # métricas
        argmax_local = int(np.argmax(curva_segmento))
        k_mejor = int(piso_envios + argmax_local)
        ganancia_max = float(curva_segmento[argmax_local])
        k_idx = max(k_mejor - 1, 0)
        thr_opt = float(y_pred_sorted[k_idx])

        mejores_cortes[nombre] = (k_mejor, ganancia_max, thr_opt)

        resumen_rows.append({
            "experimento": experimento_key,
            "modelo": nombre,
            "k_opt": int(k_mejor),
            "ganancia_max": float(ganancia_max),
            "thr_opt": float(thr_opt),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        # --- Ploteo en la figura conjunta ---
        p = plt.plot(x_envios, curva_segmento, label=nombre, linewidth=LINEWIDTH, alpha=ALPHA_MODELOS)
        color_linea = p[0].get_color()
        plt.axvline(x=k_mejor, color=color_linea, linestyle='--', linewidth=0.8, alpha=0.5)

        # --- Guardado Individual ---
        fig_temp = plt.figure(figsize=(10, 6))
        plt.plot(x_envios, curva_segmento, color=color_linea, label=nombre)
        plt.axvline(x=k_mejor, color=color_linea, linestyle='--', label=f'Max: {k_mejor}')
        plt.title(f'{nombre} - {meses_titulo}')
        plt.xlabel('Envíos')
        plt.ylabel('Ganancia')
        plt.legend()
        plt.grid(True, alpha=0.3)

        nombre_archivo_individual = f"{nombre}.jpg"
        plt.savefig(target_folder / nombre_archivo_individual, dpi=150)
        plt.close(fig_temp)

    # ----- Volvemos a la figura conjunta (promedio) -----
    plt.figure(1)

    curvas_np = np.vstack(curvas)
    promedio = curvas_np.mean(axis=0)
    x_envios = np.arange(piso_envios, piso_envios + len(promedio))

    x_argmax_local = int(np.argmax(promedio))
    x_k_mejor = int(piso_envios + x_argmax_local)
    x_ganancia_max = float(promedio[x_argmax_local])

    x_k_idx = max(x_k_mejor - 1, 0)
    x_thr_opt = float(np.mean([p_sorted[x_k_idx] for p_sorted in probs_ordenadas]))

    plt.plot(x_envios, promedio, linewidth=2.5, linestyle=LS_PROM, color='black', alpha=ALPHA_PROM,
             label=f'Promedio', zorder=10)
    plt.axvline(x=x_k_mejor, color='black', linestyle=':', linewidth=2, label=f'Corte Promedio ({x_k_mejor})')

    plt.title(f'Curvas de Ganancia - {meses_titulo} \nExp: {nombre_exp_limpio}', fontsize=12)
    plt.xlabel('Cantidad de envíos (Rank)', fontsize=12)
    plt.ylabel('Ganancia acumulada', fontsize=12)
    plt.legend(fontsize=10, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    nombre_archivo_conjunto = f"curva_ganancia_conjunta.jpg"
    plt.savefig(target_folder / nombre_archivo_conjunto, dpi=300)
    plt.close()

    # Salida normalizada
    mejores_cortes_normalizado = {
        nombre: {'k': int(k), 'ganancia': float(g), 'thr_opt': float(thr)}
        for nombre, (k, g, thr) in mejores_cortes.items()
    }
    mejores_cortes_normalizado['PROMEDIO'] = {
        'k': int(x_k_mejor),
        'ganancia': float(x_ganancia_max),
        'thr_opt': float(x_thr_opt)
    }

    # ===== Guardar/actualizar CSV resumen =====
    resumen_path = dir_model_opt / resumen_csv_name
    nuevos = pd.DataFrame(resumen_rows)

    try:
        with duckdb.connect(str(config.DB_MODELS_TRAIN_PATH)) as con:
            con.execute("CREATE OR REPLACE TEMP VIEW nuevos_data AS SELECT * FROM nuevos;")
            TABLE_NAME = _resumen_table_name(resumen_csv_name)
            try:
                con.execute(f"CREATE TABLE IF NOT EXISTS {TABLE_NAME} AS SELECT * FROM nuevos_data WHERE 1=0;")
                q_alter = f'ALTER TABLE {TABLE_NAME} ADD PRIMARY KEY (experimento, modelo);'
                con.sql(q_alter)
            except Exception as e:
                pass

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
    print(f"✅ Gráficos guardados en: {target_folder}")

    return y_predicciones, curvas, mejores_cortes_normalizado

def pred_ensamble_modelos(
    Xif: pd.DataFrame,
    dir_model_opt: str | Path,   # p.ej. ".../src/models/STUDY_NAME_OPTUNA_202003"
    experimento: str,            # p.ej. "STUDY_NAME_OPTUNA_202003"
    k: int,
    output_path,
    resumen_csv_name: str = "resumen_ganancias.csv"
) -> pd.DataFrame:
    """
    Ensambla las predicciones de los top-k modelos (por 'ganancia_max')
    del experimento/mes indicado, aplicando voto de mayoría con el 'thr_opt' por modelo.
    Guarda CSV final de predicciones y retorna el DataFrame.
    """
    base_dir = Path(dir_model_opt)
    base_dir.mkdir(parents=True, exist_ok=True)

    # ----- Arreglando tipos de datos -----
    if isinstance(Xif, pl.DataFrame):
        Xif = Xif.to_pandas()

    # 🔧 Arreglo clave:
    Xif = _coerce_object_cols(Xif)
    # ----- Arreglando tipos de datos -----

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
        df_final[['numero_de_cliente', 'y_pred']]
        .drop_duplicates(subset=['numero_de_cliente'], keep='last')
        .reset_index(drop=True)
    )
    df_final_out = df_final_out.rename(columns={
        'y_pred': 'Predicted'
    })


    # Guardar CSV
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{experimento}.csv"
    df_final_out.to_csv(out_csv, index=False)
    logger.info(f"[{experimento}] Ensemble guardado en {out_csv} (clientes={len(df_final_out)})")

    return df_final_out

def pred_ensamble_desde_experimentos(
    Xif: pd.DataFrame,
    experiments: list[dict],   # [{"dir": ".../models/.../201901", "experimento": "exp_201901"}, ...]
    k: int,
    output_path,
    output_basename: str,
    resumen_csv_name: str = "resumen_ganancias.csv"
) -> pd.DataFrame:

    # Normalización de tipos (igual que en pred_ensamble_modelos)
    if isinstance(Xif, pl.DataFrame):
        Xif = Xif.to_pandas()
    Xif = _coerce_object_cols(Xif)

    votos = []
    with duckdb.connect(str(config.DB_MODELS_TRAIN_PATH)) as con:
        table_resumen = _resumen_table_name(resumen_csv_name)

        for item in experiments:
            base_dir = Path(item["dir"])
            experimento = item["experimento"]

            q = f"""
                SELECT modelo, thr_opt, ganancia_max
                FROM {table_resumen}
                WHERE experimento = ?
                ORDER BY ganancia_max DESC
                LIMIT {int(k)};
            """
            df_top_k = con.execute(q, [experimento]).df()
            if df_top_k.empty:
                logger.warning(f"[{experimento}] sin modelos en {table_resumen}")
                continue

            for _, row in df_top_k.iterrows():
                modelo = str(row['modelo'])
                thr_opt = float(row['thr_opt'])

                model_path = base_dir / modelo
                if not model_path.suffix:
                    cand_txt = model_path.with_suffix('.txt')
                    cand_bin = model_path.with_suffix('.bin')
                    if cand_txt.exists():
                        model_path = cand_txt
                    elif cand_bin.exists():
                        model_path = cand_bin

                if not model_path.exists():
                    logger.error(f"[{experimento}] Modelo no encontrado: {model_path}")
                    continue

                booster = lgb.Booster(model_file=str(model_path))
                feature_names = booster.feature_name()
                y_pred_prob = booster.predict(Xif[feature_names])

                df_voto = Xif[['numero_de_cliente', 'foto_mes']].copy()
                df_voto[f'voto_{experimento}_{Path(modelo).stem}'] = (y_pred_prob >= thr_opt).astype(int)
                votos.append(df_voto)

    if not votos:
        logger.error("No se generaron votos. Devuelvo vacío.")
        return pd.DataFrame(columns=['numero_de_cliente', 'Predicted'])

    df_final = votos[0]
    for df_pred in votos[1:]:
        df_final = df_final.merge(df_pred, on=['numero_de_cliente', 'foto_mes'], how='left')

    voto_cols = [c for c in df_final.columns if c.startswith('voto_')]
    df_final[voto_cols] = df_final[voto_cols].fillna(0).astype(int)

    n_modelos = len(voto_cols)
    umbral_mayoria = n_modelos / 2.0

    df_final['votos_positivos'] = df_final[voto_cols].sum(axis=1)
    df_final['Predicted'] = (df_final['votos_positivos'] > umbral_mayoria).astype(int)

    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{output_basename}.csv"
    df_out = df_final[['numero_de_cliente', 'Predicted']].drop_duplicates('numero_de_cliente', keep='last')
    df_out.to_csv(out_csv, index=False)
    logger.info(f"[{output_basename}] Ensemble multi-experimento guardado en {out_csv} (clientes={len(df_out)})")

    return df_out
