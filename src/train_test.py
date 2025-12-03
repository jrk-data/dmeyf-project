import logging
import pandas as pd
import polars as pl
import lightgbm as lgb
from pathlib import Path
import json
import src.config as config
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import duckdb
from src.utils import _coerce_object_cols
import re
import traceback

logger = logging.getLogger(__name__)

# --- INSTRUCCIÓN PARA SILENCIAR MATPLOTLIB ---
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)


# ==============================================================================
#  FUNCIONES HELPER
# ==============================================================================

def _resumen_table_name(resumen_csv_name: str) -> str:
    return f"{Path(resumen_csv_name).stem}_test"


def _update_csv_metrics(rows: list, csv_path: Path):
    if not rows: return
    nuevos_df = pd.DataFrame(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_path.exists():
        try:
            prev_df = pd.read_csv(csv_path)
            merged_df = pd.concat([prev_df, nuevos_df], ignore_index=True)
            merged_df.drop_duplicates(subset=["experimento", "modelo"], keep="last", inplace=True)
            merged_df.to_csv(csv_path, index=False)
            logger.info(f"✅ CSV actualizado: {csv_path}")
        except Exception as e:
            logger.error(f"Error actualizando CSV: {e}")
            nuevos_df.to_csv(csv_path.with_name(f"backup_{csv_path.name}"), index=False)
    else:
        nuevos_df.to_csv(csv_path, index=False)
        logger.info(f"✅ CSV creado: {csv_path}")


def _update_duckdb_metrics(rows: list, resumen_csv_name: str):
    nuevos = pd.DataFrame(rows)
    try:
        with duckdb.connect(str(config.DB_MODELS_TRAIN_PATH)) as con:
            con.execute("CREATE OR REPLACE TEMP VIEW nuevos_data AS SELECT * FROM nuevos;")
            TABLE_NAME = _resumen_table_name(resumen_csv_name)
            con.execute(f"CREATE TABLE IF NOT EXISTS {TABLE_NAME} AS SELECT * FROM nuevos_data WHERE 1=0;")
            try:
                con.sql(f'ALTER TABLE {TABLE_NAME} ADD PRIMARY KEY (experimento, modelo);')
            except:
                pass

            ts = datetime.now().strftime("'%Y-%m-%d %H:%M:%S'")
            con.execute(f"""
                MERGE INTO {TABLE_NAME} AS t USING nuevos_data AS s
                ON t.experimento = s.experimento AND t.modelo = s.modelo
                WHEN MATCHED THEN UPDATE SET 
                    ganancia_max = s.ganancia_max, k_opt = s.k_opt, thr_opt = s.thr_opt, timestamp = {ts}
                WHEN NOT MATCHED THEN INSERT *;
            """)
    except Exception as e:
        logger.error(f"Error DuckDB metrics: {e}")


def _extract_rank_from_filename(filename: str) -> int:
    """Extrae el rank N del nombre de archivo 'lgb_topN_seed_XXXX.txt'"""
    # Busca el patrón 'top' seguido de digitos
    match = re.search(r"top(\d+)_", filename)
    if match:
        return int(match.group(1))
    return -1


# ==============================================================================
#  FUNCIONES PRINCIPALES
# ==============================================================================

def train_model(study, X_train, y_train, weights, k,
                base_study_name: str,
                mes,
                save_root, seeds, logger,
                selected_ranks: list = None):  # <--- NUEVO ARGUMENTO

    if isinstance(X_train, pl.DataFrame): X_train = X_train.to_pandas()
    X_train = _coerce_object_cols(X_train)

    df_trials = study.trials_dataframe()

    # Ordenar trials por valor (descendente) para determinar el Rank real
    df_trials = df_trials.sort_values("value", ascending=False).reset_index(drop=True)
    # Asignar columna de Rank (1-based)
    df_trials["rank_interno"] = df_trials.index + 1

    # Filtrar trials
    if selected_ranks and len(selected_ranks) > 0:
        logger.info(f"🎯 Seleccionando SOLAMENTE los modelos con Rank: {selected_ranks}")
        target_trials = df_trials[df_trials["rank_interno"].isin(selected_ranks)]
        if target_trials.empty:
            logger.warning("⚠️ Ninguno de los ranks seleccionados existe en el estudio de Optuna.")
    else:
        logger.info(f"🏆 Seleccionando los Top {k} modelos.")
        target_trials = df_trials.head(k)

    # Validar columna iteraciones
    col_best_iter = "user_attrs_mean_best_iter"
    if col_best_iter not in df_trials.columns:
        col_best_iter = "user_attrs_best_iter" if "user_attrs_best_iter" in df_trials.columns else None

    if not col_best_iter:
        raise KeyError("No se encontró columna de iteraciones en Optuna.")

    number_to_trial = {t.number: t for t in study.trials}
    train_data = lgb.Dataset(X_train, label=y_train, weight=weights)

    save_dir = Path(save_root) / str(base_study_name) / str(mes)
    save_dir.mkdir(parents=True, exist_ok=True)

    final_params = {
        'objective': 'binary', 'boosting_type': 'gbdt', 'feature_pre_filter': False,
        'metric': 'None', 'max_bin': 31, 'verbosity': -1, 'n_jobs': -1
    }

    resumen_rows = []

    for _, row in target_trials.iterrows():
        top_rank = int(row["rank_interno"])  # Usamos el rank calculado
        trial_num = int(row["number"])
        trial_val = float(row["value"])
        trial_obj = number_to_trial[trial_num]

        raw_iter = trial_obj.user_attrs.get("mean_best_iter") or trial_obj.user_attrs.get("best_iter")
        if raw_iter is None:
            logger.error(f"Falta num_boost_round en trial {trial_num}");
            continue
        num_boost_round = int(raw_iter)

        params = final_params.copy()
        params.update(trial_obj.params)

        for seed in seeds:
            try:
                # El nombre del archivo SIEMPRE lleva el rank para identificarlo
                file = f"lgb_top{top_rank}_seed_{int(seed)}.txt"
                check_path = save_dir / file

                if check_path.exists():
                    logger.info(f"✅ Modelo {file} ya existe. Saltando entrenamiento.")
                else:
                    logger.info(f"🏋️ Entrenando {file} (Rank {top_rank}, Rounds: {num_boost_round})")
                    params.update({'seed': seed})
                    model = lgb.train(params=params, train_set=train_data, num_boost_round=num_boost_round)
                    model.save_model(str(check_path))

                # Siempre agregamos a metadata
                resumen_rows.append({
                    "top_rank": top_rank,
                    "trial_number": trial_num,
                    "trial_value": trial_val,
                    "best_iter": num_boost_round,
                    "seed": int(seed),
                    "model_path": str(check_path),
                    "params": json.dumps(params)
                })

            except Exception as e:
                logger.error(f"Error entrenando {file}: {e}")

    # Metadata a DuckDB
    meta = pd.DataFrame(resumen_rows)
    if not meta.empty:
        TABLE_NAME = f"{base_study_name}_train"
        try:
            with duckdb.connect(str(config.DB_MODELS_TRAIN_PATH)) as con:
                con.execute(f"CREATE TABLE IF NOT EXISTS {TABLE_NAME} AS SELECT * FROM meta WHERE 1=0;")
                try:
                    con.sql(f'ALTER TABLE {TABLE_NAME} ADD PRIMARY KEY ("model_path");')
                except:
                    pass
                con.execute(f"INSERT INTO {TABLE_NAME} SELECT * FROM meta ON CONFLICT (model_path) DO NOTHING;")
        except Exception as e:
            logger.error(f"Error DuckDB metadata: {e}")

    return meta


def calculo_curvas_ganancia(Xif, y_test_class, dir_model_opt,
                            experimento_key: str,
                            resumen_csv_name: str = "resumen_ganancias.csv",
                            selected_ranks: list = None):  # <--- NUEVO ARGUMENTO
    piso_envios = 4000
    techo_envios = 20000
    LINEWIDTH = 1.5
    ALPHA_MODELOS = 0.5
    ALPHA_PROM = 1.0
    LS_PROM = '-'

    if isinstance(Xif, pl.DataFrame): Xif = Xif.to_pandas()
    Xif = _coerce_object_cols(Xif)

    meses_titulo = ", ".join(
        [str(m) for m in sorted(Xif['foto_mes'].unique())]) if 'foto_mes' in Xif.columns else "Desconocido"
    meses_str = "_".join(
        [str(m) for m in sorted(Xif['foto_mes'].unique())]) if 'foto_mes' in Xif.columns else "meses_desconocidos"

    curvas = []
    mejores_cortes = {}
    probs_ordenadas = []
    y_predicciones = []

    dir_model_opt = Path(dir_model_opt)
    all_files = sorted([p for p in dir_model_opt.glob("*.txt")] + [p for p in dir_model_opt.glob("*.bin")])

    # --- FILTRO POR SELECTED_RANKS ---
    modelos_validos = []
    for p in all_files:
        if not p.exists() or p.stat().st_size == 0: continue

        # Lógica de filtrado
        if selected_ranks and len(selected_ranks) > 0:
            rank = _extract_rank_from_filename(p.name)
            if rank not in selected_ranks:
                # Si el rank extraido no está en la lista deseada, saltamos este archivo
                continue

        try:
            _ = lgb.Booster(model_file=str(p))
            modelos_validos.append(p)
        except:
            pass

    if not modelos_validos:
        msg = f"No hay modelos válidos en {dir_model_opt}"
        if selected_ranks: msg += f" filtrando por ranks {selected_ranks}"
        raise RuntimeError(msg)

    logger.info(f"📈 Calculando curvas para {len(modelos_validos)} modelos seleccionados.")

    nombre_exp_limpio = Path(experimento_key).name
    folder_name = f"{meses_str}_{nombre_exp_limpio}"
    target_folder = dir_model_opt / "curvas_de_complejidad" / folder_name
    target_folder.mkdir(parents=True, exist_ok=True)

    resumen_rows = []
    ganancia = np.where(y_test_class == "BAJA+2", config.GANANCIA_ACIERTO, 0) - \
               np.where(y_test_class != "BAJA+2", config.COSTO_ESTIMULO, 0)

    plt.figure(figsize=(12, 7))

    for model_file in modelos_validos:
        nombre = model_file.stem
        model = lgb.Booster(model_file=str(model_file))
        Xif_filtered = Xif[model.feature_name()]
        y_pred = model.predict(Xif_filtered)

        df_pred_export = Xif[['numero_de_cliente', 'foto_mes']].copy()
        df_pred_export['y_pred'] = y_pred
        y_predicciones.append(df_pred_export)

        idx = np.argsort(y_pred)[::-1]
        y_pred_sorted = y_pred[idx]
        gan_cum = np.cumsum(ganancia[idx])
        curva_segmento = gan_cum[piso_envios:techo_envios]
        curvas.append(curva_segmento)
        probs_ordenadas.append(y_pred_sorted)

        x_envios = np.arange(piso_envios, piso_envios + len(curva_segmento))
        argmax_local = int(np.argmax(curva_segmento))
        k_mejor = int(piso_envios + argmax_local)
        ganancia_max = float(curva_segmento[argmax_local])
        thr_opt = float(y_pred_sorted[max(k_mejor - 1, 0)])

        mejores_cortes[nombre] = (k_mejor, ganancia_max, thr_opt)

        resumen_rows.append({
            "experimento": experimento_key, "modelo": nombre, "k_opt": int(k_mejor),
            "ganancia_max": float(ganancia_max), "thr_opt": float(thr_opt),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        p = plt.plot(x_envios, curva_segmento, label=nombre, linewidth=LINEWIDTH, alpha=ALPHA_MODELOS)

        # Gráfico individual
        fig_temp = plt.figure(figsize=(10, 6))
        plt.plot(x_envios, curva_segmento, color=p[0].get_color(), label=nombre)
        plt.axvline(x=k_mejor, color='gray', linestyle='--')
        plt.title(f'{nombre} - {meses_titulo}')
        plt.grid(True, alpha=0.3)
        plt.savefig(target_folder / f"{nombre}.jpg", dpi=100)
        plt.close(fig_temp)

    # Gráfico conjunto
    plt.figure(1)
    curvas_np = np.vstack(curvas)
    promedio = curvas_np.mean(axis=0)
    x_envios = np.arange(piso_envios, piso_envios + len(promedio))
    x_k_mejor = int(piso_envios + np.argmax(promedio))
    x_ganancia_max = float(np.max(promedio))

    plt.plot(x_envios, promedio, linewidth=2.5, linestyle=LS_PROM, color='black', alpha=ALPHA_PROM, label='Promedio',
             zorder=10)
    plt.axvline(x=x_k_mejor, color='black', linestyle=':')
    plt.title(f'Curvas de Ganancia - {meses_titulo} \nExp: {nombre_exp_limpio}')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    # Nombre de archivo dinámico si es selección manual
    suffix = "_seleccion_manual" if selected_ranks else ""
    plt.savefig(target_folder / f"curva_ganancia_conjunta{suffix}.jpg", dpi=300)
    plt.close()

    # Guardado seguro
    full_csv_path = dir_model_opt / resumen_csv_name
    _update_csv_metrics(resumen_rows, full_csv_path)
    _update_duckdb_metrics(resumen_rows, resumen_csv_name)

    return y_predicciones, curvas, mejores_cortes


def graficar_curva_ensamble_soft(Xif, y_test_class, dir_model_opt,
                                 experimento_key: str,
                                 folder_name: str = None,
                                 resumen_csv_name: str = "resumen_ganancias.csv",
                                 selected_ranks: list = None):  # <--- NUEVO ARGUMENTO

    piso_envios, techo_envios = 4000, 20000
    if isinstance(Xif, pl.DataFrame): Xif = Xif.to_pandas()
    Xif = _coerce_object_cols(Xif)

    meses_titulo = ", ".join(
        [str(m) for m in sorted(Xif['foto_mes'].unique())]) if 'foto_mes' in Xif.columns else "Desconocido"
    dir_model_opt = Path(dir_model_opt)
    all_files = sorted([p for p in dir_model_opt.glob("*.txt")] + [p for p in dir_model_opt.glob("*.bin")])

    # --- FILTRO POR SELECTED_RANKS ---
    modelos_validos = []
    for p in all_files:
        if not p.exists() or p.stat().st_size == 0: continue

        if selected_ranks and len(selected_ranks) > 0:
            rank = _extract_rank_from_filename(p.name)
            if rank not in selected_ranks: continue

        modelos_validos.append(p)

    if not modelos_validos: raise RuntimeError("No hay modelos para ensamble.")

    logger.info(f"🤖 Calculando Ensamble Soft con {len(modelos_validos)} modelos.")
    list_y_preds = []

    for model_file in modelos_validos:
        try:
            model = lgb.Booster(model_file=str(model_file))
            list_y_preds.append(model.predict(Xif.reindex(columns=model.feature_name(), fill_value=0)))
        except Exception as e:
            logger.warning(f"Error {model_file.name}: {e}")

    y_ensamble_prob = np.mean(np.vstack(list_y_preds), axis=0)

    ganancia_real = np.where(y_test_class == "BAJA+2", config.GANANCIA_ACIERTO, 0) - \
                    np.where(y_test_class != "BAJA+2", config.COSTO_ESTIMULO, 0)

    idx_sorted = np.argsort(y_ensamble_prob)[::-1]
    curva_segmento = np.cumsum(ganancia_real[idx_sorted])[piso_envios:techo_envios]

    idx_max = np.argmax(curva_segmento)
    k_mejor = int(piso_envios + idx_max)
    ganancia_max = float(curva_segmento[idx_max])
    thr_opt = float(y_ensamble_prob[idx_sorted][k_mejor - 1])

    # Plot
    plt.figure(figsize=(10, 6))
    x_envios = np.arange(piso_envios, piso_envios + len(curva_segmento))
    label_ens = f'Ensamble Soft (Rank: {selected_ranks})' if selected_ranks else f'Ensamble Soft (Todos)'

    plt.plot(x_envios, curva_segmento, color='green', linewidth=2.0, label=label_ens)
    plt.axvline(x=k_mejor, color='green', linestyle='--', label=f'Corte: {k_mejor}')
    plt.title(f'Ensamble Soft - {experimento_key} \n {meses_titulo}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    out_dir = dir_model_opt / "curvas_de_complejidad"
    if folder_name: out_dir /= folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = "_seleccion_manual" if selected_ranks else ""
    plt.savefig(out_dir / f"curva_ensamble_soft_{experimento_key}{suffix}.jpg", dpi=300)
    plt.close()

    resumen_rows = [{
        "experimento": experimento_key,
        "modelo": f"ENSAMBLE_SOFT_VOTING{suffix}",  # Nombre distinto para no pisar el ensamble full
        "k_opt": int(k_mejor), "ganancia_max": float(ganancia_max),
        "thr_opt": float(thr_opt), "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }]

    full_csv_path = dir_model_opt / resumen_csv_name
    _update_csv_metrics(resumen_rows, full_csv_path)
    _update_duckdb_metrics(resumen_rows, resumen_csv_name)

    return k_mejor, ganancia_max, thr_opt


def pred_ensamble_desde_experimentos(
        Xif: pd.DataFrame,
        experiments: list[dict],
        k: int,  # Se mantiene por compatibilidad, pero la lógica prioriza archivos en disco
        output_path,
        output_basename: str,
        resumen_csv_name: str = "resumen_ganancias.csv",  # Se ignora
        selected_ranks: list = None,
        cut_off_rank: int = 10000
) -> pd.DataFrame:
    logger.info("============== INICIO PREDICCIÓN DIRECTA (DISCO) ==============")

    # 1. Preparar Datos
    if isinstance(Xif, pl.DataFrame):
        Xif = Xif.to_pandas()
    Xif = _coerce_object_cols(Xif)

    df_accum = Xif[['numero_de_cliente', 'foto_mes']].copy()
    df_accum['sum_proba'] = 0.0
    modelos_contados = 0

    # 2. Configurar Filtros
    if not selected_ranks:
        selected_ranks = []  # Asegurar lista vacía
        logger.info("📋 Modo: Tomar TODOS los modelos encontrados en el directorio.")
    else:
        logger.info(f"🎯 Modo: Filtrar modelos con Ranks {selected_ranks}")

    # 3. Iterar por Experimentos (Carpetas)
    for i, item in enumerate(experiments):
        base_dir = Path(item["dir"])
        experimento = item["experimento"]

        logger.info(f"--- Escaneando carpeta: {base_dir} ---")

        if not base_dir.exists():
            logger.warning(f"⚠️ El directorio no existe: {base_dir}")
            continue

        # BUSQUEDA DIRECTA DE ARCHIVOS
        # Buscamos patrones .txt y .bin que coincidan con la estructura de nombres
        files = sorted([f for f in base_dir.glob("lgb_top*_seed_*.txt")] +
                       [f for f in base_dir.glob("lgb_top*_seed_*.bin")])

        if not files:
            logger.warning(f"❌ No se encontraron modelos en {base_dir}")
            continue

        logger.info(f"    📂 Archivos detectados: {len(files)}")

        # 4. Iterar sobre archivos encontrados
        for m_path in files:
            try:
                # Extraer Rank
                rank = _extract_rank_from_filename(m_path.name)

                # --- LOGICA DE FILTRADO ---
                # Si selected_ranks NO está vacío, y el rank no está en la lista -> saltar
                if selected_ranks and (rank not in selected_ranks):
                    continue

                # Si selected_ranks ESTÁ vacío, tomamos todo (no hay 'continue')

                # PREDICCIÓN
                bst = lgb.Booster(model_file=str(m_path))

                # Validar features (evita crash si faltan columnas)
                model_feats = bst.feature_name()

                # Predicción optimizada (solo columnas necesarias)
                y_prob = bst.predict(Xif[model_feats])

                df_accum['sum_proba'] += y_prob
                modelos_contados += 1

                # Feedback visual ocasional
                if modelos_contados % 5 == 0:
                    logger.info(f"       -> Procesados {modelos_contados} modelos...")

            except Exception as e:
                logger.error(f"    ❌ Error corrupto/lectura en {m_path.name}: {e}")

    # 5. Validación Final
    if modelos_contados == 0:
        logger.error("❌ ERROR FATAL: No se generaron predicciones (0 modelos usados).")
        return pd.DataFrame()

    logger.info(f"✅ Total final de modelos promediados: {modelos_contados}")

    # 6. Promedio y Ranking
    # Promedio
    df_accum['prob_promedio'] = df_accum['sum_proba'] / modelos_contados

    # Ordenar por probabilidad (Mayor a menor)
    df_accum = df_accum.sort_values('prob_promedio', ascending=False).reset_index(drop=True)

    # Aplicar Corte
    df_accum['Predicted'] = 0
    corte_real = min(cut_off_rank, len(df_accum))

    if corte_real > 0:
        # Asignar 1 a los top K
        df_accum.iloc[:corte_real, df_accum.columns.get_loc('Predicted')] = 1

        # Logs de control
        prob_corte = df_accum.iloc[corte_real - 1]['prob_promedio']
        prob_max = df_accum.iloc[0]['prob_promedio']
        logger.info(f"📊 Prob. Máxima: {prob_max:.4f}")
        logger.info(f"✂️ Corte en envíos {corte_real}. Prob. de Corte: {prob_corte:.5f}")

    # 7. Guardado
    suffix = "_seleccion_manual" if selected_ranks else ""
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{output_basename}{suffix}.csv"

    df_final = df_accum[['numero_de_cliente', 'Predicted']]
    df_final.to_csv(out_file, index=False)

    logger.info(f"💾 Archivo de predicción guardado: {out_file}")

    return df_final


# Mantenemos esta función por compatibilidad con el escenario simple,
# pero aplicando la misma lógica directa.
def pred_ensamble_modelos(
        Xif: pd.DataFrame,
        dir_model_opt: str | Path,
        experimento: str,
        k: int,
        output_path,
        resumen_csv_name: str = "resumen_ganancias.csv",  # Ignorado
        selected_ranks: list = None,
        cut_off_rank: int = 10000
) -> pd.DataFrame:
    # Envolvemos en una lista de dicts y reutilizamos la lógica robusta de arriba
    experiments_payload = [{
        "dir": str(dir_model_opt),
        "experimento": experimento
    }]

    return pred_ensamble_desde_experimentos(
        Xif=Xif,
        experiments=experiments_payload,
        k=k,
        output_path=output_path,
        output_basename=experimento,
        selected_ranks=selected_ranks,
        cut_off_rank=cut_off_rank
    )