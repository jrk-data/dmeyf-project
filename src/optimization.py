from typing import Tuple, Dict, Any , Iterable, Union

import numpy as np
import optuna
import lightgbm as lgb
import polars as pl
import pandas as pd
from .config import (
    N_TRIALS, N_STARTUP_TRIALS,
     GANANCIA_ACIERTO, COSTO_ESTIMULO
)
from src.gain_functions import lgb_gan_eval
from logging import getLogger
from lightgbm import early_stopping
import src.config as config
from src.create_seeds import create_seed
from src.utils import _coerce_object_cols
import gc
from typing import List

logger = getLogger(__name__)


def lgb_gan_eval_individual(y_pred, data):
    """Métrica de evaluación individual (feval) para LightGBM. Retorna np.max(ganancia)."""

    #logger.info("Calculo ganancia INDIVIDUAL")
    # Usado para el entrenamiento, pero su score no es el que usa Optuna
    weight = data.get_weight()
    ganancia = np.where(weight == 1.00002, GANANCIA_ACIERTO, 0) - np.where(weight < 1.00002, COSTO_ESTIMULO, 0)
    #logger.info(f"ganancia : {ganancia}")
    ganancia = ganancia[np.argsort(y_pred)[::-1]]
    #logger.info(f"ganancia sorted : {ganancia}")
    ganancia_acumulada = np.cumsum(ganancia)
    #logger.info(f"ganancia acumulada : {ganancia_acumulada}")
    ##.info(f"ganancia max acumulada : {np.max(ganancia_acumulada)}")
    #logger.info(f"cliente optimo : {np.argmax(ganancia_acumulada)}")
    return 'gan_eval', np.max(ganancia_acumulada) , True

def lgb_gan_eval_ensamble(y_pred_ensamble: np.ndarray, val_data: lgb.Dataset) -> Tuple[float, int, float]:
    """Calcula la Ganancia Media en Meseta sobre las predicciones ensambladas."""
    logger.info("Calculo ganancia ENSAMBLE")
    # 1. Obtener los pesos del dataset de validación
    weight = val_data.get_weight()

    # 2. Asignar ganancia/costo por fila según su peso real (clase real)
    ganancia_individual = np.where(weight == 1.00002, GANANCIA_ACIERTO, 0) - \
                          np.where(weight < 1.00002, COSTO_ESTIMULO, 0)
    #logger.info(f"ganancia : {ganancia_individual}")

    # 3. Ordenar la ganancia individual según la probabilidad predicha (y_pred_ensamble)
    ganancia_sorted = ganancia_individual[np.argsort(y_pred_ensamble)[::-1]]
    #logger.info(f"ganancia sorted : {ganancia_sorted}")

    ganancia_acumulada = np.cumsum(ganancia_sorted)
    #logger.info(f"ganancia acumulada : {ganancia_acumulada}")
    ganancia_max = np.max(ganancia_acumulada)
    idx_max_gan = np.argmax(ganancia_acumulada)
    #logger.info(f"ganancia max acumulada : {ganancia_max}")
    #logger.info(f"cliente oprimo : {idx_max_gan}")

    # Ventana de Meseta (500 antes, 500 después del pico)
    inicio = max(0, idx_max_gan - 500)
    fin = min(len(ganancia_acumulada), idx_max_gan + 500)

    ganancia_media_meseta = np.mean(ganancia_acumulada[inicio: fin])
    #logger.debug(
        #f"Media Meseta: {ganancia_media_meseta:.0f}, Cliente óptimo: {idx_max_gan}, Ganancia Máx: {ganancia_max:.0f}")
    return ganancia_media_meseta, idx_max_gan, ganancia_max


def run_study(X_train: pd.DataFrame, y_train: pd.Series, semillas: List[int], SEED: int, w_train: pd.Series,
              matching_categorical_features: None, storage_optuna: str, study_name_optuna: str,
              optimizar: bool = False):
    # 0. Obtener constantes del módulo config (Asumimos que están disponibles)
    try:
        MES_VALIDACION = config.MES_VALIDACION
        EARLY_STOPPING_ROUNDS = config.EARLY_STOPPING_ROUNDS
        NUM_BOOST_ROUND_MAX = config.FIXED_PARAMS_REF['num_boost_round']
        N_TRIALS = config.N_TRIALS
    except AttributeError:
        logger.warning("Usando valores por defecto para MES_VALIDACION y EARLY_STOPPING_ROUNDS.")
        # Usamos los valores de las dependencias asumidas si config no funciona
        pass

    # Aseguramos tipos de entrada
    X_train = _coerce_object_cols(X_train)

    # 1. Separación Train/Validation Fija
    valid_months = MES_VALIDACION if isinstance(MES_VALIDACION, list) else [MES_VALIDACION]
    f_val = X_train["foto_mes"].isin(valid_months)

    # Validacion de seguridad: Si no hay datos de validacion, avisar.
    if f_val.sum() == 0:
        logger.warning(f"¡CUIDADO! Validation size es 0. Revisa que {MES_VALIDACION} esté incluido en tu X_train.")

    # Preparar máscaras booleanas para Numpy
    mask_val = f_val.values

    # Asegurar que y_train y w_train sean Arrays 1D planos para poder filtrarlos
    # (Vienen como numpy arrays desde main.py, por lo que no tienen .index ni .to_numpy)
    y_train_arr = np.array(y_train).ravel()
    w_train_arr = np.array(w_train).ravel()

    # Split de X (Pandas)
    X_val = X_train.loc[f_val].drop(columns=["foto_mes"])
    X_train = X_train.loc[~f_val].drop(columns=["foto_mes"])  # Sobreescribimos X_train con solo entrenamiento

    # Split de y, w (Numpy usando máscara booleana, NO indices)
    y_val_binaria = y_train_arr[mask_val]
    w_val = w_train_arr[mask_val]

    y_train_binaria = y_train_arr[~mask_val]
    w_train = w_train_arr[~mask_val]

    logger.info(f"Train/Val Split. Train size: {len(X_train)}, Validation size: {len(X_val)}")

    # Convertir a arrays de NumPy 2D (N, 1) para LightGBM
    # Nota: Ya son numpy arrays, solo hacemos reshape. No usamos .to_numpy()
    y_train_2d = y_train_binaria.reshape(-1, 1)
    w_train_2d = w_train.reshape(-1, 1)
    y_val_2d = y_val_binaria.reshape(-1, 1)
    w_val_2d = w_val.reshape(-1, 1)

    # Creación de Datasets LightGBM
    train_data = lgb.Dataset(X_train, label=y_train_2d.ravel(), weight=w_train_2d.ravel())
    val_data = lgb.Dataset(X_val, label=y_val_2d.ravel(), weight=w_val_2d.ravel())

    # --- 2. Función Objetivo con Early Stopping y Semillerío ---
    def objective(trial: optuna.Trial) -> float:

        params = config.FIXED_PARAMS_REF.copy()

        # 2.1 Sugerencia de Parámetros (Excluyendo num_boost_round)
        for name, info in config.SEARCHABLE_PARAMS_REF.items():
            ptype = info["type"]

            if ptype == "integer":
                params[name] = trial.suggest_int(name, info["lower"], info["upper"])
            elif ptype == "float":
                if info.get("log", False):
                    params[name] = trial.suggest_float(name, info["lower"], info["upper"], log=True)
                else:
                    params[name] = trial.suggest_float(name, info["lower"], info["upper"])
            elif ptype == "categorical":
                params[name] = trial.suggest_categorical(name, info["choices"])

        # 2.2 Semillerío (Entrenar N modelos con Early Stopping)
        y_preds = []
        best_iterations = []

        for exp_idx, semilla in enumerate(semillas):
            params['seed'] = semilla

            model_i = lgb.train(
                params=params,
                train_set=train_data,
                num_boost_round=NUM_BOOST_ROUND_MAX,  # Usamos el máximo de rondas (ej: 1000)
                valid_sets=[val_data],
                valid_names=['valid'],
                feval=lgb_gan_eval_individual,  # Usa la métrica custom para el ES
                callbacks=[
                    early_stopping(first_metric_only=False, stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=True),
                    lgb.log_evaluation(period=100, show_stdv=False)
                ]
            )

            # CRUCIAL: Predecir usando la mejor iteración determinada por Early Stopping
            best_iter = model_i.best_iteration
            y_pred_i = model_i.predict(X_val, num_iteration=best_iter)

            y_preds.append(y_pred_i)
            best_iterations.append(best_iter)

            del model_i
            gc.collect()

        # 2.3 Ensamblaje Final (Promedio de las N semillas)
        y_preds_matrix = np.vstack(y_preds)
        y_pred_ensamble = np.mean(y_preds_matrix, axis=0)

        # 2.4 Evaluación Final: Ganancia Media Meseta
        ganancia_media_meseta, idx_max, ganancia_max = lgb_gan_eval_ensamble(y_pred_ensamble, val_data)

        final_score = float(ganancia_media_meseta)

        # Registrar métricas útiles
        trial.set_user_attr("mean_best_iter", np.mean(best_iterations))
        trial.set_user_attr("mean_meseta_gain", float(ganancia_media_meseta))
        trial.set_user_attr("max_gain", float(ganancia_max))
        trial.set_user_attr("idx_max", int(idx_max))

        return final_score
    # --- 3. Creación y Ejecución del Estudio Optuna ---

    sampler = optuna.samplers.TPESampler(seed=SEED)  # Usamos SEED de config
    study = optuna.create_study(
        study_name=study_name_optuna,
        direction="maximize",
        sampler=sampler,
        storage=storage_optuna,
        load_if_exists=True
    )

    if optimizar == True:
        n_trials_realizados = len(study.trials)
        n_trials_total = config.N_TRIALS  # Asumiendo N_TRIALS existe en config

        if n_trials_realizados >= n_trials_total:
            logger.info(f"Ya hay {n_trials_realizados} trials realizados. Saltando optimización.")
        else:
            n_trials_faltantes = n_trials_total - n_trials_realizados
            logger.info(f"Ejecutando {n_trials_faltantes} trials adicionales.")

            study.optimize(
                objective,
                n_trials=n_trials_faltantes,
                show_progress_bar=True
            )

    return study

def train_final_model(X_trainval, y_trainval, best_params: Dict[str, Any], best_iter: int, SEED) -> lgb.Booster:
    """Entrena el modelo final con los mejores hiperparámetros e iteraciones."""
    params = best_params.copy()
    # limpieza de llaves no válidas en params (por si Optuna las dejó)
    for k in ("num_iterations",):
        params.pop(k, None)

    # aseguramos config básica
    params.update({
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "seed": SEED,
        "feature_pre_filter": False,
        "boost_from_average": True,
        "min_data_in_leaf": 3,
    })

    dtrain = lgb.Dataset(X_trainval, label=y_trainval, free_raw_data=False)
    model = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=best_iter
    )
    return model
