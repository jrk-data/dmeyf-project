from typing import Tuple, Dict, Any , Iterable, Union

import numpy as np
import optuna
import lightgbm as lgb
import polars as pl
from .config import (
    STUDY_NAME,
    N_TRIALS, N_STARTUP_TRIALS,
    NFOLD, EARLY_STOPPING_ROUNDS, MES_TRAIN, MES_TEST, SEEDS, GANANCIA_ACIERTO, COSTO_ESTIMULO
)
from src.gain_functions import lgb_gan_eval
from logging import getLogger
from lightgbm import early_stopping
import src.config as config
from src.create_seeds import create_seed
from src.utils import _coerce_object_cols


logger = getLogger(__name__)


def lgb_gan_eval_individual(y_pred, data):
    logger.info("Calculo ganancia INDIVIDUAL")
    weight = data.get_weight()
    ganancia = np.where(weight == 1.00002, GANANCIA_ACIERTO, 0) - np.where(weight < 1.00002, COSTO_ESTIMULO, 0)
    logger.info(f"ganancia : {ganancia}")
    ganancia = ganancia[np.argsort(y_pred)[::-1]]
    logger.info(f"ganancia sorted : {ganancia}")
    ganancia = np.cumsum(ganancia)
    logger.info(f"ganancia acumulada : {ganancia}")
    #con polars
    # df_eval = pl.DataFrame({"y_pred":y_pred , "weight":weight})
    # df_sorted = df_eval.sort("y_pred" , descending=True)
    # df_sorted = df_sorted.with_columns([pl.when(pl.col('weight') == 1.00002).then(GANANCIA).otherwise(-ESTIMULO).alias('ganancia_individual')])
    # df_sorted = df_sorted.with_columns([pl.col('ganancia_individual').cum_sum().alias('ganancia_acumulada')])
    # ganancia_maxima = df_sorted.select(pl.col('ganancia_acumulada').max()).item()
    # id_gan_max = df_sorted["ganancia_acumulada"].arg_max()
    # media_meseta = df_sorted.slice(id_gan_max-500, 1000)['ganancia_acumulada'].mean()
    logger.info(f"ganancia max acumulada : {np.max(ganancia)}")
    logger.info(f"cliente optimo : {np.argmax(ganancia)}")
    return 'gan_eval', np.max(ganancia) , True

def lgb_gan_eval_ensamble(y_pred , data):
    logger.info("Calculo ganancia ENSAMBLE")
    weight = data.get_weight()
    ganancia =np.where(weight == 1.00002 , GANANCIA_ACIERTO, 0) - np.where(weight < 1.00002 , COSTO_ESTIMULO ,0)
    logger.info(f"ganancia : {ganancia}")
    ganancia_sorted = ganancia[np.argsort(y_pred)[::-1]]
    logger.info(f"ganancia sorted : {ganancia_sorted}")
    ganancia_acumulada = np.cumsum(ganancia_sorted)
    logger.info(f"ganancia acumulada : {ganancia_acumulada}")
    ganancia_max = np.max(ganancia_acumulada)
    idx_max_gan = np.argmax(ganancia_acumulada)
    logger.info(f"ganancia max acumulada : {ganancia_max}")
    logger.info(f"cliente oprimo : {idx_max_gan}")
    ganancia_media_meseta = np.mean(ganancia_acumulada[idx_max_gan-500 : idx_max_gan+500])
    logger.info(f"ganancia media meseta : {ganancia_media_meseta}")
    return ganancia_media_meseta ,idx_max_gan ,ganancia_max


def run_study(X_train, y_train, semillas, SEED,w_train, matching_categorical_features: None
              ,storage_optuna, study_name_optuna, optimizar = False):
    """Crea/ejecuta el estudio Optuna (TPE bayesiano)."""

    logger.info(f"Comienzo optimizacion hiperp binario: {study_name_optuna}")
    if isinstance(X_train, pl.DataFrame):
        X_train = X_train.to_pandas()
    if isinstance(y_train, pl.Series):
        y_train_binaria = y_train.to_pandas()
    if isinstance(w_train, pl.Series):
        w_train = w_train.to_pandas()

    logger.info("Se cargaron X_train, y_train_binaria y w_train")
    num_meses = len(MES_TRAIN)
    f_val = X_train["foto_mes"] == config.MES_VALIDACION

    X_val = X_train.loc[f_val]
    y_val_binaria = y_train_binaria[X_val.index]
    w_val = w_train[X_val.index]

    X_train = X_train.loc[~f_val]
    y_train_binaria = y_train_binaria[X_train.index]
    w_train = w_train[X_train.index]

    logger.info(f"Meses train en bayesiana : {X_train['foto_mes'].unique()}")
    logger.info(f"Meses validacion en bayesiana : {X_val['foto_mes'].unique()}")

    def objective(trial: optuna.Trial) -> float:
        num_leaves = trial.suggest_int('num_leaves', 3, 3000)
        learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.3)  # mas bajo, más iteraciones necesita
        min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 1, 20000)
        feature_fraction = trial.suggest_float('feature_fraction', 0.1, 0.9)
        bagging_fraction = trial.suggest_float('bagging_fraction', 0.1, 0.9)
        lambbda_1 = trial.suggest_float('lambda_1', 0.0, 10.0)
        lambda_2 = trial.suggest_float('lambda_2', 0.0, 10.0)

        params = {
            'objective': 'binary',
            'metric': 'None',
            'boosting_type': 'gbdt',
            'first_metric_only': True,
            'boost_from_average': True,
            'feature_pre_filter': False,
            'max_bin': 31,
            'num_leaves': num_leaves,
            'learning_rate': learning_rate,
            'min_data_in_leaf': min_data_in_leaf,
            'feature_fraction': feature_fraction,
            'bagging_fraction': bagging_fraction,
            'lambda_l1': lambbda_1,
            'lambda_l2': lambda_2,
            'bagging_freq': 1,
            'seed': SEED,
            'verbose': -1,
            'extra_trees': True
        }
        train_data = lgb.Dataset(X_train,
                                 label=y_train,  # eligir la clase
                                 weight=w_train,
                                 # categorical_feature=matching_categorical_features
                                 )
        val_data = lgb.Dataset(X_val,label=y_val_binaria,weight=w_val)
        y_preds=[]
        best_iters=[]
        for semilla in semillas:
            params['seed'] = semilla
            model_i = lgb.train(
                    params=params,
                    train_set=train_data,
                    num_boost_round= config.N_BOOSTS,
                    valid_sets=[val_data],
                    valid_names=['valid'],
                    feval=lgb_gan_eval_individual,
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=int(50 + 5/learning_rate), verbose=False),
                        lgb.log_evaluation(period=200),
                        ]
                    )
            y_pred_i = model_i.predict(X_val,num_iteration=model_i.best_iteration)
            y_preds.append(y_pred_i)
            best_iters.append(model_i.best_iteration)

        y_preds_matrix = np.vstack(y_preds)
        y_pred_ensamble = np.mean(y_preds_matrix , axis=0)
        ganancia_media_meseta , cliente_optimo,ganancia_max = lgb_gan_eval_ensamble(y_pred_ensamble , val_data)
        best_iter_promedio =  np.mean(best_iters)

        #guardar_iteracion(trial,ganancia_media_meseta,cliente_optimo,ganancia_max,best_iter_promedio,y_preds_matrix,best_iters,name,fecha,semillas)

        return float(ganancia_media_meseta) * num_meses

    sampler = optuna.samplers.TPESampler(seed=123, n_startup_trials=N_STARTUP_TRIALS)
    study = optuna.create_study(
        study_name=study_name_optuna,
        direction="maximize",
        sampler=sampler,
        storage=storage_optuna,
        load_if_exists=True
    )
    #study.optimize(objective(SEED,X_train, y_train,w_train, w_train, None), n_trials=N_TRIALS, show_progress_bar=True)

    if optimizar == True:
        study.optimize(
            objective,
            n_trials=N_TRIALS,
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


def run_study_cv(X_train, y_train, SEED,w_train, matching_categorical_features: None
              ,storage_optuna, study_name_optuna, optimizar = False):
    """Crea/ejecuta el estudio Optuna (TPE bayesiano)."""

    if isinstance(X_train, pl.DataFrame):
        X_train = X_train.to_pandas()

    # 🔧 Arreglo clave:
    X_train = _coerce_object_cols(X_train)


    def objective(trial: optuna.Trial) -> float:
        num_leaves = trial.suggest_int('num_leaves', 3, 3000)
        learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.3)  # mas bajo, más iteraciones necesita
        min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 1, 20000)
        feature_fraction = trial.suggest_float('feature_fraction', 0.1, 0.9)
        bagging_fraction = trial.suggest_float('bagging_fraction', 0.1, 0.9)
        lambbda_1 = trial.suggest_float('lambda_1', 0.0, 10.0)
        lambda_2 = trial.suggest_float('lambda_2', 0.0, 10.0)

        params = {
            'objective': 'binary',
            'metric': 'None',
            'boosting_type': 'gbdt',
            'first_metric_only': True,
            'boost_from_average': True,
            'feature_pre_filter': False,
            'max_bin': 31,
            'num_leaves': num_leaves,
            'learning_rate': learning_rate,
            'min_data_in_leaf': min_data_in_leaf,
            'feature_fraction': feature_fraction,
            'bagging_fraction': bagging_fraction,
            'lambda_l1': lambbda_1,
            'lambda_l2': lambda_2,
            'bagging_freq': 1,
            'seed': config.SEED,
            'verbose': -1,
            'extra_trees': True
        }


        train_data = lgb.Dataset(X_train,
                                 label=y_train,  # eligir la clase
                                 weight=w_train,
                                 # categorical_feature=matching_categorical_features
                                 )
        cv_results = lgb.cv(
            params,
            train_data,
            num_boost_round=1000,  # modificar, subit y subir... y descomentar la línea inferior
            # early_stopping_rounds= int(50 + 5 / learning_rate),
            feval=lgb_gan_eval,
            stratified=True,
            nfold= config.NFOLD,
            seed=SEED,
            callbacks=[early_stopping(stopping_rounds=int(50 + 5 / learning_rate), verbose=False)
                       ]
        )
        max_gan = max(cv_results['valid gan_eval-mean'])
        best_iter = cv_results['valid gan_eval-mean'].index(max_gan) + 1

        # Guardamos cual es la mejor iteración del modelo
        trial.set_user_attr("best_iter", best_iter)

        return max_gan * config.NFOLD

    sampler = optuna.samplers.TPESampler(seed=config.SEED, n_startup_trials=N_STARTUP_TRIALS)
    study = optuna.create_study(
        study_name=study_name_optuna,
        direction="maximize",
        sampler=sampler,
        storage=storage_optuna,
        load_if_exists=True
    )
    #study.optimize(objective(SEED,X_train, y_train,w_train, w_train, None), n_trials=N_TRIALS, show_progress_bar=True)

    if optimizar == True:
        study.optimize(
            objective,
            n_trials=N_TRIALS,
            show_progress_bar=True
        )

    return study