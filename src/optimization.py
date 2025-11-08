from typing import Tuple, Dict, Any , Iterable, Union

import numpy as np
import optuna
import lightgbm as lgb
import polars as pl
from .config import (
    STUDY_NAME,
    N_TRIALS, N_STARTUP_TRIALS,
    NFOLD, EARLY_STOPPING_ROUNDS, MES_TRAIN, MES_TEST,SEEDS
)
from src.gain_functions import lgb_gan_eval
from logging import getLogger
from lightgbm import early_stopping



logger = getLogger(__name__)






def run_study(X_train, y_train, SEED,w_train, matching_categorical_features: None
              ,storage_optuna, study_name_optuna, optimizar = False):
    """Crea/ejecuta el estudio Optuna (TPE bayesiano)."""

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
        cv_results = lgb.cv(
            params,
            train_data,
            num_boost_round=400,  # modificar, subit y subir... y descomentar la línea inferior
            # early_stopping_rounds= int(50 + 5 / learning_rate),
            feval=lgb_gan_eval,
            stratified=True,
            nfold=5,
            seed=SEED,
            callbacks=[early_stopping(stopping_rounds=int(50 + 5 / learning_rate), verbose=False)
                       ]
        )
        max_gan = max(cv_results['valid gan_eval-mean'])
        best_iter = cv_results['valid gan_eval-mean'].index(max_gan) + 1

        # Guardamos cual es la mejor iteración del modelo
        trial.set_user_attr("best_iter", best_iter)

        return max_gan * 5

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

