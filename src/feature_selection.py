import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
from logging import getLogger

logger = getLogger(__name__)


def perform_canaritos_selection(X_train: pl.DataFrame,
                                y_train: np.ndarray,
                                n_canaritos: int = 20,
                                seed: int = 42) -> list:
    """
    Recibe X_train (Polars) e y_train (Numpy) ya procesados por split_train_data.
    Agrega ruido, entrena y devuelve features sobrevivientes.
    """
    logger.info(f"🦆 Iniciando Canaritos con {n_canaritos} sondas...")

    # 1. Convertir a Pandas (LGBM maneja mejor Pandas nativo para nombres de columnas)
    # X_train viene de split_train_data como Polars DataFrame
    X_pd = X_train.to_pandas()

    # 2. Generar Canaritos
    np.random.seed(seed)

    # Detectar columnas numéricas para sacar rangos realistas
    numeric_cols = X_pd.select_dtypes(include=[np.number]).columns.tolist()

    canarito_names = []
    logger.info("Generando variables de ruido...")

    for i in range(n_canaritos):
        # Elegir columna random para imitar su escala
        ref_col = np.random.choice(numeric_cols)
        min_val = X_pd[ref_col].min()
        max_val = X_pd[ref_col].max()

        name = f"canarito_{i}"
        canarito_names.append(name)
        # Ruido uniforme
        X_pd[name] = np.random.uniform(min_val, max_val, size=len(X_pd))

    # 3. Entrenar LGBM Rápido
    # Parámetros robustos pero rápidos
    params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'verbosity': -1,
        'seed': seed,
        'n_jobs': -1
    }

    # Crear dataset LGBM
    dtrain = lgb.Dataset(X_pd, label=y_train)

    logger.info("Entrenando modelo sonda...")
    model = lgb.train(params, dtrain, num_boost_round=150)

    # 4. Feature Importance (Gain)
    importance = pd.DataFrame({
        'feature': model.feature_name(),
        'gain': model.feature_importance(importance_type='gain')
    }).sort_values(by='gain', ascending=False).reset_index(drop=True)

    # 5. Corte
    # Buscamos el mejor rank de un canarito
    canaritos_ranks = importance[importance['feature'].isin(canarito_names)].index

    features_col = X_train.columns  # Originales

    if len(canaritos_ranks) > 0:
        best_rank = canaritos_ranks.min()
        best_canarito = importance.iloc[best_rank]['feature']
        gain_canarito = importance.iloc[best_rank]['gain']

        logger.info(f"✂️ CORTE: Mejor canarito '{best_canarito}' en pos #{best_rank + 1} (Gain: {gain_canarito:.2f})")

        # Seleccionamos todo lo que esté ARRIBA del canarito
        selected = importance.iloc[:best_rank]['feature'].tolist()

        # Limpiar por si quedó algún canarito (improbable si rank es min, pero por seguridad)
        selected = [f for f in selected if f not in canarito_names]
    else:
        logger.warning("⚠️ Ningún canarito entró en el ranking. Se conservan todas.")
        selected = features_col

    drop_count = len(features_col) - len(selected)
    logger.info(f"✅ Variables Finales: {len(selected)} (Eliminadas: {drop_count})")

    return selected