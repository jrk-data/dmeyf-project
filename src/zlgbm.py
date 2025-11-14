"""
R_to_py: Conversión completa de workflow R a Python
Workflow con 3 etapas: Train (201901-202102) -> Test (202104, 202106) -> Final (202108)

Autor: Data Scientist Junior
Fecha: 2025-11-13
"""

import pandas as pd
import numpy as np
import polars as pl
import lightgbm as lgb
import gc
import os
import logging
import json
from datetime import datetime
from sklearn.utils import resample
import warnings
from src.utils import _coerce_object_cols

warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURACIÓN - DEFINIR TODO AQUÍ
# ============================================================================

# -----------------------------
# Parámetros del negocio
# -----------------------------
COSTO_ESTIMULO = 20000
GANANCIA_ACIERTO = 780000

# -----------------------------
# Experimento
# -----------------------------
EXPERIMENTO = "apo-506"
SEMILLA_PRIMIGENIA = 102191
APO = 1
KSEMILLERIO = 1

# -----------------------------
# Dataset
# -----------------------------
DATASET_PATH = "~/datasets/competencia_02_crudo.csv.gz"

# -----------------------------
# Periodos (Estructura de 3 etapas)
# -----------------------------
# TRAIN: Todos los meses desde 201901 hasta 202102
FOTO_MES_TRAIN_INICIO = 201901
FOTO_MES_TRAIN_FIN = 202102

# TEST: Dos meses de validación
FOTO_MES_TEST_1 = 202104
FOTO_MES_TEST_2 = 202106

# FINAL: Predicción final
FOTO_MES_FINAL = 202108

# -----------------------------
# Semillas
# -----------------------------
SEMILLAS_EXPERIMENTO = 30   # Para testing/optimización
SEMILLAS_FINAL = 100        # Para predicción final

# -----------------------------
# Feature Engineering
# -----------------------------
QCANARITOS = 5  # Cantidad de variables aleatorias (canaritos)

# Lags y Deltas
FEATURE_ENGINEERING_LAGS = True  # Activar/desactivar lags y deltas
LAGS_ORDEN = [1, 2]  # Órdenes de lags a crear (1 y 2)
# Si LAGS_ORDEN = [1, 2, 3] creará lag1, lag2, lag3 y delta1, delta2, delta3

# -----------------------------
# Undersampling
# -----------------------------
UNDERSAMPLING = True
UNDERSAMPLING_RATIO = 0.1  # Proporción de clase mayoritaria a mantener (0.1 = 10%)
# Si es 0.1, mantenemos solo 10% de CONTINUA y todos los BAJA+1 y BAJA+2

# -----------------------------
# LightGBM - Parámetros (estilo zlightgbm)
# -----------------------------
MIN_DATA_IN_LEAF = 2000
LEARNING_RATE = 1.0
GRADIENT_BOUND = 0.01
NUM_LEAVES = 300
FEATURE_FRACTION = 0.8
BAGGING_FRACTION = 0.8
BAGGING_FREQ = 5
MAX_BIN = 31  # Reducir para ahorrar memoria
NUM_BOOST_ROUND = 1000
EARLY_STOPPING_ROUNDS = 200

# -----------------------------
# Cortes para evaluar
# -----------------------------
CORTES = [8000, 8500, 9000, 9500, 10000, 10500, 11000, 
          11500, 12000, 12500, 13000, 13500, 14000, 14500, 15000]

# -----------------------------
# Rutas
# -----------------------------
BASE_PATH = "./exp"


# ============================================================================
# FUNCIÓN DE GANANCIA CON POLARS
# ============================================================================

def calcular_ganancia(y_pred, y_true):
    """
    Calcula la ganancia máxima acumulada ordenando las predicciones de mayor a menor.

    Args:
        y_true: Valores reales (0 o 1)
        y_pred: Predicciones (probabilidades o scores continuos) -> DEBE SER CONTINUO

    Returns:
        tuple[float, np.ndarray]: Ganancia máxima acumulada y la serie acumulada completa.
    """

    def _to_polars_series(
            values, name: str, dtype: pl.DataType | None = None
    ) -> pl.Series:
        """Convierte valores a serie de Polars"""
        if isinstance(values, pl.Series):
            series = pl.Series(name, values.to_list())
        elif isinstance(values, pd.Series):
            series = pl.Series(name, values.to_list())
        else:
            if not isinstance(values, (list, tuple)):
                values = list(values)
            series = pl.Series(name, values)

        if dtype is not None:
            try:
                series = series.cast(dtype, strict=False)
            except pl.ComputeError:
                series = series.cast(pl.Float64, strict=False)

        return series

    # Convertir a series de Polars
    y_true_series = _to_polars_series(y_true, "y_true", dtype=pl.Float64)
    y_pred_series = _to_polars_series(y_pred, "y_pred_proba", dtype=pl.Float64)

    # Validaciones
    if y_true_series.is_empty() or y_pred_series.is_empty():
        logger.debug("Ganancia calculada: 0 (datasets vacíos)")
        return 0.0, np.array([], dtype=float)

    if y_true_series.len() != y_pred_series.len():
        raise ValueError("y_true y y_pred deben tener la misma longitud")

    # Calcular ganancia
    acumulado_df = (
        pl.DataFrame({"y_true": y_true_series, "y_pred_proba": y_pred_series})
        .sort("y_pred_proba", descending=True)
        .with_columns([
            pl.when(pl.col("y_true").round(0) == 1.0)
            .then(pl.lit(GANANCIA_ACIERTO, dtype=pl.Int64))
            .otherwise(pl.lit(-COSTO_ESTIMULO, dtype=pl.Int64))
            .alias("ganancia_individual")
        ])
        .with_columns([
            pl.col("ganancia_individual")
            .cum_sum()
            .alias("ganancia_acumulada")
        ])
    )

    ganancia_acumulada_series = acumulado_df["ganancia_acumulada"]
    ganancia_total = ganancia_acumulada_series.max()

    if ganancia_total > 2_147_483_647:
        ganancia_total = float(ganancia_total)

    ganancias_acumuladas = ganancia_acumulada_series.to_numpy()

    logger.info(f"Ganancia calculada: {ganancia_total:,.0f}")

    return ganancia_total, ganancias_acumuladas


def ganancia_lgb_binary(y_pred, y_true):
    """
    Función de ganancia para LightGBM en clasificación binaria.
    Compatible con callbacks de LightGBM (feval).

    Args:
        y_pred: Predicciones DE PROBABILIDAD (ya que LightGBM devuelve prob. para binario)
        y_true: Dataset de LightGBM con labels verdaderos

    Returns:
        tuple: (eval_name, eval_result, is_higher_better)
    """
    y_true_labels = y_true.get_label()
    # Pasamos las predicciones continuas (y_pred) a la función de ganancia
    ganancia_total, _ = calcular_ganancia(y_pred=y_pred, y_true=y_true_labels)
    # Nota: la implementación del curso usa un umbral fijo 0.025 aquí para feval,
    # pero el cálculo correcto de la ganancia máxima no necesita un umbral fijo.
    return "ganancia", ganancia_total, True


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def limpiar_memoria():
    """Limpia la memoria RAM"""
    gc.collect()


def crear_directorio(path):
    """Crea un directorio si no existe"""
    os.makedirs(path, exist_ok=True)


def crear_directorio_experimento():
    """
    Crea directorio del experimento con timestamp
    Retorna la ruta completa
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_path = os.path.join(BASE_PATH, f"{EXPERIMENTO}_{timestamp}")
    crear_directorio(exp_path)
    logger.info(f"Directorio del experimento: {exp_path}")
    return exp_path


def guardar_configuracion(exp_path):
    """
    Guarda todos los parámetros configurados en un archivo JSON
    """
    config = {
        "metadata": {
            "experimento": EXPERIMENTO,
            "timestamp": datetime.now().isoformat(),
            "fecha_ejecucion": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "negocio": {
            "costo_estimulo": COSTO_ESTIMULO,
            "ganancia_acierto": GANANCIA_ACIERTO
        },
        "dataset": {
            "path": DATASET_PATH
        },
        "periodos": {
            "train_inicio": FOTO_MES_TRAIN_INICIO,
            "train_fin": FOTO_MES_TRAIN_FIN,
            "test_1": FOTO_MES_TEST_1,
            "test_2": FOTO_MES_TEST_2,
            "final": FOTO_MES_FINAL
        },
        "semillas": {
            "semilla_primigenia": SEMILLA_PRIMIGENIA,
            "semillas_experimento": SEMILLAS_EXPERIMENTO,
            "semillas_final": SEMILLAS_FINAL,
            "ksemillerio": KSEMILLERIO
        },
        "feature_engineering": {
            "qcanaritos": QCANARITOS,
            "lags_enabled": FEATURE_ENGINEERING_LAGS,
            "lags_orden": LAGS_ORDEN
        },
        "undersampling": {
            "enabled": UNDERSAMPLING,
            "ratio": UNDERSAMPLING_RATIO
        },
        "lightgbm": {
            "min_data_in_leaf": MIN_DATA_IN_LEAF,
            "learning_rate": LEARNING_RATE,
            "gradient_bound": GRADIENT_BOUND,
            "num_leaves": NUM_LEAVES,
            "feature_fraction": FEATURE_FRACTION,
            "bagging_fraction": BAGGING_FRACTION,
            "bagging_freq": BAGGING_FREQ,
            "max_bin": MAX_BIN,
            "num_boost_round": NUM_BOOST_ROUND,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS
        },
        "cortes": CORTES,
        "apo": APO
    }
    
    config_path = os.path.join(exp_path, "configuracion.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    
    logger.info(f"Configuración guardada en: {config_path}")
    return config_path


def generar_semillas(semilla_base, cantidad):
    """Genera una lista de semillas determinísticas"""
    np.random.seed(semilla_base)
    return np.random.randint(1, 2**31-1, size=cantidad).tolist()


def generar_rango_meses(inicio, fin):
    """
    Genera lista de meses en formato YYYYMM entre inicio y fin.
    Ejemplo: generar_rango_meses(201901, 201903) -> [201901, 201902, 201903]
    """
    meses = []
    anio_ini = inicio // 100
    mes_ini = inicio % 100
    anio_fin = fin // 100
    mes_fin = fin % 100
    
    anio_actual = anio_ini
    mes_actual = mes_ini
    
    while (anio_actual * 100 + mes_actual) <= (anio_fin * 100 + mes_fin):
        meses.append(anio_actual * 100 + mes_actual)
        mes_actual += 1
        if mes_actual > 12:
            mes_actual = 1
            anio_actual += 1
    
    return meses


# ============================================================================
# PREPROCESAMIENTO
# ============================================================================

def calcular_clase_ternaria(df):
    """
    Calcula la clase_ternaria según la permanencia del cliente
    """
    logger.info("Calculando clase_ternaria...")
    
    df['periodo0'] = (df['foto_mes'] // 100) * 12 + (df['foto_mes'] % 100)
    df = df.sort_values(['numero_de_cliente', 'periodo0']).reset_index(drop=True)
    
    df['periodo1'] = df.groupby('numero_de_cliente')['periodo0'].shift(-1)
    df['periodo2'] = df.groupby('numero_de_cliente')['periodo0'].shift(-2)
    
    periodo_ultimo = df['periodo0'].max()
    periodo_anteultimo = periodo_ultimo - 1
    
    df['clase_ternaria'] = 'CONTINUA'
    
    mask_baja1 = (df['periodo0'] < periodo_ultimo) & \
                 (df['periodo1'].isna() | (df['periodo0'] + 1 < df['periodo1']))
    df.loc[mask_baja1, 'clase_ternaria'] = 'BAJA+1'
    
    mask_baja2 = (df['periodo0'] < periodo_anteultimo) & \
                 (df['periodo0'] + 1 == df['periodo1']) & \
                 (df['periodo2'].isna() | (df['periodo0'] + 2 < df['periodo2']))
    df.loc[mask_baja2, 'clase_ternaria'] = 'BAJA+2'
    
    df = df.drop(['periodo0', 'periodo1', 'periodo2'], axis=1)
    
    logger.info("Distribución de clases por periodo:")
    dist = df.groupby(['foto_mes', 'clase_ternaria']).size().reset_index(name='count')
    for _, row in dist.head(20).iterrows():
        logger.info(f"  {row['foto_mes']}: {row['clase_ternaria']} = {row['count']}")
    
    return df


def agregar_canaritos(df, num_canaritos=None, semilla=None):
    """
    Agrega variables aleatorias (canaritos) AL PRINCIPIO del dataset usando Polars.
    Mantiene los nombres canarito1, canarito2, ... para no romper la lógica existente.
    """
    if num_canaritos is None:
        num_canaritos = QCANARITOS
    if semilla is None:
        semilla = SEMILLA_PRIMIGENIA

    if num_canaritos <= 0 or len(df) == 0:
        logger.info("No se agregan canaritos (num_canaritos <= 0 o df vacío)")
        return df

    logger.info(f"Agregando {num_canaritos} canaritos con Polars (al principio)...")

    # Guardamos el orden original de columnas
    original_cols = list(df.columns)

    # Convertimos a Polars

    df_pl = pl.from_pandas(df)

    # Generamos matriz aleatoria
    np.random.seed(semilla)
    rand_matrix = np.random.rand(df_pl.height, num_canaritos)

    # Nombres de canaritos consistentes con el resto del código
    canary_cols = [f"canarito{i+1}" for i in range(num_canaritos)]

    # Agregamos columnas de canaritos
    for i, name in enumerate(canary_cols):
        df_pl = df_pl.with_columns(pl.Series(name, rand_matrix[:, i]))

    # Ponemos los canaritos AL PRINCIPIO
    df_pl = df_pl.select(canary_cols + original_cols)

    df_result = df_pl.to_pandas()

    logger.info(f"  ✓ {num_canaritos} canaritos agregados al principio")
    logger.info(f"  ✓ Primeras columnas: {list(df_result.columns[:min(10, len(df_result.columns))])}")

    return df_result


# ============================================================================
# FEATURE ENGINEERING: LAGS Y DELTAS
# ============================================================================

def agregar_lags_y_deltas(df, ordenes=None):
    """
    Agrega lags y deltas (diferencias) de variables históricas
    
    Args:
        df: DataFrame con los datos
        ordenes: Lista de órdenes de lags a crear (ej: [1, 2])
    
    Returns:
        DataFrame con lags y deltas agregados
    """
    if ordenes is None:
        ordenes = LAGS_ORDEN
    
    if not FEATURE_ENGINEERING_LAGS:
        logger.info("Feature engineering de lags/deltas desactivado")
        return df
    
    logger.info(f"Agregando lags y deltas (órdenes: {ordenes})...")
    inicio = datetime.now()
    
    # Ordenar por cliente y periodo
    df = df.sort_values(['numero_de_cliente', 'foto_mes']).reset_index(drop=True)
    
    # Identificar columnas lagueables
    # Todo es lagueable MENOS: numero_de_cliente, foto_mes, clase_ternaria, canaritos
    cols_excluir = ['numero_de_cliente', 'foto_mes', 'clase_ternaria']
    cols_excluir += [f'canarito{i}' for i in range(1, QCANARITOS + 1)]
    
    cols_lagueables = [col for col in df.columns if col not in cols_excluir]
    
    logger.info(f"  Columnas lagueables: {len(cols_lagueables)}")
    logger.info(f"  Órdenes de lag: {ordenes}")
    
    # Crear lags para cada orden
    for orden in ordenes:
        logger.info(f"  Creando lags de orden {orden}...")
        
        # Crear lags usando groupby + shift
        for col in cols_lagueables:
            nombre_lag = f'{col}_lag{orden}'
            df[nombre_lag] = df.groupby('numero_de_cliente')[col].shift(orden)
        
        # Limpiar memoria después de cada orden
        limpiar_memoria()
    
    # Crear deltas (diferencias)
    logger.info(f"  Creando deltas...")
    for orden in ordenes:
        for col in cols_lagueables:
            nombre_delta = f'{col}_delta{orden}'
            nombre_lag = f'{col}_lag{orden}'
            
            # Delta = valor actual - valor lag
            df[nombre_delta] = df[col] - df[nombre_lag]
        
        # Limpiar memoria después de cada orden
        limpiar_memoria()
    
    # Contar features creados
    n_lags = len(cols_lagueables) * len(ordenes)
    n_deltas = len(cols_lagueables) * len(ordenes)
    n_total = n_lags + n_deltas
    
    duracion = datetime.now() - inicio
    logger.info(f"  ✓ Features creados: {n_total} ({n_lags} lags + {n_deltas} deltas)")
    logger.info(f"  ✓ Duración: {duracion}")
    logger.info(f"  ✓ Shape final: {df.shape}")
    
    return df


# ============================================================================
# UNDERSAMPLING
# ============================================================================

def aplicar_undersampling(df, ratio=None, semilla=None):
    """
    Aplica undersampling a la clase mayoritaria (CONTINUA)
    
    Args:
        df: DataFrame con columna 'clase_ternaria'
        ratio: Proporción de clase mayoritaria a mantener (ej: 0.1 = 10%)
        semilla: Semilla para reproducibilidad
    
    Returns:
        DataFrame con undersampling aplicado
    """
    if ratio is None:
        ratio = UNDERSAMPLING_RATIO
    if semilla is None:
        semilla = SEMILLA_PRIMIGENIA
    
    logger.info(f"Aplicando undersampling (ratio={ratio})...")
    
    # Separar clases
    df_continua = df[df['clase_ternaria'] == 'CONTINUA']
    df_baja1 = df[df['clase_ternaria'] == 'BAJA+1']
    df_baja2 = df[df['clase_ternaria'] == 'BAJA+2']
    
    logger.info(f"  Antes - CONTINUA: {len(df_continua):,}, BAJA+1: {len(df_baja1):,}, BAJA+2: {len(df_baja2):,}")
    
    # Submuestrear CONTINUA
    n_continua_mantener = int(len(df_continua) * ratio)
    df_continua_sampled = resample(
        df_continua,
        n_samples=n_continua_mantener,
        replace=False,
        random_state=semilla
    )
    
    # Combinar
    df_balanced = pd.concat([df_continua_sampled, df_baja1, df_baja2], ignore_index=True)
    df_balanced = df_balanced.sample(frac=1, random_state=semilla).reset_index(drop=True)
    
    logger.info(f"  Después - CONTINUA: {len(df_continua_sampled):,}, BAJA+1: {len(df_baja1):,}, BAJA+2: {len(df_baja2):,}")
    logger.info(f"  Total registros: {len(df):,} -> {len(df_balanced):,}")
    
    return df_balanced


# ============================================================================
# PREPARACIÓN DE DATOS (3 ETAPAS)
# ============================================================================

def preparar_datos_train_test_final(df, meses_train, mes_test1, mes_test2, mes_final):
    """
    Prepara los datos para el workflow de 3 etapas.

    Args:
        df: DataFrame completo
        meses_train: lista de meses (int) para entrenamiento
        mes_test1: mes (int) para test 1
        mes_test2: mes (int) para test 2
        mes_final: mes (int) para predicción final

    Returns:
        df_train, df_test1, df_test2, df_final, feature_cols
    """
    logger.info("Preparando datos para workflow de 3 etapas...")

    # Aseguramos listas / ints
    meses_train = list(meses_train)

    logger.info(f"Meses de entrenamiento usados: {meses_train}")
    logger.info(f"Mes test1: {mes_test1} | Mes test2: {mes_test2} | Mes final: {mes_final}")

    # Filtrar datos
    df_train = df[df['foto_mes'].isin(meses_train)].copy()
    df_test1 = df[df['foto_mes'] == mes_test1].copy()
    df_test2 = df[df['foto_mes'] == mes_test2].copy()
    df_final = df[df['foto_mes'] == mes_final].copy()

    logger.info(f"Train: {len(df_train):,} registros ({len(meses_train)} meses)")
    logger.info(f"Test 1 ({mes_test1}): {len(df_test1):,} registros")
    logger.info(f"Test 2 ({mes_test2}): {len(df_test2):,} registros")
    logger.info(f"Final ({mes_final}): {len(df_final):,} registros")

    # Aplicar undersampling solo a train
    if UNDERSAMPLING:
        df_train = aplicar_undersampling(df_train)

    # Definir columnas de features
    cols_excluir = ['numero_de_cliente', 'foto_mes', 'clase_ternaria']
    feature_cols = [col for col in df_train.columns if col not in cols_excluir]
    logger.info(f"Features: {len(feature_cols)}")

    return df_train, df_test1, df_test2, df_final, feature_cols


# ============================================================================
# ENTRENAMIENTO CON LIGHTGBM (estilo zlightgbm)
# ============================================================================

def entrenar_lgbm(X_train, y_train, X_val, y_val, semilla, usar_ganancia=False):
    """
    Entrena un modelo LightGBM con parámetros estilo zlightgbm
    """
    # Parámetros base
    lgbm_params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'learning_rate': LEARNING_RATE,
        'num_leaves': NUM_LEAVES,
        'feature_fraction': FEATURE_FRACTION,
        'bagging_fraction': BAGGING_FRACTION,
        'bagging_freq': BAGGING_FREQ,
        'min_data_in_leaf': MIN_DATA_IN_LEAF,
        'max_bin': MAX_BIN,
        'verbose': -1,
        'seed': semilla,
        'force_row_wise': True,
    }
    
    if GRADIENT_BOUND is not None:
        lgbm_params['gradient_bound'] = GRADIENT_BOUND
    
    # Crear datasets
    train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=True)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, free_raw_data=True)
    
    # Métrica personalizada
    feval = ganancia_lgb_binary if usar_ganancia else None
    
    # Entrenar
    modelo = lgb.train(
        lgbm_params,
        train_data,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[val_data],
        valid_names=['valid'],
        feval=feval,
        callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)]
    )
    
    return modelo

def guardar_modelo_individual(modelo, semilla, exp_path):
    """
    Guarda un modelo individual del ensemble.

    Args:
        modelo: Modelo LightGBM entrenado
        semilla: Semilla usada para entrenar
        exp_path: Ruta del experimento
    """
    modelos_dir = os.path.join(exp_path, "modelos")
    crear_directorio(modelos_dir)

    model_txt_path = os.path.join(modelos_dir, f"modelo_seed{semilla}.txt")
    modelo.save_model(model_txt_path)

    logger.debug(f"  Modelo seed={semilla} guardado en: {model_txt_path}")

# ============================================================================
# ETAPA 2: TESTING (202104 y 202106)
# ============================================================================

def etapa_testing(df_train, df_test1, df_test2, feature_cols, exp_path):
    """
    Etapa 2: Testing con datos de 202104 y 202106
    Entrena con train, predice en test1 y test2, calcula ganancias por corte

    Returns:
        df_testing, mejor_corte, pred_test1, pred_test2
        (pred_* ahora son detallados)
    """
    logger.info("="*80)
    logger.info("ETAPA 2: TESTING (202104 y 202106)")
    logger.info("="*80)

    # Preparar datos
    X_train = df_train[feature_cols]
    y_train = (df_train['clase_ternaria'] == 'BAJA+2').astype(int)

    X_test1 = df_test1[feature_cols]
    y_test1 = (df_test1['clase_ternaria'] == 'BAJA+2').astype(int)

    X_test2 = df_test2[feature_cols]
    y_test2 = (df_test2['clase_ternaria'] == 'BAJA+2').astype(int)

    logger.info(f"Train - Registros: {len(X_train):,}, BAJA+2: {y_train.sum():,} ({y_train.mean()*100:.2f}%)")
    logger.info(f"Test 1 - Registros: {len(X_test1):,}, BAJA+2: {y_test1.sum():,} ({y_test1.mean()*100:.2f}%)")
    logger.info(f"Test 2 - Registros: {len(X_test2):,}, BAJA+2: {y_test2.sum():,} ({y_test2.mean()*100:.2f}%)")

    # Semillas
    semillas = generar_semillas(SEMILLA_PRIMIGENIA, SEMILLAS_EXPERIMENTO)

    # Matrices de ganancia
    matriz_gan_test1 = np.zeros((SEMILLAS_EXPERIMENTO, len(CORTES)))
    matriz_gan_test2 = np.zeros((SEMILLAS_EXPERIMENTO, len(CORTES)))

    # Acumuladores de probas
    pred_acum_test1 = np.zeros(len(X_test1))
    pred_acum_test2 = np.zeros(len(X_test2))

    modelos_entrenados = []

    for idx_sem, semilla in enumerate(semillas):
        if (idx_sem + 1) % 5 == 0 or idx_sem == 0:
            logger.info(f"  Testing con semilla {idx_sem + 1}/{SEMILLAS_EXPERIMENTO}...")

        # Mini validación para early stopping
        n_val = min(5000, len(X_train) // 10)
        indices = np.arange(len(X_train))
        np.random.seed(semilla)
        np.random.shuffle(indices)
        idx_val = indices[:n_val]

        X_val_mini = X_train.iloc[idx_val]
        y_val_mini = y_train.iloc[idx_val]

        modelo = entrenar_lgbm(
            X_train, y_train,
            X_val_mini, y_val_mini,
            semilla,
            usar_ganancia=False
        )

        # Guardar modelo individual
        guardar_modelo_individual(modelo, semilla, exp_path)
        modelos_entrenados.append(modelo)

        # Predicciones
        y_pred_test1 = modelo.predict(X_test1)
        y_pred_test2 = modelo.predict(X_test2)

        pred_acum_test1 += y_pred_test1
        pred_acum_test2 += y_pred_test2

        df_pred_test1 = pd.DataFrame({
            'y_true': y_test1.values,
            'y_pred': y_pred_test1
        }).sort_values('y_pred', ascending=False).reset_index(drop=True)

        df_pred_test2 = pd.DataFrame({
            'y_true': y_test2.values,
            'y_pred': y_pred_test2
        }).sort_values('y_pred', ascending=False).reset_index(drop=True)

        # Ganancias por corte
        for idx_c, corte in enumerate(CORTES):
            # Test 1
            n_envios = min(corte, len(df_pred_test1))
            df_pred_test1['pred_binary'] = 0
            df_pred_test1.iloc[:n_envios, df_pred_test1.columns.get_loc('pred_binary')] = 1
            g1, _ = calcular_ganancia(
                y_pred=df_pred_test1['pred_binary'].values,
                y_true=df_pred_test1['y_true'].values
            )
            matriz_gan_test1[idx_sem, idx_c] = g1

            # Test 2
            n_envios = min(corte, len(df_pred_test2))
            df_pred_test2['pred_binary'] = 0
            df_pred_test2.iloc[:n_envios, df_pred_test2.columns.get_loc('pred_binary')] = 1
            g2, _ = calcular_ganancia(
                y_pred=df_pred_test2['pred_binary'].values,
                y_true=df_pred_test2['y_true'].values
            )
            matriz_gan_test2[idx_sem, idx_c] = g2

        del modelo
        limpiar_memoria()

    # Promedio de probas
    prob_test1 = pred_acum_test1 / SEMILLAS_EXPERIMENTO
    prob_test2 = pred_acum_test2 / SEMILLAS_EXPERIMENTO

    # DataFrames DETALLADOS de test
    pred_test1_detallado = pd.DataFrame({
        'numero_de_cliente': df_test1['numero_de_cliente'].values,
        'foto_mes': df_test1['foto_mes'].values,
        'clase_ternaria': df_test1['clase_ternaria'].values,
        'y_true': y_test1.values,
        'prob': prob_test1
    }).sort_values('prob', ascending=False).reset_index(drop=True)

    pred_test2_detallado = pd.DataFrame({
        'numero_de_cliente': df_test2['numero_de_cliente'].values,
        'foto_mes': df_test2['foto_mes'].values,
        'clase_ternaria': df_test2['clase_ternaria'].values,
        'y_true': y_test2.values,
        'prob': prob_test2
    }).sort_values('prob', ascending=False).reset_index(drop=True)

    # Estadísticas de ganancia
    gan_test1_prom = matriz_gan_test1.mean(axis=0)
    gan_test1_std = matriz_gan_test1.std(axis=0)
    gan_test2_prom = matriz_gan_test2.mean(axis=0)
    gan_test2_std = matriz_gan_test2.std(axis=0)
    gan_promedio = (gan_test1_prom + gan_test2_prom) / 2

    df_testing = pd.DataFrame({
        'corte': CORTES,
        'gan_test1_prom': gan_test1_prom,
        'gan_test1_std': gan_test1_std,
        'gan_test2_prom': gan_test2_prom,
        'gan_test2_std': gan_test2_std,
        'gan_promedio': gan_promedio
    })

    idx_mejor = df_testing['gan_promedio'].idxmax()
    mejor_corte = int(df_testing.loc[idx_mejor, 'corte'])
    logger.info(f"\nMejor corte (ganancia promedio): {mejor_corte}")

    # Guardar resúmenes y matrices
    testing_path = os.path.join(exp_path, "evaluacion_testing.csv")
    df_testing.to_csv(testing_path, index=False)

    matriz_path1 = os.path.join(exp_path, "matriz_test1.csv")
    pd.DataFrame(matriz_gan_test1, columns=[f'corte_{c}' for c in CORTES]).to_csv(matriz_path1, index=False)

    matriz_path2 = os.path.join(exp_path, "matriz_test2.csv")
    pd.DataFrame(matriz_gan_test2, columns=[f'corte_{c}' for c in CORTES]).to_csv(matriz_path2, index=False)

    # NUEVO: guardar predicciones detalladas
    pred_test1_path = os.path.join(exp_path, "predicciones_test1_detallado.csv")
    pred_test1_detallado.to_csv(pred_test1_path, index=False)

    pred_test2_path = os.path.join(exp_path, "predicciones_test2_detallado.csv")
    pred_test2_detallado.to_csv(pred_test2_path, index=False)

    return df_testing, mejor_corte, pred_test1_detallado, pred_test2_detallado



# ============================================================================
# ETAPA 3: PREDICCIÓN FINAL (202108)
# ============================================================================

def etapa_final(df_train, df_final, feature_cols, exp_path):
    """
    Etapa 3: Predicción final
    Entrena con train, predice en final, genera predicciones promediadas.
    """
    logger.info("="*80)
    logger.info("ETAPA 3: PREDICCIÓN FINAL")
    logger.info("="*80)

    X_train = df_train[feature_cols]
    y_train = (df_train['clase_ternaria'] == 'BAJA+2').astype(int)
    X_final = df_final[feature_cols]

    logger.info(f"Train: {len(X_train):,} registros")
    logger.info(f"Final ({FOTO_MES_FINAL}): {len(X_final):,} registros")

    semillas = generar_semillas(SEMILLA_PRIMIGENIA, SEMILLAS_FINAL)

    pred_acum_final = np.zeros(len(X_final))

    for idx_sem, semilla in enumerate(semillas):
        if (idx_sem + 1) % 10 == 0 or idx_sem == 0:
            logger.info(f"  Final con semilla {idx_sem + 1}/{SEMILLAS_FINAL}...")

        n_val = min(5000, len(X_train) // 10)
        indices = np.arange(len(X_train))
        np.random.seed(semilla)
        np.random.shuffle(indices)
        idx_val = indices[:n_val]

        X_val_mini = X_train.iloc[idx_val]
        y_val_mini = y_train.iloc[idx_val]

        modelo = entrenar_lgbm(
            X_train, y_train,
            X_val_mini, y_val_mini,
            semilla,
            usar_ganancia=False
        )

        # Guardar modelo individual
        guardar_modelo_individual(modelo, semilla, exp_path)

        pred = modelo.predict(X_final)
        pred_acum_final += pred

        del modelo
        limpiar_memoria()

    prob_final = pred_acum_final / SEMILLAS_FINAL

    resultado = pd.DataFrame({
        'numero_de_cliente': df_final['numero_de_cliente'].values,
        'foto_mes': df_final['foto_mes'].values,
        'prob': prob_final
    }).sort_values('prob', ascending=False).reset_index(drop=True)

    logger.info(f"\nPredicciones generadas: {len(resultado):,}")
    logger.info(f"Top 10 probabilidades: {resultado['prob'].head(10).values}")

    pred_detallado_path = os.path.join(exp_path, f"predicciones_final_detallado_{FOTO_MES_FINAL}.csv")
    resultado.to_csv(pred_detallado_path, index=False)
    logger.info(f"  ✓ predicciones_final_detallado_{FOTO_MES_FINAL}.csv ({len(resultado):,} registros)")

    return resultado


# ============================================================================
# GENERACIÓN DE SUBMISSIONS
# ============================================================================

def generar_submissions(predicciones, exp_path, cortes=None, sufijo=""):
    """
    Genera archivos de submission para cada corte.
    MEJORA:
      - Archivo Kaggle simple (numero_de_cliente, Predicted)
      - Archivo detallado (prob + Predicted + foto_mes + resto de columnas)

    Args:
        predicciones: DataFrame con predicciones ordenadas
        exp_path: Ruta del directorio del experimento
        cortes: Lista de cortes a evaluar
        sufijo: Sufijo para diferenciar archivos (ej: "_test_202104", "_final_202108")

    Returns:
        DataFrame con resumen de cortes (corte, envios, archivos generados)
    """
    if cortes is None:
        cortes = CORTES

    # Log del mes (si existe)
    if 'foto_mes' in predicciones.columns:
        mes = predicciones['foto_mes'].iloc[0]
        logger.info(f"Generando {len(cortes)} submissions para mes {mes}...")
    else:
        logger.info(f"Generando {len(cortes)} submissions{sufijo}...")

    kaggle_dir = os.path.join(exp_path, "kaggle")
    crear_directorio(kaggle_dir)

    resultados = []

    for corte in cortes:
        pred_temp = predicciones.copy()
        pred_temp['Predicted'] = (pred_temp.index < corte).astype(int)

        # 1) Archivo Kaggle SIMPLE
        filename_simple = f"KA{EXPERIMENTO}_{corte}{sufijo}.csv"
        filepath_simple = os.path.join(kaggle_dir, filename_simple)
        pred_temp[['numero_de_cliente', 'Predicted']].to_csv(filepath_simple, index=False)

        # 2) Archivo DETALLADO
        filename_detallado = f"KA{EXPERIMENTO}_{corte}{sufijo}_detallado.csv"
        filepath_detallado = os.path.join(kaggle_dir, filename_detallado)
        pred_temp.to_csv(filepath_detallado, index=False)

        envios = int(pred_temp['Predicted'].sum())
        resultados.append({
            'corte': corte,
            'envios': envios,
            'archivo_kaggle': filename_simple,
            'archivo_detallado': filename_detallado,
            'sufijo': sufijo
        })

        if corte % 2500 == 0 or corte == cortes[0]:
            logger.info(f"  Corte {corte}: {envios} envíos")
            logger.info(f"    → {filename_simple} (Kaggle)")
            logger.info(f"    → {filename_detallado} (Detallado)")

    df_resultados = pd.DataFrame(resultados)
    resultados_path = os.path.join(exp_path, f"resultados_cortes{sufijo}.csv")
    df_resultados.to_csv(resultados_path, index=False)

    logger.info(f"\nArchivos generados:")
    logger.info(f"  ✓ {len(cortes)} submissions Kaggle (simples) en: {kaggle_dir}")
    logger.info(f"  ✓ {len(cortes)} archivos detallados en: {kaggle_dir}")
    logger.info(f"  ✓ resultados_cortes{sufijo}.csv")

    return df_resultados


# ============================================================================
# WORKFLOW PRINCIPAL
# ============================================================================

def main(df=None, meses_train=None, mes_test1=None, mes_test2=None, mes_final=None):

    """Función principal del workflow de 3 etapas"""
    print("="*80)
    print("R_to_py: Workflow Completo de 3 Etapas")
    print("="*80)
    
    inicio_ejecucion = datetime.now()
    
    logger.info(f"Inicio: {inicio_ejecucion.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Experimento: {EXPERIMENTO}")
    logger.info(f"Ganancia por acierto: ${GANANCIA_ACIERTO:,}")
    logger.info(f"Costo por estímulo: ${COSTO_ESTIMULO:,}")
    logger.info(f"Canaritos: {QCANARITOS}")
    logger.info(f"Lags y Deltas: {FEATURE_ENGINEERING_LAGS} (órdenes: {LAGS_ORDEN if FEATURE_ENGINEERING_LAGS else 'N/A'})")
    logger.info(f"Undersampling: {UNDERSAMPLING} (ratio={UNDERSAMPLING_RATIO})")
    print()
    
    # ========================================================================
    # CREAR DIRECTORIO CON TIMESTAMP Y GUARDAR CONFIGURACIÓN
    # ========================================================================
    exp_path = crear_directorio_experimento()
    config_path = guardar_configuracion(exp_path)
    logger.info(f"Configuración guardada en: {config_path}\n")
    
    # ========================================================================
    # PASO 1: CARGA Y PREPROCESAMIENTO
    # ========================================================================
    logger.info("="*80)
    logger.info("PASO 1: Carga y preprocesamiento")
    logger.info("="*80)

    if df is None:
        logger.info(f"Cargando dataset desde {DATASET_PATH}...")
        df = pd.read_csv(DATASET_PATH, compression='gzip')
        logger.info(f"Dataset cargado: {df.shape}")
    else:
        logger.info("Usando DataFrame provisto externamente (BigQuery / runner).")
        logger.info(f"Shape df externo: {df.shape}")
        # transformo df de polasrs a pandas
        df = df.to_pandas()
        # pasamos todos los campos a int
        df = _coerce_object_cols(df)

    # Calcular clase_ternaria solo si no existe
    if "clase_ternaria" not in df.columns or df["clase_ternaria"].isna().all():
        df = calcular_clase_ternaria(df)
    else:
        logger.info("Columna 'clase_ternaria' ya existe y contiene valores.")

    # Agregar canaritos
    df = agregar_canaritos(df, QCANARITOS)
    
    # Agregar lags y deltas
    if FEATURE_ENGINEERING_LAGS:
        df = agregar_lags_y_deltas(df, LAGS_ORDEN)
    
    limpiar_memoria()
    print()
    
    # ========================================================================
    # PASO 2: PREPARACIÓN DE DATOS (3 ETAPAS)
    # ========================================================================
    logger.info("="*80)
    logger.info("PASO 2: Preparación de datos (3 etapas)")
    logger.info("="*80)

    # -----------------------------
    # Definir meses efectivos
    # -----------------------------
    # TRAIN
    if meses_train is None:
        # Si no me pasan lista explícita, uso el rango global
        meses_train_efectivos = generar_rango_meses(
            FOTO_MES_TRAIN_INICIO,
            FOTO_MES_TRAIN_FIN
        )
        logger.info(
            f"Train (por rango global): {FOTO_MES_TRAIN_INICIO} a {FOTO_MES_TRAIN_FIN} "
            f"({len(meses_train_efectivos)} meses)"
        )
    else:
        meses_train_efectivos = list(meses_train)
        logger.info(
            f"Train (meses explícitos): {meses_train_efectivos} "
            f"({len(meses_train_efectivos)} meses)"
        )

    # TEST 1
    if mes_test1 is None:
        mes_test1_efectivo = FOTO_MES_TEST_1
    else:
        mes_test1_efectivo = mes_test1

    # TEST 2
    if mes_test2 is None:
        mes_test2_efectivo = FOTO_MES_TEST_2
    else:
        mes_test2_efectivo = mes_test2

    # FINAL
    if mes_final is None:
        mes_final_efectivo = FOTO_MES_FINAL
    else:
        mes_final_efectivo = mes_final

    logger.info(f"Test: {mes_test1_efectivo} y {mes_test2_efectivo}")
    logger.info(f"Final: {mes_final_efectivo}")
    print()

    # Ahora sí llamamos a la función de preparación con TODO explícito
    df_train, df_test1, df_test2, df_final, feature_cols = preparar_datos_train_test_final(
        df,
        meses_train=meses_train_efectivos,
        mes_test1=mes_test1_efectivo,
        mes_test2=mes_test2_efectivo,
        mes_final=mes_final_efectivo,
    )
    print()
    
    # ========================================================================
    # PASO 3: ETAPA TESTING (202104 y 202106)
    # ========================================================================
    df_testing, mejor_corte, pred_test1, pred_test2 = etapa_testing(
        df_train, df_test1, df_test2, feature_cols, exp_path
    )
    print()
    
    # ========================================================================
    # PASO 4: ETAPA FINAL (202108)
    # ========================================================================
    predicciones = etapa_final(df_train, df_final, feature_cols, exp_path)
    print()
    
    # ========================================================================
    # PASO 5: GENERACIÓN DE SUBMISSIONS
    # ========================================================================
    logger.info("="*80)
    logger.info("PASO 5: Generación de submissions")
    logger.info("="*80)
    
    df_resultados = generar_submissions(predicciones, exp_path)
    print()
    
    # ========================================================================
    # RESUMEN FINAL
    # ========================================================================
    fin_ejecucion = datetime.now()
    duracion = fin_ejecucion - inicio_ejecucion
    
    logger.info("="*80)
    logger.info("RESUMEN DE ARCHIVOS GENERADOS")
    logger.info("="*80)
    logger.info(f"\n📁 Directorio: {exp_path}\n")
    logger.info("📄 Configuración:")
    logger.info(f"  ✓ configuracion.json (parámetros del experimento)")
    logger.info("\n📊 Testing:")
    logger.info(f"  ✓ evaluacion_testing.csv (mejor corte: {mejor_corte})")
    logger.info(f"  ✓ matriz_test1.csv")
    logger.info(f"  ✓ matriz_test2.csv")
    logger.info(f"  ✓ predicciones_test1.csv ({len(pred_test1):,} registros)")
    logger.info(f"  ✓ predicciones_test2.csv ({len(pred_test2):,} registros)")
    logger.info("\n🎯 Final:")
    logger.info(f"  ✓ predicciones_final.csv ({len(predicciones):,} registros)")
    logger.info(f"  ✓ resultados_cortes.csv")
    logger.info(f"\n📤 Submissions:")
    logger.info(f"  ✓ kaggle/ ({len(CORTES)} archivos CSV)")
    
    logger.info("\n" + "="*80)
    logger.info("WORKFLOW COMPLETADO EXITOSAMENTE!")
    logger.info("="*80)
    logger.info(f"Mejor corte sugerido (testing): {mejor_corte}")
    logger.info(f"Inicio: {inicio_ejecucion.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Fin: {fin_ejecucion.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Duración: {duracion}")
    logger.info("="*80)
    
    return predicciones, df_testing, df_resultados, pred_test1, pred_test2, exp_path


if __name__ == "__main__":
    main()
