"""
R_to_py: Conversión completa de workflow R a Python
Workflow con 3 etapas: Train (201901-202102) -> Test (202104, 202106) -> Final (202108)

MEJORAS:
- Gráficos de ganancia para análisis visual
- Dos archivos por corte: Predicted (0/1) y Probabilidad
- Submissions separadas por mes

Autor: Data Scientist Junior
Fecha: 2025-11-14
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
warnings.filterwarnings('ignore')

# Imports para gráficos
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Para que funcione sin display

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
SEMILLA_PRIMIGENIA = 550007
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
FOTO_MES_TRAIN_INICIO = 202001
FOTO_MES_TRAIN_FIN = 202102

# TEST: Dos meses de validación
FOTO_MES_TEST_1 = 202104
FOTO_MES_TEST_2 = 202106

# FINAL: Predicción final
FOTO_MES_FINAL = 202108

# -----------------------------
# Semillas
# -----------------------------
SEMILLAS_EXPERIMENTO = 1   # Para testing/optimización
SEMILLAS_FINAL = 1        # Para predicción final

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
UNDERSAMPLING_RATIO = 0.05  # Proporción de clase mayoritaria a mantener (0.1 = 10%)
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
# VISUALIZACIÓN DE GANANCIAS
# ============================================================================

def generar_grafico_ganancias(df_testing, exp_path):
    """
    Genera gráficos de ganancia por corte para análisis visual
    
    Args:
        df_testing: DataFrame con resultados de testing
        exp_path: Ruta del experimento
    """
    logger.info("\nGenerando gráficos de ganancia...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Análisis de Ganancia por Corte - Testing', fontsize=16, fontweight='bold')
    
    cortes = df_testing['corte'].values
    gan_test1 = df_testing['gan_test1_prom'].values
    gan_test2 = df_testing['gan_test2_prom'].values
    gan_promedio = df_testing['gan_promedio'].values
    gan_min = df_testing['gan_min'].values
    gan_max = df_testing['gan_max'].values
    
    # Gráfico 1: Ganancia Test 1
    axes[0, 0].plot(cortes, gan_test1, marker='o', linewidth=2, markersize=6, color='#2E86AB')
    axes[0, 0].fill_between(cortes, gan_test1, alpha=0.3, color='#2E86AB')
    axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[0, 0].set_title(f'Ganancia Test 1 (Mes {FOTO_MES_TEST_1})', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Corte (N° envíos)')
    axes[0, 0].set_ylabel('Ganancia ($)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].ticklabel_format(style='plain', axis='y')
    
    # Marcar mejor corte en Test 1
    idx_mejor_t1 = np.argmax(gan_test1)
    axes[0, 0].plot(cortes[idx_mejor_t1], gan_test1[idx_mejor_t1], 
                    marker='*', markersize=20, color='gold', 
                    markeredgecolor='red', markeredgewidth=2)
    axes[0, 0].annotate(f'Mejor: {cortes[idx_mejor_t1]}\n${gan_test1[idx_mejor_t1]:,.0f}',
                       xy=(cortes[idx_mejor_t1], gan_test1[idx_mejor_t1]),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
                       fontweight='bold')
    
    # Gráfico 2: Ganancia Test 2
    axes[0, 1].plot(cortes, gan_test2, marker='o', linewidth=2, markersize=6, color='#A23B72')
    axes[0, 1].fill_between(cortes, gan_test2, alpha=0.3, color='#A23B72')
    axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[0, 1].set_title(f'Ganancia Test 2 (Mes {FOTO_MES_TEST_2})', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Corte (N° envíos)')
    axes[0, 1].set_ylabel('Ganancia ($)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].ticklabel_format(style='plain', axis='y')
    
    # Marcar mejor corte en Test 2
    idx_mejor_t2 = np.argmax(gan_test2)
    axes[0, 1].plot(cortes[idx_mejor_t2], gan_test2[idx_mejor_t2], 
                    marker='*', markersize=20, color='gold',
                    markeredgecolor='red', markeredgewidth=2)
    axes[0, 1].annotate(f'Mejor: {cortes[idx_mejor_t2]}\n${gan_test2[idx_mejor_t2]:,.0f}',
                       xy=(cortes[idx_mejor_t2], gan_test2[idx_mejor_t2]),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
                       fontweight='bold')
    
    # Gráfico 3: Comparación Test 1 vs Test 2
    axes[1, 0].plot(cortes, gan_test1, marker='o', linewidth=2, label=f'Test 1 ({FOTO_MES_TEST_1})', color='#2E86AB')
    axes[1, 0].plot(cortes, gan_test2, marker='s', linewidth=2, label=f'Test 2 ({FOTO_MES_TEST_2})', color='#A23B72')
    axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1, 0].set_title('Comparación Test 1 vs Test 2', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Corte (N° envíos)')
    axes[1, 0].set_ylabel('Ganancia ($)')
    axes[1, 0].legend(loc='best')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].ticklabel_format(style='plain', axis='y')
    
    # Gráfico 4: Ganancia PROMEDIO (Min, Promedio, Max)
    axes[1, 1].plot(cortes, gan_promedio, marker='o', linewidth=3, markersize=8, 
                    label='Promedio', color='#F18F01', zorder=3)
    axes[1, 1].fill_between(cortes, gan_min, gan_max, alpha=0.2, color='gray', label='Rango (Min-Max)')
    axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('Ganancia PROMEDIO (Test 1 + Test 2)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Corte (N° envíos)')
    axes[1, 1].set_ylabel('Ganancia ($)')
    axes[1, 1].legend(loc='best')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].ticklabel_format(style='plain', axis='y')
    
    # Marcar MEJOR CORTE PROMEDIO
    idx_mejor_prom = np.argmax(gan_promedio)
    axes[1, 1].plot(cortes[idx_mejor_prom], gan_promedio[idx_mejor_prom], 
                    marker='*', markersize=25, color='gold',
                    markeredgecolor='red', markeredgewidth=2, zorder=4)
    axes[1, 1].annotate(f'🏆 MEJOR: {cortes[idx_mejor_prom]}\n${gan_promedio[idx_mejor_prom]:,.0f}',
                       xy=(cortes[idx_mejor_prom], gan_promedio[idx_mejor_prom]),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round', facecolor='gold', alpha=0.9),
                       fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    
    # Guardar gráfico
    grafico_path = os.path.join(exp_path, "grafico_ganancias_testing.png")
    plt.savefig(grafico_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✓ grafico_ganancias_testing.png guardado")
    logger.info(f"    📊 Mejor corte: {cortes[idx_mejor_prom]} (Ganancia: ${gan_promedio[idx_mejor_prom]:,.0f})")
    
    return grafico_path


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

def agregar_canaritos(df, num_canaritos=None, semilla=None):
    """Agrega variables aleatorias (canaritos) para detectar overfitting"""
    if num_canaritos is None:
        num_canaritos = QCANARITOS
    if semilla is None:
        semilla = SEMILLA_PRIMIGENIA
        
    logger.info(f"Agregando {num_canaritos} canaritos...")
    
    np.random.seed(semilla)
    
    for i in range(num_canaritos):
        nombre = f'canarito{i+1}'
        df[nombre] = np.random.rand(len(df))
    
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

def preparar_datos_train_test_final(df):
    """
    Prepara los datos para el workflow de 3 etapas:
    1. TRAIN: 201901-202102 (todos los meses)
    2. TEST: 202104 y 202106 (validación)
    3. FINAL: 202108 (predicción final)
    
    Returns:
        Tupla de (df_train, df_test1, df_test2, df_final, feature_cols)
    """
    logger.info("Preparando datos para workflow de 3 etapas...")
    
    # Generar lista de meses de train
    meses_train = generar_rango_meses(FOTO_MES_TRAIN_INICIO, FOTO_MES_TRAIN_FIN)
    logger.info(f"Meses de entrenamiento: {meses_train[0]} a {meses_train[-1]} ({len(meses_train)} meses)")
    
    # Filtrar datos
    df_train = df[df['foto_mes'].isin(meses_train)].copy()
    df_test1 = df[df['foto_mes'] == FOTO_MES_TEST_1].copy()
    df_test2 = df[df['foto_mes'] == FOTO_MES_TEST_2].copy()
    df_final = df[df['foto_mes'] == FOTO_MES_FINAL].copy()
    
    logger.info(f"Train: {len(df_train):,} registros ({len(meses_train)} meses)")
    logger.info(f"Test 1 ({FOTO_MES_TEST_1}): {len(df_test1):,} registros")
    logger.info(f"Test 2 ({FOTO_MES_TEST_2}): {len(df_test2):,} registros")
    logger.info(f"Final ({FOTO_MES_FINAL}): {len(df_final):,} registros")
    
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


# ============================================================================
# ETAPA 2: TESTING (202104 y 202106)
# ============================================================================

def etapa_testing(df_train, df_test1, df_test2, feature_cols, exp_path):
    """
    Etapa 2: Testing con datos de 202104 y 202106
    Entrena con train, predice en test1 y test2, calcula ganancias por corte
    
    Args:
        df_train: DataFrame de entrenamiento
        df_test1: DataFrame de test 1
        df_test2: DataFrame de test 2
        feature_cols: Lista de columnas de features
        exp_path: Ruta del directorio del experimento
    
    Returns:
        Tupla de (df_testing, mejor_corte, predicciones_test1, predicciones_test2)
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
    
    # Generar semillas
    semillas = generar_semillas(SEMILLA_PRIMIGENIA, SEMILLAS_EXPERIMENTO)
    
    # Matrices para acumular ganancias
    matriz_gan_test1 = np.zeros((SEMILLAS_EXPERIMENTO, len(CORTES)))
    matriz_gan_test2 = np.zeros((SEMILLAS_EXPERIMENTO, len(CORTES)))
    
    # Acumuladores para predicciones promedio
    predicciones_acum_test1 = np.zeros(len(X_test1))
    predicciones_acum_test2 = np.zeros(len(X_test2))
    
    # Entrenar con múltiples semillas
    for idx_sem, semilla in enumerate(semillas):
        if (idx_sem + 1) % 5 == 0:
            logger.info(f"  Testing con semilla {idx_sem + 1}/{SEMILLAS_EXPERIMENTO}...")
        
        # Crear pequeño conjunto de validación para early stopping
        n_val = min(5000, len(X_train) // 10)
        indices = np.arange(len(X_train))
        np.random.seed(semilla)
        np.random.shuffle(indices)
        idx_val = indices[:n_val]
        
        X_val_mini = X_train.iloc[idx_val]
        y_val_mini = y_train.iloc[idx_val]
        
        # Entrenar
        modelo = entrenar_lgbm(
            X_train, y_train,
            X_val_mini, y_val_mini,
            semilla,
            usar_ganancia=False
        )
        
        # Predecir en test1
        y_pred_test1 = modelo.predict(X_test1)
        predicciones_acum_test1 += y_pred_test1
        
        # Predecir en test2
        y_pred_test2 = modelo.predict(X_test2)
        predicciones_acum_test2 += y_pred_test2
        
        # ====================================================================
        # CORRECCIÓN DE DATA LEAKAGE LÓGICO: 
        # Calcular curva de ganancia UNA SOLA VEZ usando PROBABILIDADES
        # ====================================================================
        
        # Test 1: Calcula la curva completa
        _, gan_acumulada_test1 = calcular_ganancia(
            y_pred=y_pred_test1, 
            y_true=y_test1.values
        )
        
        # Test 2: Calcula la curva completa
        _, gan_acumulada_test2 = calcular_ganancia(
            y_pred=y_pred_test2, 
            y_true=y_test2.values
        )
        
        # Calcular ganancia para cada corte (extrayendo del índice de la curva)
        for idx_corte, corte in enumerate(CORTES):
            n_envios = min(corte, len(gan_acumulada_test1))
            
            # Test 1: La ganancia en el corte N es el valor de la curva acumulada en el índice N-1
            if n_envios > 0:
                matriz_gan_test1[idx_sem, idx_corte] = gan_acumulada_test1[n_envios - 1]
            
            # Test 2: La ganancia en el corte N es el valor de la curva acumulada en el índice N-1
            if n_envios > 0:
                matriz_gan_test2[idx_sem, idx_corte] = gan_acumulada_test2[n_envios - 1]

        # ====================================================================
        
        del modelo
        limpiar_memoria()
    
    # Promediar predicciones
    predicciones_prom_test1 = predicciones_acum_test1 / SEMILLAS_EXPERIMENTO
    predicciones_prom_test2 = predicciones_acum_test2 / SEMILLAS_EXPERIMENTO
    
    # Crear DataFrames de predicciones promedio (ordenados por prob)
    df_pred_final_test1 = pd.DataFrame({
        'numero_de_cliente': df_test1['numero_de_cliente'].values,
        'foto_mes': df_test1['foto_mes'].values,
        'clase_ternaria': df_test1['clase_ternaria'].values,
        'prob': predicciones_prom_test1
    }).sort_values('prob', ascending=False).reset_index(drop=True)
    
    df_pred_final_test2 = pd.DataFrame({
        'numero_de_cliente': df_test2['numero_de_cliente'].values,
        'foto_mes': df_test2['foto_mes'].values,
        'clase_ternaria': df_test2['clase_ternaria'].values,
        'prob': predicciones_prom_test2
    }).sort_values('prob', ascending=False).reset_index(drop=True)
    
    # Calcular estadísticas
    gan_test1_prom = matriz_gan_test1.mean(axis=0)
    gan_test1_std = matriz_gan_test1.std(axis=0)
    
    gan_test2_prom = matriz_gan_test2.mean(axis=0)
    gan_test2_std = matriz_gan_test2.std(axis=0)
    
    # Ganancia promedio combinada
    gan_promedio = (gan_test1_prom + gan_test2_prom) / 2
    
    # Crear DataFrame de resultados
    df_testing = pd.DataFrame({
        'corte': CORTES,
        'gan_test1_prom': gan_test1_prom,
        'gan_test1_std': gan_test1_std,
        'gan_test2_prom': gan_test2_prom,
        'gan_test2_std': gan_test2_std,
        'gan_promedio': gan_promedio,
        'gan_min': np.minimum(gan_test1_prom, gan_test2_prom),
        'gan_max': np.maximum(gan_test1_prom, gan_test2_prom)
    })
    
    # Identificar mejor corte
    idx_mejor = np.argmax(gan_promedio)
    mejor_corte = CORTES[idx_mejor]
    mejor_ganancia = gan_promedio[idx_mejor]
    
    logger.info(f"\nResultados Testing:")
    logger.info(f"  Mejor corte: {mejor_corte}")
    logger.info(f"  Ganancia promedio: ${mejor_ganancia:,.0f}")
    logger.info(f"  Test 1: ${gan_test1_prom[idx_mejor]:,.0f} (±${gan_test1_std[idx_mejor]:,.0f})")
    logger.info(f"  Test 2: ${gan_test2_prom[idx_mejor]:,.0f} (±${gan_test2_std[idx_mejor]:,.0f})")
    
    # ========================================================================
    # GUARDAR RESULTADOS
    # ========================================================================
    logger.info("\nGuardando resultados de testing...")
    
    # Evaluación
    df_testing.to_csv(os.path.join(exp_path, "evaluacion_testing.csv"), index=False)
    logger.info(f"  ✓ evaluacion_testing.csv")
    
    # Matrices de ganancia
    pd.DataFrame(matriz_gan_test1, columns=[f'corte_{c}' for c in CORTES]).to_csv(
        os.path.join(exp_path, "matriz_test1.csv"), index=False
    )
    logger.info(f"  ✓ matriz_test1.csv")
    
    pd.DataFrame(matriz_gan_test2, columns=[f'corte_{c}' for c in CORTES]).to_csv(
        os.path.join(exp_path, "matriz_test2.csv"), index=False
    )
    logger.info(f"  ✓ matriz_test2.csv")
    
    # Predicciones completas (con clase_ternaria real para análisis)
    df_pred_final_test1.to_csv(
        os.path.join(exp_path, "predicciones_test1.csv"), index=False
    )
    logger.info(f"  ✓ predicciones_test1.csv ({len(df_pred_final_test1):,} registros)")
    
    df_pred_final_test2.to_csv(
        os.path.join(exp_path, "predicciones_test2.csv"), index=False
    )
    logger.info(f"  ✓ predicciones_test2.csv ({len(df_pred_final_test2):,} registros)")
    
    # ========================================================================
    # GENERAR GRÁFICO DE GANANCIAS
    # ========================================================================
    generar_grafico_ganancias(df_testing, exp_path)
    
    # ========================================================================
    # GENERAR SUBMISSIONS PARA TEST 1 Y TEST 2
    # ========================================================================
    logger.info("\nGenerando submissions para meses de testing...")
    
    # Submissions para test1 (202104)
    logger.info(f"\nMes {FOTO_MES_TEST_1}:")
    generar_submissions(df_pred_final_test1, exp_path, sufijo=f"_test_{FOTO_MES_TEST_1}")
    
    # Submissions para test2 (202106)
    logger.info(f"\nMes {FOTO_MES_TEST_2}:")
    generar_submissions(df_pred_final_test2, exp_path, sufijo=f"_test_{FOTO_MES_TEST_2}")
    
    return df_testing, mejor_corte, df_pred_final_test1, df_pred_final_test2


# ============================================================================
# ETAPA 3: PREDICCIÓN FINAL (202108)
# ============================================================================

def etapa_final(df_train, df_final, feature_cols, exp_path):
    """
    Etapa 3: Predicción final para 202108
    Entrena con train, predice en final con múltiples semillas
    
    Args:
        df_train: DataFrame de entrenamiento
        df_final: DataFrame final (202108)
        feature_cols: Lista de columnas de features
        exp_path: Ruta del directorio del experimento
    
    Returns:
        DataFrame con predicciones ordenadas
    """
    logger.info("="*80)
    logger.info("ETAPA 3: PREDICCIÓN FINAL (202108)")
    logger.info("="*80)
    
    # Preparar datos
    X_train = df_train[feature_cols]
    y_train = (df_train['clase_ternaria'] == 'BAJA+2').astype(int)
    X_final = df_final[feature_cols]
    
    logger.info(f"Train: {len(X_train):,} registros")
    logger.info(f"Final ({FOTO_MES_FINAL}): {len(X_final):,} registros")
    logger.info(f"Entrenando con {SEMILLAS_FINAL} semillas (ENSAMBLE)...")
    
    # Generar semillas
    semillas = generar_semillas(SEMILLA_PRIMIGENIA, SEMILLAS_FINAL)
    
    # Acumulador de predicciones
    predicciones_acum = np.zeros(len(X_final))
    
    # Entrenar múltiples modelos
    for idx, semilla in enumerate(semillas, 1):
        if idx % 10 == 0:
            logger.info(f"  Modelo {idx}/{SEMILLAS_FINAL}...")
        
        # Crear conjunto de validación mínimo
        n_val = min(5000, len(X_train) // 10)
        indices = np.arange(len(X_train))
        np.random.seed(semilla)
        np.random.shuffle(indices)
        idx_val = indices[:n_val]
        
        X_val_mini = X_train.iloc[idx_val]
        y_val_mini = y_train.iloc[idx_val]
        
        # Entrenar
        modelo = entrenar_lgbm(
            X_train, y_train,
            X_val_mini, y_val_mini,
            semilla,
            usar_ganancia=False
        )
        
        # Predecir
        predicciones = modelo.predict(X_final)
        predicciones_acum += predicciones
        
        del modelo
        limpiar_memoria()
    
    # Promediar predicciones (ENSAMBLE)
    predicciones_promedio = predicciones_acum / SEMILLAS_FINAL
    
    # Crear DataFrame de resultados
    resultado = pd.DataFrame({
        'numero_de_cliente': df_final['numero_de_cliente'].values,
        'foto_mes': df_final['foto_mes'].values,
        'prob': predicciones_promedio
    })
    
    resultado = resultado.sort_values('prob', ascending=False).reset_index(drop=True)
    
    logger.info(f"\nPredicciones generadas: {len(resultado):,}")
    logger.info(f"Top 10 probabilidades: {resultado['prob'].head(10).values}")
    
    # ========================================================================
    # GUARDAR PREDICCIONES COMPLETAS
    # ========================================================================
    logger.info("\nGuardando predicciones finales...")
    pred_path = os.path.join(exp_path, "predicciones_final.csv")
    resultado.to_csv(pred_path, index=False)
    logger.info(f"  ✓ predicciones_final.csv ({len(resultado):,} registros)")
    
    return resultado


# ============================================================================
# GENERACIÓN DE SUBMISSIONS
# ============================================================================

def generar_submissions(predicciones, exp_path, cortes=None, sufijo=""):
    """
    Genera archivos de submission para cada corte
    GENERA 2 ARCHIVOS POR CORTE:
    1. Archivo con Predicted (0/1) - para Kaggle
    2. Archivo con probabilidad - para análisis
    
    Args:
        predicciones: DataFrame con predicciones ordenadas
        exp_path: Ruta del directorio del experimento
        cortes: Lista de cortes a evaluar
        sufijo: Sufijo para el nombre del archivo (ej: "_test_202104")
    
    Returns:
        DataFrame con resumen de cortes
    """
    if cortes is None:
        cortes = CORTES
        
    logger.info(f"Generando {len(cortes)} submissions{sufijo} (2 archivos por corte)...")
    
    kaggle_dir = os.path.join(exp_path, "kaggle")
    crear_directorio(kaggle_dir)
    
    resultados = []
    
    for corte in cortes:
        pred_temp = predicciones.copy()
        pred_temp['Predicted'] = (pred_temp.index < corte).astype(int)
        
        # =====================================================================
        # ARCHIVO 1: Con Predicted (0/1) - Para Kaggle
        # =====================================================================
        filename_pred = f"KA{EXPERIMENTO}_{corte}{sufijo}.csv"
        filepath_pred = os.path.join(kaggle_dir, filename_pred)
        
        pred_temp[['numero_de_cliente', 'Predicted']].to_csv(
            filepath_pred, index=False
        )
        
        # =====================================================================
        # ARCHIVO 2: Con Probabilidad - Para análisis
        # =====================================================================
        filename_prob = f"KA{EXPERIMENTO}_{corte}{sufijo}_prob.csv"
        filepath_prob = os.path.join(kaggle_dir, filename_prob)
        
        pred_temp[['numero_de_cliente', 'prob']].to_csv(
            filepath_prob, index=False
        )
        
        envios = pred_temp['Predicted'].sum()
        resultados.append({
            'corte': corte,
            'envios': envios,
            'archivo_predicted': filename_pred,
            'archivo_prob': filename_prob
        })
        
        if corte % 2500 == 0 or corte == cortes[0]:
            logger.info(f"  Corte {corte}: {envios} envíos")
            logger.info(f"    → {filename_pred} (Predicted 0/1)")
            logger.info(f"    → {filename_prob} (Probabilidades)")
    
    # Crear y guardar DataFrame de resultados
    df_resultados = pd.DataFrame(resultados)
    resultados_path = os.path.join(exp_path, f"resultados_cortes{sufijo}.csv")
    df_resultados.to_csv(resultados_path, index=False)
    
    logger.info(f"\nArchivos generados:")
    logger.info(f"  ✓ {len(cortes)} archivos Predicted (0/1)")
    logger.info(f"  ✓ {len(cortes)} archivos Probabilidad")
    logger.info(f"  ✓ Total: {len(cortes) * 2} archivos")
    logger.info(f"  ✓ resultados_cortes{sufijo}.csv")
    
    return df_resultados


# ============================================================================
# WORKFLOW PRINCIPAL
# ============================================================================

def main():
    """Función principal del workflow de 3 etapas"""
    print("="*80)
    print("R_to_py: Workflow Completo de 3 Etapas")
    print("CON GRÁFICOS Y DOBLE ARCHIVO POR CORTE")
    print("="*80)
    
    inicio_ejecucion = datetime.now()
    
    logger.info(f"Inicio: {inicio_ejecucion.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Experimento: {EXPERIMENTO}")
    logger.info(f"Ganancia por acierto: ${GANANCIA_ACIERTO:,}")
    logger.info(f"Costo por estímulo: ${COSTO_ESTIMULO:,}")
    logger.info(f"Canaritos: {QCANARITOS}")
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
    
    logger.info(f"Cargando dataset desde {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH, compression='gzip')
    logger.info(f"Dataset cargado: {df.shape}")
    
    # Calcular clase_ternaria
    df = calcular_clase_ternaria(df)
    logger.info(f"Lags y Deltas: {FEATURE_ENGINEERING_LAGS} (órdenes: {LAGS_ORDEN if FEATURE_ENGINEERING_LAGS else 'N/A'})")
    # Agregar lags y deltas
    if FEATURE_ENGINEERING_LAGS:
        df = agregar_lags_y_deltas(df, LAGS_ORDEN)
    # Agregar canaritos
    df = agregar_canaritos(df, QCANARITOS)
    
    limpiar_memoria()
    print()
    
    # ========================================================================
    # PASO 2: PREPARACIÓN DE DATOS (3 ETAPAS)
    # ========================================================================
    logger.info("="*80)
    logger.info("PASO 2: Preparación de datos (3 etapas)")
    logger.info("="*80)
    logger.info(f"Train: {FOTO_MES_TRAIN_INICIO} a {FOTO_MES_TRAIN_FIN}")
    logger.info(f"Test: {FOTO_MES_TEST_1} y {FOTO_MES_TEST_2}")
    logger.info(f"Final: {FOTO_MES_FINAL}")
    print()
    
    df_train, df_test1, df_test2, df_final, feature_cols = preparar_datos_train_test_final(df)
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
    # PASO 5: GENERACIÓN DE SUBMISSIONS PARA MES FINAL
    # ========================================================================
    logger.info("="*80)
    logger.info("PASO 5: Generación de submissions para mes FINAL")
    logger.info("="*80)
    
    logger.info(f"\nMes {FOTO_MES_FINAL}:")
    df_resultados = generar_submissions(predicciones, exp_path, sufijo=f"_final_{FOTO_MES_FINAL}")
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
    logger.info(f"\n📤 Submissions (por mes):")
    logger.info(f"  ✓ kaggle/ - Mes {FOTO_MES_TEST_1}: {len(CORTES) * 2} archivos ({len(CORTES)} Predicted + {len(CORTES)} Prob)")
    logger.info(f"  ✓ kaggle/ - Mes {FOTO_MES_TEST_2}: {len(CORTES) * 2} archivos ({len(CORTES)} Predicted + {len(CORTES)} Prob)")
    logger.info(f"  ✓ kaggle/ - Mes {FOTO_MES_FINAL}: {len(CORTES) * 2} archivos ({len(CORTES)} Predicted + {len(CORTES)} Prob)")
    logger.info(f"  ✓ Total: {len(CORTES) * 3 * 2} archivos ({len(CORTES) * 3} Predicted + {len(CORTES) * 3} Prob)")
    logger.info(f"\n📊 Visualización:")
    logger.info(f"  ✓ grafico_ganancias_testing.png (análisis visual de cortes)")
    
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