# src/reporting.py
import numpy as np
import pandas as pd
from datetime import datetime

# Usar los mismos valores que en tu proyecto
GANANCIA_ACIERTO = 25000  # o desde config
COSTO_ESTIMULO   = 500    # o desde config


def compute_gain_model(
    y_pred: np.ndarray,
    weight: np.ndarray,
    ganancia_acierto: float = GANANCIA_ACIERTO,
    costo_estimulo: float = COSTO_ESTIMULO,
):
    """
    Replica la lógica de lgb_gan_eval para evaluar un modelo sobre un conjunto dado.

    Parameters
    ----------
    y_pred : np.ndarray
        Probabilidades predichas (salida del modelo).
    weight : np.ndarray
        Pesos usados en el LGBM Dataset. En tu caso:
            weight == 1.00002  -> verdadero 1 (positivo)
            weight <  1.00002  -> verdadero 0 (negativo)
    ganancia_acierto : float
    costo_estimulo : float

    Returns
    -------
    dict con:
        ganancia_modelo : float
        k_optimo        : int
        threshold_opt   : float
        curva_ganancia  : np.ndarray (cumsum a lo largo del ranking)
    """
    # vector de ganancias por registro
    ganancia = (
        np.where(weight == 1.00002, ganancia_acierto, 0.0)
        - np.where(weight < 1.00002, costo_estimulo, 0.0)
    )

    # ordenar de mayor a menor probabilidad
    order = np.argsort(y_pred)[::-1]
    gan_ord = ganancia[order]
    y_pred_sorted = y_pred[order]

    curva = np.cumsum(gan_ord)
    ganancia_modelo = float(np.max(curva))
    k_optimo = int(np.argmax(curva) + 1)
    threshold_opt = float(y_pred_sorted[k_optimo - 1])

    return {
        "ganancia_modelo": ganancia_modelo,
        "k_optimo": k_optimo,
        "threshold_opt": threshold_opt,
        "curva_ganancia": curva,
    }


def compute_gain_max_possible(
    weight: np.ndarray,
    ganancia_acierto: float = GANANCIA_ACIERTO,
):
    """
    Ganancia máxima teórica posible, bajo el mismo esquema de ganancia, asumiendo:
    - el modelo es perfecto y pone todos los positivos primero.
    - se corta justo después del último positivo (no se incurren costos por negativos).

    En ese caso, la ganancia máxima = (#positivos) * GANANCIA_ACIERTO
    """
    n_positivos = int(np.sum(weight == 1.00002))
    gan_max = float(n_positivos * ganancia_acierto)
    return gan_max, n_positivos



def generar_reporte_html_ganancia(
    y_pred: np.ndarray,
    weight: np.ndarray,
    experimento: str,
    mes: int,
    output_html_path: str,
    ganancia_acierto: float = GANANCIA_ACIERTO,
    costo_estimulo: float = COSTO_ESTIMULO,
):
    """
    Genera un reporte HTML comparando:
    - Ganancia máxima teórica posible.
    - Ganancia obtenida por el modelo (misma lógica que lgb_gan_eval).

    Parameters
    ----------
    y_pred : np.ndarray
        Probabilidades predichas para el mes (por ej. 202104).
    weight : np.ndarray
        Pesos que codifican los verdaderos targets (==1.00002 -> 1, <1.00002 -> 0).
    experimento : str
        Nombre del experimento (ej: 'experimento_estacional_abril_01_202104').
    mes : int
        Mes evaluado (ej: 202104).
    output_html_path : str
        Ruta completa del archivo HTML a generar.
    """

    # 1) Métricas de modelo
    res_model = compute_gain_model(
        y_pred=y_pred,
        weight=weight,
        ganancia_acierto=ganancia_acierto,
        costo_estimulo=costo_estimulo,
    )

    gan_modelo = res_model["ganancia_modelo"]
    k_optimo = res_model["k_optimo"]
    thr_opt = res_model["threshold_opt"]
    curva = res_model["curva_ganancia"]

    # 2) Ganancia máxima teórica
    gan_max, n_positivos = compute_gain_max_possible(
        weight=weight,
        ganancia_acierto=ganancia_acierto,
    )

    n_total = int(len(weight))
    n_negativos = n_total - n_positivos
    ratio = gan_modelo / gan_max if gan_max > 0 else np.nan

    # 3) Armar tabla resumen como DataFrame (para debug si querés)
    resumen = pd.DataFrame(
        {
            "metrica": [
                "Ganancia máxima posible",
                "Ganancia modelo",
                "Gap (max - modelo)",
                "Fracción capturada",
                "N registros",
                "N positivos (weight==1.00002)",
                "N negativos (weight<1.00002)",
                "k óptimo (envíos)",
                "threshold óptimo",
                "GANANCIA_ACIERTO",
                "COSTO_ESTIMULO",
            ],
            "valor": [
                gan_max,
                gan_modelo,
                gan_max - gan_modelo,
                ratio,
                n_total,
                n_positivos,
                n_negativos,
                k_optimo,
                thr_opt,
                ganancia_acierto,
                costo_estimulo,
            ],
        }
    )

    # 4) HTML simple (sin gráficos por ahora, pero ya listo para compartir)
    fecha_gen = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Convertimos el resumen a tabla HTML
    resumen_html = resumen.to_html(
        index=False,
        float_format=lambda x: f"{x:,.2f}",
        classes="table table-striped",
        border=0,
    )

    html = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Reporte de Ganancia - {experimento} - {mes}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 30px;
            background-color: #f7f7f7;
        }}
        h1, h2, h3 {{
            color: #333333;
        }}
        .meta {{
            font-size: 0.9em;
            color: #666666;
        }}
        .card {{
            background-color: #ffffff;
            border-radius: 6px;
            padding: 20px;
            margin-bottom: 25px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin-top: 10px;
        }}
        th, td {{
            text-align: left;
            padding: 8px;
        }}
        th {{
            background-color: #eeeeee;
        }}
        tr:nth-child(even) {{
            background-color: #fafafa;
        }}
        .highlight {{
            font-weight: bold;
            color: #1a73e8;
        }}
    </style>
</head>
<body>

    <h1>Reporte de Ganancia</h1>
    <div class="meta">
        <p><strong>Experimento:</strong> {experimento}</p>
        <p><strong>Mes evaluado:</strong> {mes}</p>
        <p><strong>Generado:</strong> {fecha_gen}</p>
    </div>

    <div class="card">
        <h2>Resumen de Métricas</h2>
        {resumen_html}
    </div>

    <div class="card">
        <h2>Interpretación rápida</h2>
        <p>
            La ganancia máxima teórica posible bajo este esquema de costos y beneficios es de
            <span class="highlight">{gan_max:,.0f}</span>.
            El modelo obtuvo una ganancia de
            <span class="highlight">{gan_modelo:,.0f}</span>,
            lo que representa
            <span class="highlight">{ratio*100:.2f}%</span> de la ganancia máxima.
        </p>
        <p>
            El punto óptimo de corte según la curva de ganancia se alcanza con
            <span class="highlight">{k_optimo}</span> envíos,
            correspondiente a un threshold aproximado de
            <span class="highlight">{thr_opt:.4f}</span> sobre la probabilidad predicha.
        </p>
    </div>

</body>
</html>
"""

    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html)

    return {
        "ganancia_modelo": gan_modelo,
        "ganancia_maxima": gan_max,
        "ratio": ratio,
        "k_optimo": k_optimo,
        "threshold_opt": thr_opt,
        "path": output_html_path,
    }
