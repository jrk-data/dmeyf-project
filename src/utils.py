import pandas as pd
import numpy as np
import re

YES_NO_MAP = {
    "S": 1, "N": 0, "SI": 1, "NO": 0,
    "Y": 1, "N0": 0, "YES": 1, "NO": 0,
    "True": 1, "False": 0, "TRUE": 1, "FALSE": 0,
    True: 1, False: 0, 1: 1, 0: 0
}

def _coerce_object_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte columnas object a tipos válidos para LightGBM:
       - Booleans S/N, SI/NO, YES/NO, True/False -> 0/1
       - Fechas (nombres que contienen 'fecha' o 'Finiciomora') -> días desde epoch (int)
       - Resto: to_numeric(errors='coerce')"""
    if df is None or df.empty:
        return df
    df = df.copy()

    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if not obj_cols:
        return df

    # Normalizo strings y vacío
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip().replace({"": np.nan, "None": np.nan, "NULL": np.nan, "NaN": np.nan})

    # 1) Mapear booleans/yes-no si son de ese tipo
    def looks_like_yesno(series: pd.Series) -> bool:
        sample = series.dropna().astype(str).str.upper().head(200)
        return sample.isin(["S","N","SI","NO","Y","YES","TRUE","FALSE","0","1"]).mean() > 0.8

    # 2) Detectar fechas por nombre de columna
    def is_date_col(name: str) -> bool:
        name_low = name.lower()
        return ("fecha" in name_low) or ("finiciomora" in name_low) or re.search(r"(date|fec|f_ini|fini)", name_low) is not None

    for c in obj_cols:
        s = df[c]

        if looks_like_yesno(s):
            df[c] = s.map(lambda x: YES_NO_MAP.get(str(x), np.nan)).astype("float32")
            continue

        if is_date_col(c):
            # Intento parsear fecha
            parsed = pd.to_datetime(s, errors="coerce", utc=True)
            # Convierto a días desde epoch (evita fugas de tz y es más estable)
            df[c] = (parsed.view("int64") // 10**9 // 86400).astype("float32")
            continue

    # 3) Lo que quede en object -> numérico (coerce)
    rem_obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if rem_obj_cols:
        for c in rem_obj_cols:
            df[c] = pd.to_numeric(df[c].str.replace(",", ".", regex=False), errors="coerce").astype("float32")

    return df
