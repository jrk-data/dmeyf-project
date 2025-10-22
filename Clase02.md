# 📘 Clase02.md

## **Modularización: separar responsabilidades (pandas + SQL, con logging)**

---

## Git - Flujo de trabajo básico

Antes de comenzar con la clase, es importante mantener nuestro repositorio actualizado y sincronizado. Ejecuta los siguientes comandos en la consola:

```bash
git status
git add .
git commit -m "actualizacion"
git pull origin
git status
git add .
git push
```

**Explicación del flujo:**

1. `git status` - Verifica el estado actual del repositorio, mostrando archivos modificados, agregados o eliminados
2. `git add .` - Agrega todos los cambios locales al área de staging (preparación para commit)
3. `git commit -m "actualizacion"` - Crea un commit con los cambios locales y un mensaje descriptivo
4. `git pull origin` - Descarga y fusiona los cambios más recientes del repositorio remoto
5. `git status` - Verifica nuevamente el estado después del pull para detectar posibles conflictos
6. `git add .` - Agrega cualquier archivo que pueda haber sido modificado durante la fusión
7. `git push` - Sube los cambios locales al repositorio remoto

Este flujo asegura que tu trabajo esté sincronizado con el repositorio principal y evita conflictos de versiones.

**Al finalizar la clase, ejecuta nuevamente:**

```bash
git add .
git commit -m "Clase02: Modularización completada - separación de responsabilidades con logging"
git push
```

---

## 🎯 Objetivo

Entender que **cada archivo tiene un rol** y orquestar:
**cargar → transformar (SQL) → loguear → guardar**, ahora con **logging profesional**.

---

## 📑 Índice de la clase

1. **Estructura de carpetas** - Organizaremos el proyecto con una arquitectura clara que separe datos, código fuente, logs y resultados
2. **Configuración de logs con `logging`** - Implementaremos un sistema de logging profesional para rastrear la ejecución y detectar errores
3. **Función en `main.py` para carga de dataset** - Crearemos una función básica de carga con documentación clara de parámetros y valores de retorno
4. **Crear `src/loader.py` y trasladar la función `cargar_datos(path)`** - Modularizaremos el código moviendo la lógica de carga a un módulo especializado
5. **Importaciones y `__init__.py`** - Configuraremos el sistema de paquetes Python para permitir importaciones limpias entre módulos
6. **`src/features.py`: SQL directo (DuckDB) para generar `Lag`** - Implementaremos feature engineering usando SQL para crear variables de rezago temporal
7. **`main.py` orquestando todo con logs** - Integraremos todos los módulos en un flujo principal con logging completo

---

## 1) Estructura de carpetas

La organización del proyecto es fundamental para mantener un código limpio y escalable. Esta estructura separa claramente las responsabilidades:

```
proyecto_ml/
├── main.py              # Archivo principal que orquesta todo el flujo
├── requirements.txt     # Dependencias del proyecto
├── data/               # Carpeta para datasets de entrada y salida
│   └── competencia_01.csv
├── logs/               # Archivos de log para debugging y monitoreo
├── output/             # Resultados finales y archivos procesados
└── src/                # Código fuente modularizado
    ├── __init__.py     # Convierte src/ en un paquete Python
    ├── loader.py       # Módulo especializado en carga de datos
    └── features.py     # Módulo para feature engineering
```

**Beneficios de esta estructura:**

- **Separación de responsabilidades**: Cada carpeta tiene un propósito específico
- **Escalabilidad**: Fácil agregar nuevos módulos en `src/`
- **Mantenibilidad**: Código organizado y fácil de encontrar
- **Profesionalismo**: Estructura estándar en proyectos

## 2) Configuración de logs con `logging`

El sistema de logging es esencial para el debugging y monitoreo de aplicaciones en producción. Python ofrece el módulo `logging` que permite registrar eventos durante la ejecución del programa.

**¿Por qué usar logging en lugar de print()?**

- **Niveles de severidad**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Formato personalizable**: Timestamps, nombres de módulos, números de línea
- **Múltiples destinos**: Archivo, consola, servicios remotos
- **Control granular**: Activar/desactivar logs por módulo o nivel

**Documentación oficial**: https://docs.python.org/3/howto/logging.html

```python
import pandas as pd
import os
import datetime
import logging

from src.loader import cargar_datos
from src.features import feature_engineering_lag

## config basico logging
os.makedirs("logs", exist_ok=True)

fecha = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
monbre_log = f"log_{fecha}.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s %(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/{monbre_log}", mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
```

---

## 3) Función en `main.py`: carga de dataset

En esta etapa creamos una función básica para cargar datos. Es importante documentar claramente los parámetros de entrada y el valor de retorno usando **type hints** y **docstrings**.

**Conceptos clave:**

- **Type hints**: Especifican el tipo de datos esperado (`str`, `pd.DataFrame`)
- **Docstring**: Documentación integrada que explica qué hace la función
- **Valor de retorno**: Siempre especificar qué devuelve la función

```python
# main.py (paso didáctico inicial)

import pandas as pd

def cargar_datos(path: str) -> pd.DataFrame:
    '''
    Carga un CSV desde 'path' y retorna un pandas.DataFrame.
    '''
    df = pd.read_csv(path)
    return df

if __name__ == "__main__":
    df = cargar_datos("data/competencia_01.csv")
    print(df.head())
```

---

## 4) Crear `src/loader.py` y trasladar la función

La **modularización** es clave en el desarrollo de software. Movemos la función de carga a un módulo especializado para:

**Ventajas de la modularización:**

- **Reutilización**: La función puede usarse en otros scripts
- **Mantenimiento**: Cambios en la lógica de carga solo afectan un archivo
- **Testing**: Más fácil crear pruebas unitarias para funciones específicas
- **Colaboración**: Diferentes desarrolladores pueden trabajar en módulos separados

**Mejoras en esta versión:**

- **Manejo de errores**: Try-catch para capturar problemas de carga
- **Logging integrado**: Registra el proceso de carga y posibles errores
- **Type hints mejorados**: `pd.DataFrame | None` indica que puede retornar None en caso de error

```python
# src/loader.py
import pandas as pd
import logging

logger = logging.getLogger("__name__")

## Funcion para cargar datos
def cargar_datos(path: str) -> pd.DataFrame | None:

    '''
    Carga un CSV desde 'path' y retorna un pandas.DataFrame.
    '''

    logger.info(f"Cargando dataset desde {path}")
    try:
        df = pd.read_csv(path)
        logger.info(f"Dataset cargado con {df.shape[0]} filas y {df.shape[1]} columnas")
        return df
    except Exception as e:
        logger.error(f"Error al cargar el dataset: {e}")
        raise
```

---

## 5) Importaciones y `__init__.py`

El archivo `__init__.py` convierte una carpeta en un **paquete Python**, permitiendo importaciones limpias y organizadas.

**¿Por qué es importante `__init__.py`?**

- **Reconocimiento de paquete**: Python identifica la carpeta como un módulo importable
- **Importaciones limpias**: Permite usar `from src.loader import cargar_datos`
- **Namespace control**: Define qué funciones están disponibles al importar el paquete
- **Inicialización**: Puede ejecutar código cuando se importa el paquete por primera vez

**Buenas prácticas:**

- Mantener `__init__.py` vacío o con código mínimo de inicialización
- Usar importaciones absolutas para mayor claridad
- Documentar las dependencias entre módulos

En `main.py`:

```python
from src.loader import cargar_datos
from src.features import generar_features
```

Archivo vacío `src/__init__.py`:

```python
# src/__init__.py
# (intencionalmente vacío)
```

---

## 6) `src/features.py`: SQL directo (DuckDB) para generar `Lag`

**Feature Engineering** es el proceso de crear nuevas variables a partir de datos existentes. En este caso, usamos **DuckDB** para generar variables de **lag** (rezago temporal) usando SQL.

**¿Qué son las variables Lag?**

- **Lag 1**: Valor de la variable en el período anterior
- **Lag 2**: Valor de la variable hace 2 períodos
- **Utilidad**: Capturan patrones temporales y tendencias históricas

**¿Por qué usar DuckDB?**

- **Performance**: Mucho más rápido que pandas para operaciones complejas
- **SQL familiar**: Sintaxis conocida para transformaciones de datos
- **Integración**: Se conecta perfectamente con pandas DataFrames
- **Window functions**: Funciones como `LAG()` y `PARTITION BY` son nativas

**Conceptos SQL clave:**

- `LAG(columna, n)`: Obtiene el valor n períodos atrás
- `PARTITION BY`: Agrupa los datos (por cliente en este caso)
- `ORDER BY`: Define el orden temporal (por foto_mes)

```python
# src/features.py
import pandas as pd
import duckdb
import logging

logger = logging.getLogger("__name__")

def feature_engineering_lag(df: pd.DataFrame, columnas: list[str], cant_lag: int = 1) -> pd.DataFrame:
    """
    Genera variables de lag para los atributos especificados utilizando SQL.
  
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    columnas : list
        Lista de atributos para los cuales generar lags. Si es None, no se generan lags.
    cant_lag : int, default=1
        Cantidad de lags a generar para cada atributo
  
    Returns:
    --------
    pd.DataFrame
        DataFrame con las variables de lag agregadas
    """

    logger.info(f"Realizando feature engineering con {cant_lag} lags para {len(columnas) if columnas else 0} atributos")

    if columnas is None or len(columnas) == 0:
        logger.warning("No se especificaron atributos para generar lags")
        return df
  
    # Construir la consulta SQL
    sql = "SELECT *"
  
    # Agregar los lags para los atributos especificados
    for attr in columnas:
        if attr in df.columns:
            for i in range(1, cant_lag + 1):
                sql += f", lag({attr}, {i}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) AS {attr}_lag_{i}"
        else:
            logger.warning(f"El atributo {attr} no existe en el DataFrame")
  
    # Completar la consulta
    sql += " FROM df"

    logger.debug(f"Consulta SQL: {sql}")

    # Ejecutar la consulta SQL
    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df = con.execute(sql).df()
    con.close()

    print(df.head())
  
    logger.info(f"Feature engineering completado. DataFrame resultante con {df.shape[1]} columnas")

    return df
```

---

## 7) `main.py` orquestando todo con logs

Esta es la **orquestación final** donde integramos todos los módulos creados. El archivo `main.py` actúa como el **director de orquesta**, coordinando el flujo completo del procesamiento de datos.

**Patrón de orquestación:**

1. **Configuración inicial**: Setup de logging y directorios
2. **Carga de datos**: Usando el módulo `loader.py`
3. **Transformación**: Aplicando feature engineering con `features.py`
4. **Persistencia**: Guardando los resultados procesados
5. **Logging completo**: Registrando cada paso del proceso

**Beneficios de este enfoque:**

- **Trazabilidad**: Cada operación queda registrada en logs
- **Modularidad**: Cada responsabilidad está en su módulo correspondiente
- **Mantenibilidad**: Fácil modificar o extender funcionalidades
- **Debugging**: Los logs facilitan identificar problemas
- **Profesionalismo**: Estructura típica de proyectos de producción

```python
# main.py
import pandas as pd
import os
import datetime
import logging

from src.loader import cargar_datos
from src.features import feature_engineering_lag

## config basico logging
os.makedirs("logs", exist_ok=True)

fecha = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
monbre_log = f"log_{fecha}.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s %(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/{monbre_log}", mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

## Funcion principal
def main():
    logger.info("Inicio de ejecucion.")

    #00 Cargar datos
    os.makedirs("data", exist_ok=True)
    path = "data/competencia_01.csv"
    df = cargar_datos(path)   

    #01 Feature Engineering
    atributos = ["ctrx_quarter"]
    cant_lag = 2
    df = feature_engineering_lag(df, columnas=atributos, cant_lag=cant_lag)
  
    #02 Guardar datos
    path = "data/competencia_01_lag.csv"
    df.to_csv(path, index=False)
  
    logger.info(f">>> Ejecución finalizada. Revisar logs para mas detalles.{monbre_log}")

if __name__ == "__main__":
    main()
```

---

📦 **Dependencias necesarias**:

```bash
pip install pandas duckdb
```
