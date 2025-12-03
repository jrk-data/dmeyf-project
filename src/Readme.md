# 📉 Proyecto de Predicción de Churn (DMEyF)

Este proyecto implementa un pipeline completo de Ciencia de Datos para la predicción de bajas de clientes bancarios (Churn). El flujo abarca desde la creación de la base de datos en la nube hasta la predicción final, orquestado mediante archivos de configuración YAML.

## 📋 Requisitos Previos

1.  **Entorno Python:** Python 3.10+ recomendado.
2.  **Librerías:** Instalar dependencias necesarias (incluyendo el soporte para Google Storage).
    ```bash
    pip install -r requirements.txt
    pip install gcsfs
    ```
3.  **Credenciales GCP:** Tener configurado el acceso a Google Cloud (BigQuery y Storage) en el entorno donde se ejecuta (VM o local).

---

## ⚙️ Cómo Ejecutar el Proyecto

El proyecto se controla desde un único punto de entrada: **`config.yaml`**.

Para cambiar de etapa (Crear Datos, Optimizar, Entrenar o Predecir), solo debes modificar la variable `EXPERIMENT_FILE` en `config.yaml` apuntando al archivo de configuración específico de esa etapa.

### Paso 1: Configurar el Controlador
Abre `config.yaml` y edita la línea:

```yaml
# Ejemplo para correr la creación de base de datos
EXPERIMENT_FILE: "experiments/01_crear_base.yaml"
```
### Paso 2: Ejecutar el Script
Corre el script principal. Usa el flag -vm si estás en la máquina virtual de Google Cloud para usar las rutas correctas:

```bash
python main.py -vm
```


---

## 🚀 Flujo de Trabajo (Paso a Paso)
Existen 4 etapas secuenciales. Para ejecutar una, edita config.yaml y asigna el archivo correspondiente:

1. Creación de Base de Datos (ETL)
Carga datos crudos desde GCS, los procesa con DuckDB y crea/consolida las tablas en BigQuery con ingeniería de características (lags, deltas, targets).

- Archivo YAML: experiments/01_crear_base.yaml
- Acción en config.yaml:

```YAML
EXPERIMENT_FILE: "experiments/corrida_c03_DB.yaml"
```
- Resultado: Tablas c03 (consolidada), targets y c03_features_historical creadas en BigQuery.

2. Optimización Bayesiana (Optuna)
Busca los mejores hiperparámetros para LightGBM utilizando validación cruzada, respetando los meses de entrenamiento y validación definidos para evitar data leakage.

- Archivo YAML: experiments/02_optimizacion.yaml
- Acción en config.yaml:

```YAML
EXPERIMENT_FILE: "experiments/corrida_c03_OPT.yaml"
```
- Resultado: Hiperparámetros guardados en la base de datos SQLite de Optuna.

3. Entrenamiento y Evaluación (Curvas de Ganancia)
Entrena los mejores modelos (Top K) con los parámetros encontrados, genera predicciones sobre un mes de testeo y grafica las curvas de ganancia para decidir el corte de envíos.

- Archivo YAML: experiments/03_train_test.yaml
- Acción en config.yaml:

```YAML
EXPERIMENT_FILE: "experiments/corrida_c03_Test.yaml"
```
- Resultado: 
  - Modelos .txt guardados en la carpeta models/.

  - Gráficos de ganancia en models/.../curvas_de_complejidad/.

  - Archivo resumen_ganancias.csv (Metadata de los modelos, necesaria si se usa lógica de reporte).

4. Predicción Final (Producción)
Utiliza los modelos entrenados (generalmente con estrategia Retraining for Production incluyendo meses recientes) para predecir sobre un mes futuro (ej. Septiembre 2021). Aplica ensamble (Soft Voting) y genera el archivo de entrega.

- Archivo YAML: experiments/04_prediccion_final.yaml
- Acción en config.yaml:

```YAML
EXPERIMENT_FILE: "experiments/corrida_c03_Predict.yaml"
```
- Nota: Hay que correrlo primero con START_POINT: "TRAIN" (en el yaml corrida_c03_Predict) para re-entrenar con toda la historia, y luego cambiar a START_POINT: "PREDICT" para generar el archivo final.

- Resultado: Archivo .csv con las predicciones en la carpeta outputs/.

