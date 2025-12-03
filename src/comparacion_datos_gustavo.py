from google.cloud import bigquery
import logging
import os

# Configuración básica de logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def join_features_and_targets(
        project_id: str,
        dataset_id: str,
        source_table_id: str,
        target_table_id: str = "targets_gustavo",
        output_table_id: str = "dataset_final_training"
) -> None:
    """
    Ejecuta una consulta en BigQuery para unir una tabla de features
    con la tabla de targets, excluyendo la columna 'clase_ternaria' de la
    tabla de features y seleccionándola únicamente de la tabla de targets.

    Args:
        project_id: ID del proyecto de Google Cloud.
        dataset_id: ID del dataset donde se encuentran las tablas.
        source_table_id: ID de la tabla de features (la que contiene los datos).
        target_table_id: ID de la tabla de targets (por defecto, 'targets_gustavo').
        output_table_id: ID de la nueva tabla donde se guardará el resultado del join.
    """

    logger.info(f"Iniciando JOIN entre {source_table_id} y {target_table_id}...")

    try:
        # Inicializar el cliente de BigQuery
        client = bigquery.Client(project=project_id)

        # Definición de las referencias completas de las tablas
        source_ref = f"`{project_id}.{dataset_id}.{source_table_id}`"
        target_ref = f"`{project_id}.{dataset_id}.{target_table_id}`"
        output_ref = f"`{project_id}.{dataset_id}.{output_table_id}`"

        # --- Construcción de la Query SQL ---

        query = f"""
        CREATE OR REPLACE TABLE {output_ref} 
        AS
        SELECT 
            t1.* EXCEPT(clase_ternaria), -- Selecciona todas las columnas de t1 EXCEPTO clase_ternaria
            t2.clase_ternaria           -- Selecciona SOLO clase_ternaria de t2
        FROM 
            {source_ref} AS t1
        INNER JOIN 
            {target_ref} AS t2
        ON 
            t1.foto_mes = t2.foto_mes AND t1.numero_de_cliente = t2.numero_de_cliente
        """

        # Ejecución de la consulta
        job = client.query(query)  # Lanza la query
        job.result()  # Espera a que termine la query

        logger.info(f"✅ Query ejecutada exitosamente. El resultado se guardó en {output_table_id}")

        # Opcional: imprimir el número de filas de la tabla resultante
        rows = client.get_table(f"{dataset_id}.{output_table_id}").num_rows
        logger.info(f"La tabla resultante '{output_table_id}' contiene {rows} filas.")

    except Exception as e:
        logger.error(f"❌ Error al ejecutar la query de BigQuery: {e}")
        # En caso de error, puedes querer re-lanzar la excepción
        raise