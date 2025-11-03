from google.cloud import bigquery

from src.config import (BQ_PROJECT, BQ_DATASET, BQ_TABLE, BQ_TABLE_TARGETS)


def run_bigquery_update(query: str, project_id: str):
    client = bigquery.Client(project=project_id)
    print("🚀 Ejecutando query en BigQuery...\n")
    job = client.query(query)
    job.result()  # Espera a que termine la ejecución
    print("✅ Query ejecutada correctamente.")
    print(f"Job ID: {job.job_id}")

def select_bigquery(query: str, project_id: str):
    client = bigquery.Client(project=project_id)
    print("🚀 Ejecutando query en BigQuery...\n")
    job = client.query(query)
    job.result()  # Espera a que termine la ejecución
    print("✅ Query ejecutada correctamente.")
    print(f"Job ID: {job.job_id}")

if __name__ == '__main__':
    query = f'SELECT * FROM `{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE_TARGETS}` where foto_mes = 202003 LIMIT 10'
    #print(query)
    select = select_bigquery(query, BQ_PROJECT)
    print(select)