import duckdb
import os
from src.config import DB_PATH, CSV_COMP
from pathlib import Path


print(DB_PATH)



def create_dataset_c01(DB_PATH,CSV_COMP = CSV_COMP):
    CSV_COMP = CSV_COMP
    try:

        con = duckdb.connect(str(DB_PATH))


        query_competencia_01 = f'''
        CREATE OR REPLACE TABLE competencia_01 AS
        SELECT * FROM read_csv_auto('{CSV_COMP}');
        
        ALTER TABLE competencia_01
        ADD COLUMN IF NOT EXISTS clase_ternaria VARCHAR DEFAULT NULL;
        '''
        con.sql(query_competencia_01)

        query_crear_categorias = '''
        create or replace table clases_ternarias as
        with usuarios_ultimo_a_primer_es as(
          select
          foto_mes
          , numero_de_cliente
          , row_number() over (partition by numero_de_cliente order by foto_mes desc) as row_number
          from competencia_01
        ) select
        foto_mes
        ,numero_de_cliente
        , case
        when row_number = 1 and foto_mes < 202106 then 'BAJA+1'
        when row_number = 2 and foto_mes < 202105 then 'BAJA+2'
        when row_number >= 3 then 'CONTINUA'
        else null
        end as clase_ternaria
        from usuarios_ultimo_a_primer_es; 
        '''

        query_update_competencia_01 = '''
        update competencia_01
        set clase_ternaria = clases_ternarias.clase_ternaria
        from clases_ternarias  
        where competencia_01.numero_de_cliente = clases_ternarias.numero_de_cliente and competencia_01.foto_mes = clases_ternarias.foto_mes;
        '''

        con.sql(query_crear_categorias)

        con.sql(query_update_competencia_01)

        con.close()
    except Exception as e:
        con.close()
        print(e)
    finally:
        con.close()


def select_c01(DB_PATH):
    try:
        con = duckdb.connect(str(DB_PATH))
        query = con.sql("SELECT * FROM competencia_01").pl()
        con.close()

    except Exception as e:
        con.close()
        print(e)
    finally:
        con.close()
    return query