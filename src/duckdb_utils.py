try:
    with duckdb.connect(str(DB_MODELS_TRAIN_PATH)) as con:

        # 2. Definir el nombre de la tabla
        TABLE_NAME = resumen_path + '_test'

        # 3. Crear la tabla (solo si no existe) a partir del esquema de 'meta'
        try:
            # Usar 'meta' para inferir el esquema
            # La sintaxis 'AS SELECT * FROM meta WHERE 1=0' crea la tabla vacía
            # con la estructura de tu DataFrame 'meta'.
            con.execute(f"CREATE TABLE IF NOT EXISTS {TABLE_NAME} AS SELECT * FROM meta WHERE 1=0;")
            logger.info(f"Tabla '{TABLE_NAME}' asegurada (creada si no existía).")

        except Exception as e:
            # Capturar errores específicos de la creación de la tabla si es necesario,
            # aunque 'CREATE IF NOT EXISTS' suele ser seguro.
            logger.error(f"Error al crear la tabla {TABLE_NAME}: {e}")
            # Si la creación falla aquí, el flujo saldrá del 'try' principal.
            raise  # Re-lanza la excepción para que el programa se detenga si no puede crear la tabla

        # 4. Insertar los datos
        # Si la creación fue exitosa, o si la tabla ya existía, procedemos a insertar.
        # 'meta' es el DataFrame de Pandas en memoria
        con.execute(f"INSERT INTO {TABLE_NAME} SELECT * FROM meta;")
        logger.info(f"Datos insertados en la tabla '{TABLE_NAME}'.")

except Exception as e:
    # Este bloque maneja cualquier error que ocurra fuera de la creación de la tabla (ej. error de conexión, error de inserción)
    logger.error(f"Error general en la operación de DuckDB: {e}")