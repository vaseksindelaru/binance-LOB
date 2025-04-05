"""
Script para configurar la base de datos MySQL para binance-LOB.
Este script crea las tablas necesarias para almacenar los datos del libro de órdenes
y realizar los análisis de los cinco parámetros implementados.
"""

import mysql.connector
from mysql.connector import Error
from mysql_config import MYSQL_CONFIG
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_connection():
    """Crea una conexión a la base de datos MySQL."""
    try:
        connection = mysql.connector.connect(
            host=MYSQL_CONFIG["host"],
            user=MYSQL_CONFIG["user"],
            password=MYSQL_CONFIG["password"]
        )
        logger.info("Conexión a MySQL establecida correctamente")
        return connection
    except Error as e:
        logger.error(f"Error al conectar a MySQL: {e}")
        return None

def create_database(connection):
    """Crea la base de datos si no existe."""
    try:
        cursor = connection.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {MYSQL_CONFIG['database']}")
        logger.info(f"Base de datos '{MYSQL_CONFIG['database']}' creada o ya existente")
        cursor.execute(f"USE {MYSQL_CONFIG['database']}")
    except Error as e:
        logger.error(f"Error al crear la base de datos: {e}")

def create_tables(connection):
    """Crea las tablas necesarias para el análisis del libro de órdenes."""
    try:
        cursor = connection.cursor()
        
        # Tabla para almacenar snapshots completos del libro de órdenes
        create_depth_snapshot_table = """
        CREATE TABLE IF NOT EXISTS DepthSnapshot (
            id INT AUTO_INCREMENT PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            timestamp DATETIME NOT NULL,
            bids_price JSON NOT NULL,
            bids_quantity JSON NOT NULL,
            asks_price JSON NOT NULL,
            asks_quantity JSON NOT NULL,
            INDEX idx_symbol_timestamp (symbol, timestamp)
        ) ENGINE=InnoDB;
        """
        
        # Tabla para almacenar actualizaciones incrementales del libro de órdenes
        create_diff_depth_stream_table = """
        CREATE TABLE IF NOT EXISTS DiffDepthStream (
            id INT AUTO_INCREMENT PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            timestamp DATETIME NOT NULL,
            update_type VARCHAR(10) NOT NULL,
            side VARCHAR(4) NOT NULL,
            price FLOAT NOT NULL,
            quantity FLOAT NOT NULL,
            INDEX idx_symbol_timestamp (symbol, timestamp)
        ) ENGINE=InnoDB;
        """
        
        # Tabla opcional para logging
        create_logging_msg_table = """
        CREATE TABLE IF NOT EXISTS LoggingMsg (
            id INT AUTO_INCREMENT PRIMARY KEY,
            timestamp DATETIME NOT NULL,
            message TEXT NOT NULL,
            INDEX idx_timestamp (timestamp)
        ) ENGINE=InnoDB;
        """
        
        # Ejecutar las consultas de creación de tablas
        cursor.execute(create_depth_snapshot_table)
        logger.info("Tabla 'DepthSnapshot' creada o ya existente")
        
        cursor.execute(create_diff_depth_stream_table)
        logger.info("Tabla 'DiffDepthStream' creada o ya existente")
        
        cursor.execute(create_logging_msg_table)
        logger.info("Tabla 'LoggingMsg' creada o ya existente")
        
        connection.commit()
    except Error as e:
        logger.error(f"Error al crear las tablas: {e}")

def main():
    """Función principal para configurar la base de datos."""
    connection = create_connection()
    if connection:
        create_database(connection)
        create_tables(connection)
        connection.close()
        logger.info("Configuración de la base de datos completada")
    else:
        logger.error("No se pudo establecer conexión con MySQL")

if __name__ == "__main__":
    main()
