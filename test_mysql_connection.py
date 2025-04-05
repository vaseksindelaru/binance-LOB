"""
Script para probar la conexión a la base de datos MySQL y configurar las tablas.
"""

import sys
import logging
from datetime import datetime, timedelta
import random
import json
import numpy as np
from setup_database import create_connection, create_database, create_tables
from sql_adapter import MySQLAdapter
from mysql_analyzer import MySQLOrderBookAnalyzer

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_test_data(symbol: str, num_snapshots: int = 10, num_updates_per_snapshot: int = 5):
    """
    Genera datos de prueba para el libro de órdenes.
    
    Args:
        symbol: Símbolo del par de trading (ej. "BTCUSDT")
        num_snapshots: Número de snapshots a generar
        num_updates_per_snapshot: Número de actualizaciones por snapshot
        
    Returns:
        tuple: (snapshots, updates) donde cada uno es una lista de diccionarios
    """
    snapshots = []
    updates = []
    
    # Precio base para el símbolo
    base_price = 50000.0 if symbol == "BTCUSDT" else 2000.0
    
    # Generar snapshots
    start_time = datetime.now() - timedelta(hours=num_snapshots)
    
    for i in range(num_snapshots):
        timestamp = start_time + timedelta(minutes=i*10)
        
        # Generar precios y cantidades para bids (órdenes de compra)
        bids_price = [base_price - j * 0.5 - random.random() for j in range(10)]
        bids_quantity = [random.uniform(0.1, 2.0) for _ in range(10)]
        
        # Generar precios y cantidades para asks (órdenes de venta)
        asks_price = [base_price + j * 0.5 + random.random() for j in range(10)]
        asks_quantity = [random.uniform(0.1, 2.0) for _ in range(10)]
        
        # Crear snapshot
        snapshot = {
            'symbol': symbol,
            'timestamp': timestamp,
            'bids_price': bids_price,
            'bids_quantity': bids_quantity,
            'asks_price': asks_price,
            'asks_quantity': asks_quantity
        }
        
        snapshots.append(snapshot)
        
        # Generar actualizaciones para este snapshot
        for j in range(num_updates_per_snapshot):
            update_timestamp = timestamp + timedelta(seconds=j*30)
            
            # Decidir aleatoriamente el tipo de actualización
            update_type = random.choice(['add', 'remove', 'update'])
            side = random.choice(['bid', 'ask'])
            
            # Seleccionar un precio aleatorio cerca del precio base
            if side == 'bid':
                price = base_price - random.uniform(0, 5)
            else:
                price = base_price + random.uniform(0, 5)
            
            # Cantidad para la actualización
            quantity = random.uniform(0.1, 1.0)
            
            # Crear actualización
            update = {
                'symbol': symbol,
                'timestamp': update_timestamp,
                'update_type': update_type,
                'side': side,
                'price': price,
                'quantity': quantity
            }
            
            updates.append(update)
    
    return snapshots, updates

def insert_test_data(adapter: MySQLAdapter, symbol: str):
    """
    Inserta datos de prueba en la base de datos.
    
    Args:
        adapter: Adaptador de MySQL
        symbol: Símbolo del par de trading (ej. "BTCUSDT")
    """
    # Generar datos de prueba
    snapshots, updates = generate_test_data(symbol)
    
    # Insertar snapshots
    for snapshot in snapshots:
        # Convertir a formato de tuplas para save_depth_snapshot
        bids = list(zip(snapshot['bids_price'], snapshot['bids_quantity']))
        asks = list(zip(snapshot['asks_price'], snapshot['asks_quantity']))
        
        adapter.save_depth_snapshot(
            symbol=snapshot['symbol'],
            timestamp=snapshot['timestamp'],
            bids=bids,
            asks=asks
        )
    
    # Insertar actualizaciones
    for update in updates:
        adapter.save_depth_update(
            symbol=update['symbol'],
            timestamp=update['timestamp'],
            update_type=update['update_type'],
            side=update['side'],
            price=update['price'],
            quantity=update['quantity']
        )
    
    logger.info(f"Insertados {len(snapshots)} snapshots y {len(updates)} actualizaciones para {symbol}")

def test_analyzer(analyzer: MySQLOrderBookAnalyzer, symbol: str):
    """
    Prueba las funcionalidades del analizador.
    
    Args:
        analyzer: Analizador de libro de órdenes
        symbol: Símbolo del par de trading (ej. "BTCUSDT")
    """
    # Definir rango de tiempo para las pruebas
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=2)
    
    # Probar ratio de compra/venta
    try:
        ratio = analyzer.get_buy_sell_ratio(symbol)
        logger.info(f"Ratio de compra/venta para {symbol}: {ratio}")
    except Exception as e:
        logger.error(f"Error al obtener ratio de compra/venta: {e}")
    
    # Probar velocidad de cambio
    try:
        change_rate = analyzer.get_order_book_change_rate(symbol, start_time, end_time)
        logger.info(f"Velocidad de cambio para {symbol}: {change_rate}")
    except Exception as e:
        logger.error(f"Error al obtener velocidad de cambio: {e}")
    
    # Probar tamaño medio de órdenes
    try:
        avg_sizes = analyzer.get_average_order_size(symbol)
        logger.info(f"Tamaño medio de órdenes para {symbol}: {avg_sizes}")
    except Exception as e:
        logger.error(f"Error al obtener tamaño medio de órdenes: {e}")
    
    # Probar profundidad del mercado
    try:
        depth = analyzer.get_market_depth(symbol)
        logger.info(f"Profundidad del mercado para {symbol}: {depth}")
    except Exception as e:
        logger.error(f"Error al obtener profundidad del mercado: {e}")
    
    # Probar tiempo de vida de órdenes
    try:
        lifetime = analyzer.estimate_order_lifetime(symbol, start_time, end_time)
        logger.info(f"Tiempo de vida de órdenes para {symbol}: {lifetime}")
    except Exception as e:
        logger.error(f"Error al estimar tiempo de vida de órdenes: {e}")

def main():
    """Función principal para probar la conexión y funcionalidades."""
    logger.info("Iniciando prueba de conexión a MySQL...")
    
    # Crear conexión y configurar base de datos
    connection = create_connection()
    if not connection:
        logger.error("No se pudo establecer conexión con MySQL. Abortando.")
        sys.exit(1)
    
    create_database(connection)
    create_tables(connection)
    connection.close()
    
    logger.info("Base de datos configurada correctamente.")
    
    # Crear adaptador y analizador
    adapter = MySQLAdapter()
    analyzer = MySQLOrderBookAnalyzer()
    
    # Símbolos para probar
    symbols = ["BTCUSDT", "ETHUSDT"]
    
    # Insertar datos de prueba para cada símbolo
    for symbol in symbols:
        insert_test_data(adapter, symbol)
    
    # Probar analizador para cada símbolo
    for symbol in symbols:
        logger.info(f"\n--- Probando análisis para {symbol} ---")
        test_analyzer(analyzer, symbol)
    
    # Cerrar conexión
    adapter.disconnect()
    logger.info("Prueba completada con éxito.")

if __name__ == "__main__":
    main()
