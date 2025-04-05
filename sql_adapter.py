"""
Adaptador para la base de datos MySQL para binance-LOB.
Este módulo proporciona funciones para interactuar con la base de datos MySQL,
permitiendo guardar y recuperar datos del libro de órdenes para su análisis.
"""

import json
import mysql.connector
from mysql.connector import Error
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging
import numpy as np
from mysql_config import MYSQL_CONFIG

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MySQLAdapter:
    """Adaptador para interactuar con la base de datos MySQL."""
    
    def __init__(self):
        """Inicializa el adaptador de MySQL."""
        self.connection = None
        self.connect()
    
    def connect(self):
        """Establece la conexión con la base de datos MySQL."""
        try:
            self.connection = mysql.connector.connect(
                host=MYSQL_CONFIG["host"],
                user=MYSQL_CONFIG["user"],
                password=MYSQL_CONFIG["password"],
                database=MYSQL_CONFIG["database"]
            )
            logger.info("Conexión a MySQL establecida correctamente")
        except Error as e:
            logger.error(f"Error al conectar a MySQL: {e}")
    
    def disconnect(self):
        """Cierra la conexión con la base de datos MySQL."""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("Conexión a MySQL cerrada")
    
    def reconnect_if_needed(self):
        """Reconecta a la base de datos si la conexión se ha perdido."""
        if not self.connection or not self.connection.is_connected():
            logger.info("Reconectando a MySQL...")
            self.connect()
    
    def save_depth_snapshot(self, symbol: str, timestamp: datetime, 
                           bids: List[Tuple[float, float]], 
                           asks: List[Tuple[float, float]]) -> bool:
        """
        Guarda un snapshot completo del libro de órdenes en la base de datos.
        
        Args:
            symbol: Símbolo del par de trading (ej. "BTCUSDT")
            timestamp: Marca de tiempo del snapshot
            bids: Lista de tuplas (precio, cantidad) para órdenes de compra
            asks: Lista de tuplas (precio, cantidad) para órdenes de venta
            
        Returns:
            bool: True si se guardó correctamente, False en caso contrario
        """
        self.reconnect_if_needed()
        
        try:
            cursor = self.connection.cursor()
            
            # Separar precios y cantidades
            bids_price = [bid[0] for bid in bids]
            bids_quantity = [bid[1] for bid in bids]
            asks_price = [ask[0] for ask in asks]
            asks_quantity = [ask[1] for ask in asks]
            
            # Convertir a JSON para almacenamiento
            bids_price_json = json.dumps(bids_price)
            bids_quantity_json = json.dumps(bids_quantity)
            asks_price_json = json.dumps(asks_price)
            asks_quantity_json = json.dumps(asks_quantity)
            
            query = """
            INSERT INTO DepthSnapshot 
            (symbol, timestamp, bids_price, bids_quantity, asks_price, asks_quantity)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            
            cursor.execute(query, (
                symbol, 
                timestamp, 
                bids_price_json, 
                bids_quantity_json, 
                asks_price_json, 
                asks_quantity_json
            ))
            
            self.connection.commit()
            logger.info(f"Snapshot guardado para {symbol} a las {timestamp}")
            return True
            
        except Error as e:
            logger.error(f"Error al guardar snapshot: {e}")
            return False
    
    def save_depth_update(self, symbol: str, timestamp: datetime, 
                         update_type: str, side: str, 
                         price: float, quantity: float) -> bool:
        """
        Guarda una actualización incremental del libro de órdenes.
        
        Args:
            symbol: Símbolo del par de trading (ej. "BTCUSDT")
            timestamp: Marca de tiempo de la actualización
            update_type: Tipo de actualización ("add", "remove", "update")
            side: Lado del libro ("bid" para compra, "ask" para venta)
            price: Precio de la orden actualizada
            quantity: Cantidad de la orden actualizada
            
        Returns:
            bool: True si se guardó correctamente, False en caso contrario
        """
        self.reconnect_if_needed()
        
        try:
            cursor = self.connection.cursor()
            
            query = """
            INSERT INTO DiffDepthStream 
            (symbol, timestamp, update_type, side, price, quantity)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            
            cursor.execute(query, (
                symbol, 
                timestamp, 
                update_type, 
                side, 
                price, 
                quantity
            ))
            
            self.connection.commit()
            logger.debug(f"Actualización guardada para {symbol} a las {timestamp}")
            return True
            
        except Error as e:
            logger.error(f"Error al guardar actualización: {e}")
            return False
    
    def get_latest_snapshot(self, symbol: str, timestamp: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
        """
        Obtiene el snapshot más reciente del libro de órdenes.
        
        Args:
            symbol: Símbolo del par de trading (ej. "BTCUSDT")
            timestamp: Si se proporciona, obtiene el snapshot más reciente antes de esta marca de tiempo
            
        Returns:
            Dict o None: Datos del snapshot o None si no se encuentra
        """
        self.reconnect_if_needed()
        
        try:
            cursor = self.connection.cursor(dictionary=True)
            
            if timestamp:
                query = """
                SELECT * FROM DepthSnapshot 
                WHERE symbol = %s AND timestamp <= %s 
                ORDER BY timestamp DESC LIMIT 1
                """
                cursor.execute(query, (symbol, timestamp))
            else:
                query = """
                SELECT * FROM DepthSnapshot 
                WHERE symbol = %s 
                ORDER BY timestamp DESC LIMIT 1
                """
                cursor.execute(query, (symbol,))
            
            result = cursor.fetchone()
            
            if result:
                # Convertir JSON a listas
                result['bids_price'] = json.loads(result['bids_price'])
                result['bids_quantity'] = json.loads(result['bids_quantity'])
                result['asks_price'] = json.loads(result['asks_price'])
                result['asks_quantity'] = json.loads(result['asks_quantity'])
                
                return result
            return None
            
        except Error as e:
            logger.error(f"Error al obtener snapshot: {e}")
            return None
    
    def get_snapshots_in_range(self, symbol: str, start_time: datetime, 
                              end_time: datetime) -> List[Dict[str, Any]]:
        """
        Obtiene todos los snapshots en un rango de tiempo.
        
        Args:
            symbol: Símbolo del par de trading (ej. "BTCUSDT")
            start_time: Inicio del rango de tiempo
            end_time: Fin del rango de tiempo
            
        Returns:
            List[Dict]: Lista de snapshots en el rango especificado
        """
        self.reconnect_if_needed()
        
        try:
            cursor = self.connection.cursor(dictionary=True)
            
            query = """
            SELECT * FROM DepthSnapshot 
            WHERE symbol = %s AND timestamp >= %s AND timestamp <= %s 
            ORDER BY timestamp
            """
            
            cursor.execute(query, (symbol, start_time, end_time))
            results = cursor.fetchall()
            
            for result in results:
                # Convertir JSON a listas
                result['bids_price'] = json.loads(result['bids_price'])
                result['bids_quantity'] = json.loads(result['bids_quantity'])
                result['asks_price'] = json.loads(result['asks_price'])
                result['asks_quantity'] = json.loads(result['asks_quantity'])
            
            return results
            
        except Error as e:
            logger.error(f"Error al obtener snapshots en rango: {e}")
            return []
    
    def get_updates_in_range(self, symbol: str, start_time: datetime, 
                           end_time: datetime) -> List[Dict[str, Any]]:
        """
        Obtiene todas las actualizaciones en un rango de tiempo.
        
        Args:
            symbol: Símbolo del par de trading (ej. "BTCUSDT")
            start_time: Inicio del rango de tiempo
            end_time: Fin del rango de tiempo
            
        Returns:
            List[Dict]: Lista de actualizaciones en el rango especificado
        """
        self.reconnect_if_needed()
        
        try:
            cursor = self.connection.cursor(dictionary=True)
            
            query = """
            SELECT * FROM DiffDepthStream 
            WHERE symbol = %s AND timestamp >= %s AND timestamp <= %s 
            ORDER BY timestamp
            """
            
            cursor.execute(query, (symbol, start_time, end_time))
            return cursor.fetchall()
            
        except Error as e:
            logger.error(f"Error al obtener actualizaciones en rango: {e}")
            return []
    
    def count_updates_in_range(self, symbol: str, start_time: datetime, 
                             end_time: datetime) -> int:
        """
        Cuenta el número de actualizaciones en un rango de tiempo.
        
        Args:
            symbol: Símbolo del par de trading (ej. "BTCUSDT")
            start_time: Inicio del rango de tiempo
            end_time: Fin del rango de tiempo
            
        Returns:
            int: Número de actualizaciones en el rango especificado
        """
        self.reconnect_if_needed()
        
        try:
            cursor = self.connection.cursor()
            
            query = """
            SELECT COUNT(*) FROM DiffDepthStream 
            WHERE symbol = %s AND timestamp >= %s AND timestamp <= %s
            """
            
            cursor.execute(query, (symbol, start_time, end_time))
            result = cursor.fetchone()
            
            return result[0] if result else 0
            
        except Error as e:
            logger.error(f"Error al contar actualizaciones: {e}")
            return 0
    
    def log_message(self, message: str) -> bool:
        """
        Guarda un mensaje de log en la base de datos.
        
        Args:
            message: Mensaje a guardar
            
        Returns:
            bool: True si se guardó correctamente, False en caso contrario
        """
        self.reconnect_if_needed()
        
        try:
            cursor = self.connection.cursor()
            
            query = """
            INSERT INTO LoggingMsg (timestamp, message)
            VALUES (%s, %s)
            """
            
            cursor.execute(query, (datetime.now(), message))
            self.connection.commit()
            
            return True
            
        except Error as e:
            logger.error(f"Error al guardar mensaje de log: {e}")
            return False
