"""
Adaptador para el analizador del libro de órdenes usando MySQL.
Este módulo proporciona una implementación de OrderBookAnalyzer que utiliza
la base de datos MySQL en lugar de ClickHouse.
"""

from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import numpy as np
import json
import logging
from sql_adapter import MySQLAdapter
from mysql_config import MYSQL_CONFIG

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MySQLOrderBookAnalyzer:
    """
    Implementación del analizador del libro de órdenes que utiliza MySQL.
    """
    
    def __init__(self):
        """Inicializa el analizador con una conexión a MySQL."""
        self.db_adapter = MySQLAdapter()
    
    def get_buy_sell_ratio(self, symbol: str, timestamp: Optional[datetime] = None,
                           lookback_seconds: int = 60) -> Dict[str, float]:
        """
        Calcula la relación entre el volumen total de órdenes de compra y venta.
        
        Args:
            symbol: Símbolo del par de trading (ej. "BTCUSDT")
            timestamp: Marca de tiempo para el análisis (usa la más reciente si es None)
            lookback_seconds: Segundos hacia atrás para considerar en el análisis
            
        Returns:
            Dict con la relación de compra/venta y volúmenes totales
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Obtener el snapshot más reciente antes del timestamp
        snapshot = self.db_adapter.get_latest_snapshot(symbol, timestamp)
        
        if not snapshot:
            logger.warning(f"No se encontró snapshot para {symbol} antes de {timestamp}")
            return {"buy_sell_ratio": 0.0, "total_bid_volume": 0.0, "total_ask_volume": 0.0}
        
        # Calcular volumen total de bids y asks
        bids_quantity = snapshot['bids_quantity']
        asks_quantity = snapshot['asks_quantity']
        
        total_bid_volume = sum(bids_quantity)
        total_ask_volume = sum(asks_quantity)
        
        # Calcular ratio
        ratio = total_bid_volume / total_ask_volume if total_ask_volume > 0 else float('inf')
        
        return {
            "buy_sell_ratio": ratio,
            "total_bid_volume": total_bid_volume,
            "total_ask_volume": total_ask_volume
        }
    
    def get_order_book_change_rate(self, symbol: str, start_time: datetime, 
                                  end_time: datetime) -> Dict[str, float]:
        """
        Calcula la velocidad de cambio en el libro de órdenes.
        
        Args:
            symbol: Símbolo del par de trading (ej. "BTCUSDT")
            start_time: Inicio del período de análisis
            end_time: Fin del período de análisis
            
        Returns:
            Dict con la tasa de cambio y métricas relacionadas
        """
        # Contar actualizaciones en el rango de tiempo
        update_count = self.db_adapter.count_updates_in_range(symbol, start_time, end_time)
        
        # Calcular duración en segundos
        duration_seconds = (end_time - start_time).total_seconds()
        
        # Calcular tasa de cambio (actualizaciones por segundo)
        change_rate = update_count / duration_seconds if duration_seconds > 0 else 0
        
        return {
            "change_rate": change_rate,  # Actualizaciones por segundo
            "update_count": update_count,
            "duration_seconds": duration_seconds
        }
    
    def get_average_order_size(self, symbol: str, timestamp: Optional[datetime] = None, 
                              side: str = 'both') -> Dict[str, float]:
        """
        Calcula el tamaño promedio de las órdenes en el libro.
        
        Args:
            symbol: Símbolo del par de trading (ej. "BTCUSDT")
            timestamp: Marca de tiempo para el análisis (usa la más reciente si es None)
            side: Lado del libro a analizar ('bid', 'ask', o 'both')
            
        Returns:
            Dict con el tamaño promedio de las órdenes
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Obtener el snapshot más reciente antes del timestamp
        snapshot = self.db_adapter.get_latest_snapshot(symbol, timestamp)
        
        if not snapshot:
            logger.warning(f"No se encontró snapshot para {symbol} antes de {timestamp}")
            return {"average_size": 0.0, "side": side}
        
        # Calcular tamaño promedio según el lado especificado
        if side == 'bid' or side == 'both':
            bids_quantity = snapshot['bids_quantity']
            bid_avg = sum(bids_quantity) / len(bids_quantity) if bids_quantity else 0
        
        if side == 'ask' or side == 'both':
            asks_quantity = snapshot['asks_quantity']
            ask_avg = sum(asks_quantity) / len(asks_quantity) if asks_quantity else 0
        
        # Devolver resultados según el lado solicitado
        if side == 'bid':
            return {"average_size": bid_avg, "side": "bid"}
        elif side == 'ask':
            return {"average_size": ask_avg, "side": "ask"}
        else:  # both
            combined_avg = (bid_avg + ask_avg) / 2
            return {
                "average_size": combined_avg,
                "bid_average": bid_avg,
                "ask_average": ask_avg,
                "side": "both"
            }
    
    def get_market_depth(self, symbol: str, timestamp: Optional[datetime] = None, 
                        depth_levels: int = 10) -> Dict[str, float]:
        """
        Calcula la profundidad del mercado a diferentes niveles de precio.
        
        Args:
            symbol: Símbolo del par de trading (ej. "BTCUSDT")
            timestamp: Marca de tiempo para el análisis (usa la más reciente si es None)
            depth_levels: Número de niveles de precio a considerar
            
        Returns:
            Dict con métricas de profundidad del mercado
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Obtener el snapshot más reciente antes del timestamp
        snapshot = self.db_adapter.get_latest_snapshot(symbol, timestamp)
        
        if not snapshot:
            logger.warning(f"No se encontró snapshot para {symbol} antes de {timestamp}")
            return {"total_depth": 0.0, "bid_depth": 0.0, "ask_depth": 0.0}
        
        # Limitar a los niveles especificados
        bids_price = snapshot['bids_price'][:depth_levels]
        bids_quantity = snapshot['bids_quantity'][:depth_levels]
        asks_price = snapshot['asks_price'][:depth_levels]
        asks_quantity = snapshot['asks_quantity'][:depth_levels]
        
        # Calcular profundidad total (volumen) para bids y asks
        bid_depth = sum(bids_quantity)
        ask_depth = sum(asks_quantity)
        total_depth = bid_depth + ask_depth
        
        # Calcular precio medio ponderado por volumen (VWAP)
        bid_value = sum(p * q for p, q in zip(bids_price, bids_quantity))
        ask_value = sum(p * q for p, q in zip(asks_price, asks_quantity))
        
        bid_vwap = bid_value / bid_depth if bid_depth > 0 else 0
        ask_vwap = ask_value / ask_depth if ask_depth > 0 else 0
        
        # Calcular spread (diferencia entre el mejor precio de compra y venta)
        best_bid = bids_price[0] if bids_price else 0
        best_ask = asks_price[0] if asks_price else 0
        spread = best_ask - best_bid if best_bid > 0 and best_ask > 0 else 0
        
        return {
            "total_depth": total_depth,
            "bid_depth": bid_depth,
            "ask_depth": ask_depth,
            "bid_vwap": bid_vwap,
            "ask_vwap": ask_vwap,
            "spread": spread,
            "best_bid": best_bid,
            "best_ask": best_ask
        }
    
    def estimate_order_lifetime(self, symbol: str, start_time: datetime, end_time: datetime, 
                               price_levels: int = 10, min_volume_threshold: float = 0.1) -> Dict[str, float]:
        """
        Estima el tiempo de vida promedio de las órdenes en el libro.
        
        Args:
            symbol: Símbolo del par de trading (ej. "BTCUSDT")
            start_time: Inicio del período de análisis
            end_time: Fin del período de análisis
            price_levels: Número de niveles de precio a considerar
            min_volume_threshold: Umbral mínimo de volumen para considerar una orden
            
        Returns:
            Dict con el tiempo de vida estimado de las órdenes
        """
        # Obtener snapshots en el rango de tiempo
        snapshots = self.db_adapter.get_snapshots_in_range(symbol, start_time, end_time)
        
        if len(snapshots) < 2:
            logger.warning(f"No hay suficientes snapshots para {symbol} entre {start_time} y {end_time}")
            return {"average_lifetime": 0.0, "bid_lifetime": 0.0, "ask_lifetime": 0.0}
        
        # Inicializar contadores para el seguimiento de órdenes
        bid_lifetimes = []
        ask_lifetimes = []
        
        # Analizar cada par de snapshots consecutivos
        for i in range(len(snapshots) - 1):
            current = snapshots[i]
            next_snapshot = snapshots[i + 1]
            
            # Calcular tiempo entre snapshots
            time_diff = (next_snapshot['timestamp'] - current['timestamp']).total_seconds()
            
            # Analizar órdenes de compra (bids)
            for j in range(min(price_levels, len(current['bids_price']))):
                current_price = current['bids_price'][j]
                current_qty = current['bids_quantity'][j]
                
                # Buscar el mismo nivel de precio en el siguiente snapshot
                found = False
                for k in range(len(next_snapshot['bids_price'])):
                    if abs(next_snapshot['bids_price'][k] - current_price) < 1e-6:
                        next_qty = next_snapshot['bids_quantity'][k]
                        # Si el volumen disminuyó, algunas órdenes se ejecutaron o cancelaron
                        if current_qty - next_qty > min_volume_threshold:
                            bid_lifetimes.append(time_diff)
                        found = True
                        break
                
                # Si no se encontró el nivel de precio, todas las órdenes se ejecutaron o cancelaron
                if not found and current_qty > min_volume_threshold:
                    bid_lifetimes.append(time_diff)
            
            # Analizar órdenes de venta (asks) - mismo proceso que para bids
            for j in range(min(price_levels, len(current['asks_price']))):
                current_price = current['asks_price'][j]
                current_qty = current['asks_quantity'][j]
                
                found = False
                for k in range(len(next_snapshot['asks_price'])):
                    if abs(next_snapshot['asks_price'][k] - current_price) < 1e-6:
                        next_qty = next_snapshot['asks_quantity'][k]
                        if current_qty - next_qty > min_volume_threshold:
                            ask_lifetimes.append(time_diff)
                        found = True
                        break
                
                if not found and current_qty > min_volume_threshold:
                    ask_lifetimes.append(time_diff)
        
        # Calcular promedios
        bid_lifetime = sum(bid_lifetimes) / len(bid_lifetimes) if bid_lifetimes else 0
        ask_lifetime = sum(ask_lifetimes) / len(ask_lifetimes) if ask_lifetimes else 0
        average_lifetime = (bid_lifetime + ask_lifetime) / 2 if bid_lifetimes or ask_lifetimes else 0
        
        return {
            "average_lifetime": average_lifetime,
            "bid_lifetime": bid_lifetime,
            "ask_lifetime": ask_lifetime,
            "bid_samples": len(bid_lifetimes),
            "ask_samples": len(ask_lifetimes)
        }
