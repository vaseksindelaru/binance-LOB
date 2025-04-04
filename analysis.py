from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import numpy as np
from infi.clickhouse_orm.database import Database
from model import DepthSnapshot, DiffDepthStream
from config import CONFIG


class OrderBookAnalyzer:
    """
    Clase para realizar análisis avanzados sobre los datos del libro de órdenes (LOB)
    capturados por binance-LOB.
    """
    
    def __init__(self, database: Database):
        """
        Inicializa el analizador con una conexión a la base de datos.
        
        Args:
            database: Conexión a la base de datos ClickHouse
        """
        self.db = database
    
    def get_buy_sell_ratio(self, symbol: str, timestamp: Optional[datetime] = None,
                           lookback_seconds: int = 60) -> float:
        """
        Calcula el ratio entre el volumen total de órdenes de compra (bids) y de venta (asks).
        
        Args:
            symbol: Símbolo del par de trading (ej. "BTCUSDT")
            timestamp: Momento específico para el cálculo. Si es None, se usa el último snapshot disponible.
            lookback_seconds: Segundos hacia atrás para buscar el snapshot más cercano si no hay uno exacto.
            
        Returns:
            float: Ratio de volumen de compra/venta (>1 indica más presión compradora, <1 más presión vendedora)
        """
        # Determinar el rango de tiempo para la consulta
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        start_time = timestamp - timedelta(seconds=lookback_seconds)
        
        # Obtener el snapshot más reciente dentro del rango de tiempo
        snapshot = self.db.select(
            "SELECT * FROM DepthSnapshot "
            "WHERE symbol = %(symbol)s AND timestamp >= %(start_time)s AND timestamp <= %(end_time)s "
            "ORDER BY timestamp DESC LIMIT 1",
            params={
                'symbol': symbol,
                'start_time': start_time,
                'end_time': timestamp
            }
        )
        
        if not snapshot:
            raise ValueError(f"No se encontraron datos para el símbolo {symbol} en el rango de tiempo especificado")
        
        snapshot = snapshot[0]
        
        # Calcular el volumen total de bids (compra) y asks (venta)
        total_bid_volume = sum(float(q) for q in snapshot.bids_quantity)
        total_ask_volume = sum(float(q) for q in snapshot.asks_quantity)
        
        # Evitar división por cero
        if total_ask_volume == 0:
            return float('inf')  # Infinito si no hay órdenes de venta
        
        # Calcular el ratio
        buy_sell_ratio = total_bid_volume / total_ask_volume
        
        return buy_sell_ratio
    
    def get_buy_sell_ratio_history(self, symbol: str, 
                                  start_time: datetime, 
                                  end_time: datetime,
                                  interval_minutes: int = 5) -> Dict[datetime, float]:
        """
        Calcula el ratio de compra/venta a lo largo del tiempo para un período específico.
        
        Args:
            symbol: Símbolo del par de trading
            start_time: Tiempo de inicio para el análisis
            end_time: Tiempo de fin para el análisis
            interval_minutes: Intervalo en minutos entre cada punto de datos
            
        Returns:
            Dict[datetime, float]: Diccionario con timestamps como claves y ratios como valores
        """
        # Calcular los puntos de tiempo para el análisis
        current_time = start_time
        results = {}
        
        while current_time <= end_time:
            try:
                ratio = self.get_buy_sell_ratio(symbol, current_time)
                results[current_time] = ratio
            except ValueError:
                # Si no hay datos para este punto de tiempo, lo saltamos
                pass
            
            current_time += timedelta(minutes=interval_minutes)
        
        return results
    
    def get_imbalance_score(self, symbol: str, timestamp: Optional[datetime] = None,
                           lookback_seconds: int = 60, depth_levels: int = 10) -> float:
        """
        Calcula una puntuación de desequilibrio ponderada que da más importancia a las órdenes
        cercanas al precio medio.
        
        Args:
            symbol: Símbolo del par de trading
            timestamp: Momento específico para el cálculo
            lookback_seconds: Segundos hacia atrás para buscar el snapshot
            depth_levels: Número de niveles de profundidad a considerar
            
        Returns:
            float: Puntuación de desequilibrio (-1 a 1, donde valores positivos indican presión compradora)
        """
        # Determinar el rango de tiempo para la consulta
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        start_time = timestamp - timedelta(seconds=lookback_seconds)
        
        # Obtener el snapshot más reciente dentro del rango de tiempo
        snapshot = self.db.select(
            "SELECT * FROM DepthSnapshot "
            "WHERE symbol = %(symbol)s AND timestamp >= %(start_time)s AND timestamp <= %(end_time)s "
            "ORDER BY timestamp DESC LIMIT 1",
            params={
                'symbol': symbol,
                'start_time': start_time,
                'end_time': timestamp
            }
        )
        
        if not snapshot:
            raise ValueError(f"No se encontraron datos para el símbolo {symbol} en el rango de tiempo especificado")
        
        snapshot = snapshot[0]
        
        # Limitar el análisis a los primeros depth_levels niveles
        bids_quantity = snapshot.bids_quantity[:depth_levels] if len(snapshot.bids_quantity) > depth_levels else snapshot.bids_quantity
        asks_quantity = snapshot.asks_quantity[:depth_levels] if len(snapshot.asks_quantity) > depth_levels else snapshot.asks_quantity
        
        # Crear pesos que den más importancia a las órdenes cercanas al precio medio
        weights = np.linspace(1.0, 0.1, len(bids_quantity)) if bids_quantity else []
        
        # Calcular volúmenes ponderados
        weighted_bid_volume = sum(float(q) * w for q, w in zip(bids_quantity, weights))
        weighted_ask_volume = sum(float(q) * w for q, w in zip(asks_quantity, weights))
        
        total_volume = weighted_bid_volume + weighted_ask_volume
        
        # Evitar división por cero
        if total_volume == 0:
            return 0.0
        
        # Calcular puntuación de desequilibrio normalizada entre -1 y 1
        imbalance = (weighted_bid_volume - weighted_ask_volume) / total_volume
        
        return imbalance
    
    def get_order_book_change_rate(self, symbol: str, 
                                  start_time: datetime, 
                                  end_time: datetime,
                                  interval_seconds: int = 60) -> Dict[str, float]:
        """
        Calcula la velocidad de cambio en el libro de órdenes (número de actualizaciones por intervalo).
        
        Args:
            symbol: Símbolo del par de trading
            start_time: Tiempo de inicio para el análisis
            end_time: Tiempo de fin para el análisis
            interval_seconds: Intervalo en segundos para calcular la tasa de cambio
            
        Returns:
            Dict[str, float]: Diccionario con métricas de velocidad de cambio
        """
        # Contar el número de actualizaciones en el período especificado
        updates_count = self.db.select(
            "SELECT count() as count FROM DiffDepthStream "
            "WHERE symbol = %(symbol)s AND timestamp >= %(start_time)s AND timestamp <= %(end_time)s",
            params={
                'symbol': symbol,
                'start_time': start_time,
                'end_time': end_time
            }
        )
        
        if not updates_count or updates_count[0].count == 0:
            raise ValueError(f"No se encontraron actualizaciones para el símbolo {symbol} en el rango de tiempo especificado")
        
        total_updates = updates_count[0].count
        
        # Calcular la duración total en segundos
        duration_seconds = (end_time - start_time).total_seconds()
        
        # Calcular actualizaciones por segundo
        updates_per_second = total_updates / duration_seconds if duration_seconds > 0 else 0
        
        # Calcular actualizaciones por intervalo
        updates_per_interval = updates_per_second * interval_seconds
        
        # Calcular el número de intervalos
        num_intervals = max(1, int(duration_seconds / interval_seconds))
        
        # Obtener estadísticas más detalladas por intervalos
        interval_stats = []
        current_start = start_time
        
        for _ in range(num_intervals):
            current_end = current_start + timedelta(seconds=interval_seconds)
            if current_end > end_time:
                current_end = end_time
                
            # Contar actualizaciones en este intervalo
            interval_count = self.db.select(
                "SELECT count() as count FROM DiffDepthStream "
                "WHERE symbol = %(symbol)s AND timestamp >= %(start_time)s AND timestamp <= %(end_time)s",
                params={
                    'symbol': symbol,
                    'start_time': current_start,
                    'end_time': current_end
                }
            )
            
            if interval_count and interval_count[0].count > 0:
                interval_stats.append(interval_count[0].count)
            
            current_start = current_end
        
        # Calcular estadísticas de los intervalos
        max_updates = max(interval_stats) if interval_stats else 0
        min_updates = min(interval_stats) if interval_stats else 0
        avg_updates = sum(interval_stats) / len(interval_stats) if interval_stats else 0
        
        # Devolver resultados
        return {
            'total_updates': total_updates,
            'duration_seconds': duration_seconds,
            'updates_per_second': updates_per_second,
            'updates_per_interval': updates_per_interval,
            'max_updates_in_interval': max_updates,
            'min_updates_in_interval': min_updates,
            'avg_updates_in_interval': avg_updates,
            'num_intervals_analyzed': len(interval_stats)
        }
    
    def get_order_book_change_rate_history(self, symbol: str, 
                                         start_time: datetime, 
                                         end_time: datetime,
                                         window_minutes: int = 5) -> Dict[datetime, float]:
        """
        Calcula la velocidad de cambio en el libro de órdenes a lo largo del tiempo.
        
        Args:
            symbol: Símbolo del par de trading
            start_time: Tiempo de inicio para el análisis
            end_time: Tiempo de fin para el análisis
            window_minutes: Tamaño de la ventana deslizante en minutos
            
        Returns:
            Dict[datetime, float]: Diccionario con timestamps como claves y tasas de cambio como valores
        """
        # Convertir minutos a segundos
        window_seconds = window_minutes * 60
        
        # Calcular los puntos de tiempo para el análisis
        current_time = start_time
        results = {}
        
        while current_time <= end_time:
            window_end = current_time + timedelta(seconds=window_seconds)
            if window_end > end_time:
                window_end = end_time
            
            try:
                # Contar actualizaciones en esta ventana
                updates_count = self.db.select(
                    "SELECT count() as count FROM DiffDepthStream "
                    "WHERE symbol = %(symbol)s AND timestamp >= %(start_time)s AND timestamp <= %(end_time)s",
                    params={
                        'symbol': symbol,
                        'start_time': current_time,
                        'end_time': window_end
                    }
                )
                
                if updates_count and updates_count[0].count > 0:
                    # Calcular la duración real de la ventana en segundos
                    window_duration = (window_end - current_time).total_seconds()
                    
                    # Calcular actualizaciones por segundo en esta ventana
                    updates_per_second = updates_count[0].count / window_duration if window_duration > 0 else 0
                    
                    results[current_time] = updates_per_second
            except Exception:
                # Si hay algún error, continuamos con el siguiente punto
                pass
            
            # Avanzar al siguiente punto de tiempo (con un solapamiento del 50%)
            current_time += timedelta(seconds=window_seconds // 2)
        
        return results
    
    def analyze_order_book_volatility(self, symbol: str, 
                                     start_time: datetime, 
                                     end_time: datetime,
                                     window_minutes: int = 5) -> Dict[str, float]:
        """
        Analiza la volatilidad del libro de órdenes basándose en la variación de la velocidad de cambio.
        
        Args:
            symbol: Símbolo del par de trading
            start_time: Tiempo de inicio para el análisis
            end_time: Tiempo de fin para el análisis
            window_minutes: Tamaño de la ventana deslizante en minutos
            
        Returns:
            Dict[str, float]: Diccionario con métricas de volatilidad
        """
        # Obtener el historial de tasas de cambio
        change_rate_history = self.get_order_book_change_rate_history(
            symbol, start_time, end_time, window_minutes
        )
        
        if not change_rate_history:
            raise ValueError(f"No se encontraron suficientes datos para analizar la volatilidad del libro de órdenes")
        
        # Extraer las tasas de cambio
        rates = list(change_rate_history.values())
        
        # Calcular estadísticas
        avg_rate = sum(rates) / len(rates)
        max_rate = max(rates)
        min_rate = min(rates)
        
        # Calcular la desviación estándar (medida de volatilidad)
        variance = sum((r - avg_rate) ** 2 for r in rates) / len(rates)
        std_dev = variance ** 0.5
        
        # Calcular el coeficiente de variación (volatilidad normalizada)
        cv = std_dev / avg_rate if avg_rate > 0 else 0
        
        # Calcular la tasa de cambio relativa (max/min)
        relative_change = max_rate / min_rate if min_rate > 0 else float('inf')
        
        # Devolver resultados
        return {
            'avg_updates_per_second': avg_rate,
            'max_updates_per_second': max_rate,
            'min_updates_per_second': min_rate,
            'std_dev': std_dev,
            'coefficient_of_variation': cv,
            'relative_change': relative_change,
            'num_windows_analyzed': len(rates)
        }
        
    def get_average_order_size(self, symbol: str, timestamp: Optional[datetime] = None,
                              lookback_seconds: int = 60, side: str = 'both') -> Dict[str, float]:
        """
        Calcula el tamaño medio de las órdenes en el libro de órdenes.
        
        Args:
            symbol: Símbolo del par de trading
            timestamp: Momento específico para el cálculo. Si es None, se usa el último snapshot disponible.
            lookback_seconds: Segundos hacia atrás para buscar el snapshot más cercano si no hay uno exacto.
            side: Lado del libro a analizar ('bid', 'ask', o 'both')
            
        Returns:
            Dict[str, float]: Diccionario con tamaños medios de órdenes para cada lado
        """
        # Determinar el rango de tiempo para la consulta
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        start_time = timestamp - timedelta(seconds=lookback_seconds)
        
        # Obtener el snapshot más reciente dentro del rango de tiempo
        snapshot = self.db.select(
            "SELECT * FROM DepthSnapshot "
            "WHERE symbol = %(symbol)s AND timestamp >= %(start_time)s AND timestamp <= %(end_time)s "
            "ORDER BY timestamp DESC LIMIT 1",
            params={
                'symbol': symbol,
                'start_time': start_time,
                'end_time': timestamp
            }
        )
        
        if not snapshot:
            raise ValueError(f"No se encontraron datos para el símbolo {symbol} en el rango de tiempo especificado")
        
        snapshot = snapshot[0]
        
        # Calcular tamaño medio de órdenes de compra (bids)
        bid_avg_size = 0
        if side in ['bid', 'both'] and snapshot.bids_quantity:
            bid_avg_size = sum(float(q) for q in snapshot.bids_quantity) / len(snapshot.bids_quantity)
        
        # Calcular tamaño medio de órdenes de venta (asks)
        ask_avg_size = 0
        if side in ['ask', 'both'] and snapshot.asks_quantity:
            ask_avg_size = sum(float(q) for q in snapshot.asks_quantity) / len(snapshot.asks_quantity)
        
        # Calcular tamaño medio combinado si se solicitan ambos lados
        combined_avg_size = 0
        if side == 'both' and (snapshot.bids_quantity or snapshot.asks_quantity):
            total_quantity = sum(float(q) for q in snapshot.bids_quantity) + sum(float(q) for q in snapshot.asks_quantity)
            total_orders = len(snapshot.bids_quantity) + len(snapshot.asks_quantity)
            combined_avg_size = total_quantity / total_orders if total_orders > 0 else 0
        
        # Devolver resultados
        return {
            'bid_avg_size': bid_avg_size,
            'ask_avg_size': ask_avg_size,
            'combined_avg_size': combined_avg_size
        }
    
    def get_market_depth(self, symbol: str, timestamp: Optional[datetime] = None,
                        lookback_seconds: int = 60, depth_levels: int = 10) -> Dict[str, float]:
        """
        Calcula la profundidad del mercado, que es el volumen total de órdenes
        en el libro de órdenes a diferentes niveles de precio.
        
        Args:
            symbol: Símbolo del par de trading
            timestamp: Momento específico para el cálculo. Si es None, se usa el último snapshot disponible.
            lookback_seconds: Segundos hacia atrás para buscar el snapshot más cercano si no hay uno exacto.
            depth_levels: Número de niveles de profundidad a considerar
            
        Returns:
            Dict[str, float]: Diccionario con métricas de profundidad del mercado
        """
        # Determinar el rango de tiempo para la consulta
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        start_time = timestamp - timedelta(seconds=lookback_seconds)
        
        # Obtener el snapshot más reciente dentro del rango de tiempo
        snapshot = self.db.select(
            "SELECT * FROM DepthSnapshot "
            "WHERE symbol = %(symbol)s AND timestamp >= %(start_time)s AND timestamp <= %(end_time)s "
            "ORDER BY timestamp DESC LIMIT 1",
            params={
                'symbol': symbol,
                'start_time': start_time,
                'end_time': timestamp
            }
        )
        
        if not snapshot:
            raise ValueError(f"No se encontraron datos para el símbolo {symbol} en el rango de tiempo especificado")
        
        snapshot = snapshot[0]
        
        # Limitar el análisis a los primeros depth_levels niveles
        bids_price = snapshot.bids_price[:depth_levels] if len(snapshot.bids_price) > depth_levels else snapshot.bids_price
        bids_quantity = snapshot.bids_quantity[:depth_levels] if len(snapshot.bids_quantity) > depth_levels else snapshot.bids_quantity
        asks_price = snapshot.asks_price[:depth_levels] if len(snapshot.asks_price) > depth_levels else snapshot.asks_price
        asks_quantity = snapshot.asks_quantity[:depth_levels] if len(snapshot.asks_quantity) > depth_levels else snapshot.asks_quantity
        
        # Calcular el volumen total en los niveles especificados
        total_bid_volume = sum(float(q) for q in bids_quantity)
        total_ask_volume = sum(float(q) for q in asks_quantity)
        total_volume = total_bid_volume + total_ask_volume
        
        # Calcular el precio medio ponderado por volumen (VWAP) para bids y asks
        bid_vwap = sum(float(p) * float(q) for p, q in zip(bids_price, bids_quantity)) / total_bid_volume if total_bid_volume > 0 else 0
        ask_vwap = sum(float(p) * float(q) for p, q in zip(asks_price, asks_quantity)) / total_ask_volume if total_ask_volume > 0 else 0
        
        # Calcular el spread medio (diferencia entre el mejor precio de compra y venta)
        best_bid = float(bids_price[0]) if bids_price else 0
        best_ask = float(asks_price[0]) if asks_price else 0
        spread = best_ask - best_bid if best_bid > 0 and best_ask > 0 else 0
        spread_percentage = (spread / best_bid) * 100 if best_bid > 0 else 0
        
        # Calcular la profundidad acumulada a diferentes niveles
        bid_depth_levels = {}
        ask_depth_levels = {}
        
        # Profundidad acumulada para bids
        cumulative_bid_volume = 0
        for i, (price, quantity) in enumerate(zip(bids_price, bids_quantity)):
            level = i + 1
            cumulative_bid_volume += float(quantity)
            bid_depth_levels[f"level_{level}"] = {
                "price": float(price),
                "volume": float(quantity),
                "cumulative_volume": cumulative_bid_volume
            }
        
        # Profundidad acumulada para asks
        cumulative_ask_volume = 0
        for i, (price, quantity) in enumerate(zip(asks_price, asks_quantity)):
            level = i + 1
            cumulative_ask_volume += float(quantity)
            ask_depth_levels[f"level_{level}"] = {
                "price": float(price),
                "volume": float(quantity),
                "cumulative_volume": cumulative_ask_volume
            }
        
        # Calcular la asimetría de la profundidad (qué lado tiene más volumen)
        depth_asymmetry = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume) if (total_bid_volume + total_ask_volume) > 0 else 0
        
        # Devolver resultados
        return {
            'total_bid_volume': total_bid_volume,
            'total_ask_volume': total_ask_volume,
            'total_volume': total_volume,
            'bid_vwap': bid_vwap,
            'ask_vwap': ask_vwap,
            'best_bid': best_bid,
            'best_ask': best_ask,
            'spread': spread,
            'spread_percentage': spread_percentage,
            'depth_asymmetry': depth_asymmetry,
            'bid_depth_levels': bid_depth_levels,
            'ask_depth_levels': ask_depth_levels
        }
    
    def get_market_depth_history(self, symbol: str, 
                               start_time: datetime, 
                               end_time: datetime,
                               interval_minutes: int = 5,
                               depth_levels: int = 10) -> Dict[datetime, Dict[str, float]]:
        """
        Calcula la profundidad del mercado a lo largo del tiempo para un período específico.
        
        Args:
            symbol: Símbolo del par de trading
            start_time: Tiempo de inicio para el análisis
            end_time: Tiempo de fin para el análisis
            interval_minutes: Intervalo en minutos entre cada punto de datos
            depth_levels: Número de niveles de profundidad a considerar
            
        Returns:
            Dict[datetime, Dict[str, float]]: Diccionario con timestamps como claves y métricas de profundidad como valores
        """
        # Calcular los puntos de tiempo para el análisis
        current_time = start_time
        results = {}
        
        while current_time <= end_time:
            try:
                depth_metrics = self.get_market_depth(symbol, current_time, lookback_seconds=60, depth_levels=depth_levels)
                
                # Simplificar los resultados para el historial (omitir los niveles detallados)
                simplified_metrics = {
                    'total_bid_volume': depth_metrics['total_bid_volume'],
                    'total_ask_volume': depth_metrics['total_ask_volume'],
                    'total_volume': depth_metrics['total_volume'],
                    'bid_vwap': depth_metrics['bid_vwap'],
                    'ask_vwap': depth_metrics['ask_vwap'],
                    'spread': depth_metrics['spread'],
                    'spread_percentage': depth_metrics['spread_percentage'],
                    'depth_asymmetry': depth_metrics['depth_asymmetry']
                }
                
                results[current_time] = simplified_metrics
            except ValueError:
                # Si no hay datos para este punto de tiempo, lo saltamos
                pass
            
            current_time += timedelta(minutes=interval_minutes)
        
        return results
    
    def estimate_order_lifetime(self, symbol: str, 
                              start_time: datetime, 
                              end_time: datetime,
                              price_levels: int = 10,
                              min_volume_threshold: float = 0.1) -> Dict[str, float]:
        """
        Estima el tiempo de vida promedio de las órdenes en el libro de órdenes.
        
        Este método analiza cómo cambian los volúmenes en niveles de precio específicos
        a lo largo del tiempo para estimar cuánto tiempo permanecen las órdenes en el libro.
        
        Args:
            symbol: Símbolo del par de trading
            start_time: Tiempo de inicio para el análisis
            end_time: Tiempo de fin para el análisis
            price_levels: Número de niveles de precio a considerar
            min_volume_threshold: Umbral mínimo de volumen para considerar un cambio significativo
            
        Returns:
            Dict[str, float]: Diccionario con estadísticas sobre el tiempo de vida de las órdenes
        """
        # Obtener todos los snapshots en el rango de tiempo especificado
        snapshots = self.db.select(
            "SELECT * FROM DepthSnapshot "
            "WHERE symbol = %(symbol)s AND timestamp >= %(start_time)s AND timestamp <= %(end_time)s "
            "ORDER BY timestamp ASC",
            params={
                'symbol': symbol,
                'start_time': start_time,
                'end_time': end_time
            }
        )
        
        if not snapshots or len(snapshots) < 2:
            raise ValueError(f"No se encontraron suficientes snapshots para el símbolo {symbol} en el rango de tiempo especificado")
        
        # Rastrear órdenes a lo largo del tiempo
        bid_order_lifetimes = []
        ask_order_lifetimes = []
        
        # Para cada nivel de precio, rastreamos los cambios de volumen
        for level in range(min(price_levels, len(snapshots[0].bids_price))):
            # Rastrear órdenes de compra (bids)
            bid_volume_changes = self._track_volume_changes(
                snapshots, 'bids', level, min_volume_threshold
            )
            bid_order_lifetimes.extend(bid_volume_changes)
            
            # Rastrear órdenes de venta (asks)
            ask_volume_changes = self._track_volume_changes(
                snapshots, 'asks', level, min_volume_threshold
            )
            ask_order_lifetimes.extend(ask_volume_changes)
        
        # Calcular estadísticas
        all_lifetimes = bid_order_lifetimes + ask_order_lifetimes
        
        if not all_lifetimes:
            return {
                'avg_order_lifetime_seconds': 0,
                'median_order_lifetime_seconds': 0,
                'max_order_lifetime_seconds': 0,
                'min_order_lifetime_seconds': 0,
                'bid_avg_lifetime_seconds': 0,
                'ask_avg_lifetime_seconds': 0,
                'total_orders_tracked': 0,
                'bid_orders_tracked': 0,
                'ask_orders_tracked': 0
            }
        
        # Calcular estadísticas generales
        avg_lifetime = sum(all_lifetimes) / len(all_lifetimes)
        median_lifetime = sorted(all_lifetimes)[len(all_lifetimes) // 2]
        max_lifetime = max(all_lifetimes)
        min_lifetime = min(all_lifetimes)
        
        # Calcular estadísticas por lado
        bid_avg_lifetime = sum(bid_order_lifetimes) / len(bid_order_lifetimes) if bid_order_lifetimes else 0
        ask_avg_lifetime = sum(ask_order_lifetimes) / len(ask_order_lifetimes) if ask_order_lifetimes else 0
        
        return {
            'avg_order_lifetime_seconds': avg_lifetime,
            'median_order_lifetime_seconds': median_lifetime,
            'max_order_lifetime_seconds': max_lifetime,
            'min_order_lifetime_seconds': min_lifetime,
            'bid_avg_lifetime_seconds': bid_avg_lifetime,
            'ask_avg_lifetime_seconds': ask_avg_lifetime,
            'total_orders_tracked': len(all_lifetimes),
            'bid_orders_tracked': len(bid_order_lifetimes),
            'ask_orders_tracked': len(ask_order_lifetimes)
        }
    
    def _track_volume_changes(self, snapshots, side: str, level: int, min_threshold: float) -> List[float]:
        """
        Método auxiliar para rastrear cambios de volumen en un nivel de precio específico.
        
        Args:
            snapshots: Lista de snapshots del libro de órdenes
            side: Lado del libro ('bids' o 'asks')
            level: Nivel de precio a rastrear
            min_threshold: Umbral mínimo de volumen para considerar un cambio significativo
            
        Returns:
            List[float]: Lista de tiempos de vida estimados (en segundos)
        """
        lifetimes = []
        
        # Obtener las listas de precios y cantidades
        price_attr = f"{side}_price"
        quantity_attr = f"{side}_quantity"
        
        # Rastrear órdenes por precio
        current_prices = {}  # {precio: (timestamp_inicial, volumen_inicial)}
        
        for i in range(len(snapshots) - 1):
            current_snapshot = snapshots[i]
            next_snapshot = snapshots[i + 1]
            
            # Asegurarse de que el nivel existe en ambos snapshots
            if (level >= len(getattr(current_snapshot, price_attr)) or 
                level >= len(getattr(next_snapshot, price_attr))):
                continue
            
            current_price = float(getattr(current_snapshot, price_attr)[level])
            current_quantity = float(getattr(current_snapshot, quantity_attr)[level])
            
            next_price = float(getattr(next_snapshot, price_attr)[level])
            next_quantity = float(getattr(next_snapshot, quantity_attr)[level])
            
            # Verificar si el precio cambió o si el volumen disminuyó significativamente
            if current_price in current_prices:
                # El precio ya está siendo rastreado
                initial_time, initial_volume = current_prices[current_price]
                
                # Verificar si el volumen disminuyó significativamente
                if next_price != current_price or next_quantity < (current_quantity - min_threshold):
                    # La orden fue ejecutada o cancelada
                    lifetime = (current_snapshot.timestamp - initial_time).total_seconds()
                    lifetimes.append(lifetime)
                    
                    # Eliminar el precio del rastreo
                    del current_prices[current_price]
            else:
                # Nuevo precio a rastrear
                current_prices[current_price] = (current_snapshot.timestamp, current_quantity)
        
        # Agregar los tiempos de vida de las órdenes que aún están en el libro al final del período
        last_snapshot = snapshots[-1]
        for price, (initial_time, _) in current_prices.items():
            lifetime = (last_snapshot.timestamp - initial_time).total_seconds()
            lifetimes.append(lifetime)
        
        return lifetimes
    
    def get_order_lifetime_by_price_range(self, symbol: str, 
                                        start_time: datetime, 
                                        end_time: datetime,
                                        price_ranges: List[Tuple[float, float]] = None) -> Dict[str, Dict[str, float]]:
        """
        Analiza el tiempo de vida de las órdenes por rangos de precio.
        
        Args:
            symbol: Símbolo del par de trading
            start_time: Tiempo de inicio para el análisis
            end_time: Tiempo de fin para el análisis
            price_ranges: Lista de tuplas (min_price, max_price) que definen los rangos de precio.
                          Si es None, se crean rangos automáticamente.
            
        Returns:
            Dict[str, Dict[str, float]]: Diccionario con estadísticas por rango de precio
        """
        # Obtener todos los snapshots en el rango de tiempo especificado
        snapshots = self.db.select(
            "SELECT * FROM DepthSnapshot "
            "WHERE symbol = %(symbol)s AND timestamp >= %(start_time)s AND timestamp <= %(end_time)s "
            "ORDER BY timestamp ASC",
            params={
                'symbol': symbol,
                'start_time': start_time,
                'end_time': end_time
            }
        )
        
        if not snapshots or len(snapshots) < 2:
            raise ValueError(f"No se encontraron suficientes snapshots para el símbolo {symbol} en el rango de tiempo especificado")
        
        # Si no se proporcionan rangos de precio, crearlos automáticamente
        if not price_ranges:
            # Encontrar el rango de precios en los datos
            all_prices = []
            for snapshot in snapshots:
                all_prices.extend([float(p) for p in snapshot.bids_price])
                all_prices.extend([float(p) for p in snapshot.asks_price])
            
            min_price = min(all_prices)
            max_price = max(all_prices)
            
            # Crear 5 rangos de precio equidistantes
            range_size = (max_price - min_price) / 5
            price_ranges = [
                (min_price + i * range_size, min_price + (i + 1) * range_size)
                for i in range(5)
            ]
        
        # Rastrear órdenes por rango de precio
        results = {}
        
        for i, (min_price, max_price) in enumerate(price_ranges):
            range_name = f"range_{i+1}_{min_price:.2f}_{max_price:.2f}"
            
            # Rastrear órdenes en este rango de precio
            bid_lifetimes = []
            ask_lifetimes = []
            
            for snapshot_idx in range(len(snapshots) - 1):
                current_snapshot = snapshots[snapshot_idx]
                
                # Rastrear órdenes de compra (bids) en este rango
                for j, price in enumerate(current_snapshot.bids_price):
                    price_float = float(price)
                    if min_price <= price_float <= max_price:
                        # Buscar si esta orden desaparece en snapshots posteriores
                        lifetime = self._find_order_lifetime(
                            snapshots, snapshot_idx, 'bids', j, price_float
                        )
                        if lifetime > 0:
                            bid_lifetimes.append(lifetime)
                
                # Rastrear órdenes de venta (asks) en este rango
                for j, price in enumerate(current_snapshot.asks_price):
                    price_float = float(price)
                    if min_price <= price_float <= max_price:
                        # Buscar si esta orden desaparece en snapshots posteriores
                        lifetime = self._find_order_lifetime(
                            snapshots, snapshot_idx, 'asks', j, price_float
                        )
                        if lifetime > 0:
                            ask_lifetimes.append(lifetime)
            
            # Calcular estadísticas para este rango
            all_lifetimes = bid_lifetimes + ask_lifetimes
            
            if all_lifetimes:
                avg_lifetime = sum(all_lifetimes) / len(all_lifetimes)
                median_lifetime = sorted(all_lifetimes)[len(all_lifetimes) // 2]
                bid_avg = sum(bid_lifetimes) / len(bid_lifetimes) if bid_lifetimes else 0
                ask_avg = sum(ask_lifetimes) / len(ask_lifetimes) if ask_lifetimes else 0
                
                results[range_name] = {
                    'min_price': min_price,
                    'max_price': max_price,
                    'avg_lifetime_seconds': avg_lifetime,
                    'median_lifetime_seconds': median_lifetime,
                    'bid_avg_lifetime_seconds': bid_avg,
                    'ask_avg_lifetime_seconds': ask_avg,
                    'total_orders': len(all_lifetimes),
                    'bid_orders': len(bid_lifetimes),
                    'ask_orders': len(ask_lifetimes)
                }
            else:
                results[range_name] = {
                    'min_price': min_price,
                    'max_price': max_price,
                    'avg_lifetime_seconds': 0,
                    'median_lifetime_seconds': 0,
                    'bid_avg_lifetime_seconds': 0,
                    'ask_avg_lifetime_seconds': 0,
                    'total_orders': 0,
                    'bid_orders': 0,
                    'ask_orders': 0
                }
        
        return results
    
    def _find_order_lifetime(self, snapshots, start_idx: int, side: str, level: int, price: float) -> float:
        """
        Método auxiliar para encontrar cuánto tiempo permanece una orden en el libro.
        
        Args:
            snapshots: Lista de snapshots del libro de órdenes
            start_idx: Índice del snapshot inicial
            side: Lado del libro ('bids' o 'asks')
            level: Nivel de precio
            price: Precio a rastrear
            
        Returns:
            float: Tiempo de vida estimado (en segundos), 0 si no se puede determinar
        """
        start_time = snapshots[start_idx].timestamp
        
        # Buscar en los snapshots posteriores
        for i in range(start_idx + 1, len(snapshots)):
            next_snapshot = snapshots[i]
            price_attr = f"{side}_price"
            
            # Verificar si el precio ya no está en el mismo nivel
            if level < len(getattr(next_snapshot, price_attr)):
                if float(getattr(next_snapshot, price_attr)[level]) != price:
                    # El precio cambió, la orden fue ejecutada o cancelada
                    return (next_snapshot.timestamp - start_time).total_seconds()
            else:
                # El nivel ya no existe, la orden fue ejecutada o cancelada
                return (next_snapshot.timestamp - start_time).total_seconds()
        
        # La orden permaneció hasta el final del período
        return (snapshots[-1].timestamp - start_time).total_seconds()
    
# Funciones de utilidad para usar el analizador

def analyze_buy_sell_ratio(symbol: str, lookback_minutes: int = 60):
    """
    Función de utilidad para analizar el ratio de compra/venta para un símbolo específico.
    
    Args:
        symbol: Símbolo del par de trading
        lookback_minutes: Minutos hacia atrás para analizar
    
    Returns:
        Tuple[Dict[datetime, float], float]: Historial de ratios y el ratio actual
    """
    db = Database(CONFIG.db_name, db_url=f"http://{CONFIG.host_name}:8123/")
    analyzer = OrderBookAnalyzer(db)
    
    # Calcular el ratio actual
    current_ratio = analyzer.get_buy_sell_ratio(symbol)
    
    # Calcular el historial de ratios
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(minutes=lookback_minutes)
    ratio_history = analyzer.get_buy_sell_ratio_history(
        symbol, start_time, end_time, interval_minutes=5
    )
    
    return ratio_history, current_ratio


def analyze_order_book_change_rate(symbol: str, lookback_minutes: int = 60, interval_seconds: int = 60):
    """
    Función de utilidad para analizar la velocidad de cambio del libro de órdenes.
    
    Args:
        symbol: Símbolo del par de trading
        lookback_minutes: Minutos hacia atrás para analizar
        interval_seconds: Intervalo en segundos para calcular la tasa de cambio
    
    Returns:
        Tuple[Dict[datetime, float], Dict[str, float]]: Historial de tasas de cambio y estadísticas
    """
    db = Database(CONFIG.db_name, db_url=f"http://{CONFIG.host_name}:8123/")
    analyzer = OrderBookAnalyzer(db)
    
    # Definir el rango de tiempo
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(minutes=lookback_minutes)
    
    # Calcular estadísticas generales
    change_rate_stats = analyzer.get_order_book_change_rate(
        symbol, start_time, end_time, interval_seconds
    )
    
    # Calcular el historial de tasas de cambio
    change_rate_history = analyzer.get_order_book_change_rate_history(
        symbol, start_time, end_time, window_minutes=5
    )
    
    # Analizar la volatilidad
    volatility_stats = analyzer.analyze_order_book_volatility(
        symbol, start_time, end_time, window_minutes=5
    )
    
    return change_rate_history, change_rate_stats, volatility_stats


if __name__ == "__main__":
    # Ejemplo de uso
    import matplotlib.pyplot as plt
    from config import CONFIG
    
    # Usar el primer símbolo configurado como ejemplo
    symbol = CONFIG.symbols[0]
    
    # Analizar el ratio de compra/venta para las últimas 2 horas
    ratio_history, current_ratio = analyze_buy_sell_ratio(symbol, lookback_minutes=120)
    
    # Imprimir el ratio actual
    print(f"Ratio actual de compra/venta para {symbol}: {current_ratio:.4f}")
    
    # Visualizar el historial de ratios
    if ratio_history:
        times = list(ratio_history.keys())
        ratios = list(ratio_history.values())
        
        plt.figure(figsize=(12, 6))
        plt.plot(times, ratios)
        plt.axhline(y=1.0, color='r', linestyle='--')
        plt.title(f"Ratio de Órdenes Compra/Venta para {symbol}")
        plt.xlabel("Tiempo")
        plt.ylabel("Ratio (Compra/Venta)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{symbol}_buy_sell_ratio.png")
        plt.show()
        
    # Analizar la velocidad de cambio del libro de órdenes
    try:
        change_history, change_stats, volatility = analyze_order_book_change_rate(symbol, lookback_minutes=120)
        
        print(f"\nEstadísticas de velocidad de cambio para {symbol}:")
        print(f"- Total de actualizaciones: {change_stats['total_updates']}")
        print(f"- Actualizaciones por segundo: {change_stats['updates_per_second']:.2f}")
        print(f"- Actualizaciones por minuto: {change_stats['updates_per_second'] * 60:.2f}")
        
        print(f"\nVolatilidad del libro de órdenes:")
        print(f"- Coeficiente de variación: {volatility['coefficient_of_variation']:.4f}")
        print(f"- Cambio relativo (max/min): {volatility['relative_change']:.2f}x")
        
        # Visualizar el historial de tasas de cambio
        if change_history:
            times = list(change_history.keys())
            rates = list(change_history.values())
            
            plt.figure(figsize=(12, 6))
            plt.plot(times, rates)
            plt.title(f"Velocidad de Cambio del Libro de Órdenes para {symbol}")
            plt.xlabel("Tiempo")
            plt.ylabel("Actualizaciones por segundo")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{symbol}_change_rate.png")
            plt.show()
    except ValueError as e:
        print(f"Error al analizar la velocidad de cambio: {e}")
