import argparse
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import json
import random
import os
from typing import Dict, List, Tuple, Optional


class MockOrderBookData:
    """
    Clase para generar datos simulados del libro de órdenes para demostrar
    la funcionalidad de análisis sin necesidad de una conexión a ClickHouse.
    """
    
    def __init__(self, symbol: str, start_time: datetime, end_time: datetime, interval_minutes: int = 5):
        """
        Inicializa el generador de datos simulados.
        
        Args:
            symbol: Símbolo del par de trading
            start_time: Tiempo de inicio para la simulación
            end_time: Tiempo de fin para la simulación
            interval_minutes: Intervalo en minutos entre cada punto de datos
        """
        self.symbol = symbol
        self.start_time = start_time
        self.end_time = end_time
        self.interval_minutes = interval_minutes
        self.data = self._generate_data()
        self.updates_data = self._generate_updates_data()
    
    def _generate_data(self):
        """
        Genera datos simulados del libro de órdenes.
        
        Returns:
            Dict: Datos simulados organizados por timestamp
        """
        data = {}
        current_time = self.start_time
        
        # Parámetros base para la simulación
        base_bid_volume = random.uniform(100, 500)
        base_ask_volume = random.uniform(100, 500)
        trend = random.choice([-1, 1])  # Tendencia alcista (1) o bajista (-1)
        
        while current_time <= self.end_time:
            # Simular variaciones en los volúmenes
            time_factor = (current_time - self.start_time).total_seconds() / (self.end_time - self.start_time).total_seconds()
            trend_factor = 1 + (0.5 * trend * time_factor)
            
            # Añadir algo de ruido aleatorio
            noise_bid = random.uniform(0.8, 1.2)
            noise_ask = random.uniform(0.8, 1.2)
            
            # Calcular volúmenes para este punto de tiempo
            bid_volume = base_bid_volume * trend_factor * noise_bid
            ask_volume = base_ask_volume / trend_factor * noise_ask
            
            # Generar niveles de precios y cantidades
            num_levels = 10
            bid_prices = [random.uniform(9500, 10000) for _ in range(num_levels)]
            bid_prices.sort(reverse=True)  # Ordenar de mayor a menor (bids)
            
            ask_prices = [random.uniform(10001, 10500) for _ in range(num_levels)]
            ask_prices.sort()  # Ordenar de menor a mayor (asks)
            
            # Distribuir el volumen total entre los niveles
            bid_quantities = self._distribute_volume(bid_volume, num_levels)
            ask_quantities = self._distribute_volume(ask_volume, num_levels)
            
            # Guardar los datos para este punto de tiempo
            data[current_time] = {
                'bids_price': bid_prices,
                'bids_quantity': bid_quantities,
                'asks_price': ask_prices,
                'asks_quantity': ask_quantities,
                'last_update_id': int((current_time - datetime(1970, 1, 1)).total_seconds() * 1000)
            }
            
            # Avanzar al siguiente punto de tiempo
            current_time += timedelta(minutes=self.interval_minutes)
        
        return data
    
    def _generate_updates_data(self):
        """
        Genera datos simulados de actualizaciones del libro de órdenes.
        Simula el flujo de actualizaciones incrementales que llegarían a través del WebSocket.
        
        Returns:
            Dict: Datos de actualizaciones simulados organizados por timestamp
        """
        updates_data = {}
        
        # Generar actualizaciones con mayor granularidad (cada 1 segundo)
        current_time = self.start_time
        
        # Parámetros para simular patrones en la velocidad de cambio
        base_updates_per_second = random.uniform(2, 8)  # Nivel base de actualizaciones
        volatility_periods = []  # Períodos de alta volatilidad
        
        # Generar algunos períodos de alta volatilidad aleatoriamente
        total_duration = (self.end_time - self.start_time).total_seconds()
        num_volatility_periods = random.randint(2, 5)
        
        for _ in range(num_volatility_periods):
            start_pct = random.uniform(0.1, 0.9)
            duration_pct = random.uniform(0.05, 0.2)
            intensity = random.uniform(3, 10)  # Multiplicador de actualizaciones
            
            start_offset = total_duration * start_pct
            duration = total_duration * duration_pct
            
            volatility_periods.append({
                'start': self.start_time + timedelta(seconds=start_offset),
                'end': self.start_time + timedelta(seconds=start_offset + duration),
                'intensity': intensity
            })
        
        # Generar actualizaciones segundo a segundo
        while current_time <= self.end_time:
            # Verificar si estamos en un período de alta volatilidad
            in_volatility_period = False
            intensity_multiplier = 1.0
            
            for period in volatility_periods:
                if period['start'] <= current_time <= period['end']:
                    in_volatility_period = True
                    intensity_multiplier = period['intensity']
                    break
            
            # Calcular número de actualizaciones para este segundo
            if in_volatility_period:
                # Alta volatilidad: más actualizaciones y más variabilidad
                updates = int(base_updates_per_second * intensity_multiplier * random.uniform(0.8, 1.2))
            else:
                # Volatilidad normal
                updates = int(base_updates_per_second * random.uniform(0.7, 1.3))
            
            updates_data[current_time] = updates
            
            # Avanzar al siguiente segundo
            current_time += timedelta(seconds=1)
        
        return updates_data
    
    def _distribute_volume(self, total_volume: float, num_levels: int) -> List[float]:
        """
        Distribuye el volumen total entre varios niveles de precio,
        con más volumen en los niveles más cercanos al precio medio.
        
        Args:
            total_volume: Volumen total a distribuir
            num_levels: Número de niveles de precio
            
        Returns:
            List[float]: Lista de volúmenes distribuidos
        """
        # Crear una distribución que favorezca los primeros niveles
        weights = np.linspace(1.0, 0.1, num_levels)
        weights = weights / weights.sum()
        
        # Distribuir el volumen según los pesos
        volumes = [total_volume * w for w in weights]
        
        # Añadir algo de ruido
        volumes = [v * random.uniform(0.8, 1.2) for v in volumes]
        
        return volumes
    
    def get_snapshot(self, timestamp: datetime) -> Optional[Dict]:
        """
        Obtiene un snapshot del libro de órdenes para un timestamp específico.
        
        Args:
            timestamp: Momento específico para obtener el snapshot
            
        Returns:
            Optional[Dict]: Datos del snapshot o None si no hay datos para ese timestamp
        """
        # Encontrar el timestamp más cercano
        closest_time = min(self.data.keys(), key=lambda x: abs((x - timestamp).total_seconds()))
        
        # Si la diferencia es mayor que el intervalo, devolver None
        if abs((closest_time - timestamp).total_seconds()) > self.interval_minutes * 60:
            return None
        
        return self.data[closest_time]
    
    def get_updates_count(self, start_time: datetime, end_time: datetime) -> int:
        """
        Obtiene el número de actualizaciones del libro de órdenes en un intervalo de tiempo.
        
        Args:
            start_time: Tiempo de inicio del intervalo
            end_time: Tiempo de fin del intervalo
            
        Returns:
            int: Número de actualizaciones en el intervalo
        """
        count = 0
        
        for timestamp, updates in self.updates_data.items():
            if start_time <= timestamp <= end_time:
                count += updates
        
        return count
    
    def get_market_depth_history(self, symbol: str = None, start_time: datetime = None, end_time: datetime = None,
                               interval_minutes: int = 5, depth_levels: int = 10) -> Dict[datetime, Dict[str, float]]:
        """
        Simula el historial de profundidad del mercado a lo largo del tiempo.
        """
        # Generar datos simulados
        current_time = self.start_time
        results = {}
        
        while current_time <= self.end_time:
            # Generar datos de profundidad para este punto de tiempo
            depth_data = self.get_market_depth(symbol, current_time, depth_levels=depth_levels)
            
            # Simplificar los resultados para el historial (omitir los niveles detallados)
            simplified_metrics = {
                'total_bid_volume': depth_data['total_bid_volume'],
                'total_ask_volume': depth_data['total_ask_volume'],
                'total_volume': depth_data['total_volume'],
                'bid_vwap': depth_data['bid_vwap'],
                'ask_vwap': depth_data['ask_vwap'],
                'spread': depth_data['spread'],
                'spread_percentage': depth_data['spread_percentage'],
                'depth_asymmetry': depth_data['depth_asymmetry']
            }
            
            results[current_time] = simplified_metrics
            current_time += timedelta(minutes=interval_minutes)
        
        return results
    
    def estimate_order_lifetime(self, symbol: str = None, start_time: datetime = None, end_time: datetime = None,
                              price_levels: int = 10, min_volume_threshold: float = 0.1) -> Dict[str, float]:
        """
        Simula la estimación del tiempo de vida promedio de las órdenes en el libro de órdenes.
        """
        # Generar datos simulados
        # En un caso real, esto analizaría cómo cambian los volúmenes en niveles de precio específicos
        
        # Simular tiempos de vida diferentes para órdenes de compra y venta
        # Valores típicos: entre 30 segundos y 30 minutos (1800 segundos)
        bid_lifetimes = np.random.exponential(scale=300, size=100)  # Media de 5 minutos
        ask_lifetimes = np.random.exponential(scale=180, size=100)  # Media de 3 minutos
        
        # Limitar valores extremos
        bid_lifetimes = np.clip(bid_lifetimes, 10, 3600)  # Entre 10 segundos y 1 hora
        ask_lifetimes = np.clip(ask_lifetimes, 10, 3600)
        
        # Combinar todos los tiempos de vida
        all_lifetimes = np.concatenate([bid_lifetimes, ask_lifetimes])
        
        # Calcular estadísticas
        avg_lifetime = np.mean(all_lifetimes)
        median_lifetime = np.median(all_lifetimes)
        max_lifetime = np.max(all_lifetimes)
        min_lifetime = np.min(all_lifetimes)
        
        bid_avg_lifetime = np.mean(bid_lifetimes)
        ask_avg_lifetime = np.mean(ask_lifetimes)
        
        return {
            'avg_order_lifetime_seconds': float(avg_lifetime),
            'median_order_lifetime_seconds': float(median_lifetime),
            'max_order_lifetime_seconds': float(max_lifetime),
            'min_order_lifetime_seconds': float(min_lifetime),
            'bid_avg_lifetime_seconds': float(bid_avg_lifetime),
            'ask_avg_lifetime_seconds': float(ask_avg_lifetime),
            'total_orders_tracked': len(all_lifetimes),
            'bid_orders_tracked': len(bid_lifetimes),
            'ask_orders_tracked': len(ask_lifetimes)
        }
    
    def get_order_lifetime_by_price_range(self, symbol: str = None, start_time: datetime = None, end_time: datetime = None,
                                        price_ranges: List[Tuple[float, float]] = None) -> Dict[str, Dict[str, float]]:
        """
        Simula el análisis del tiempo de vida de las órdenes por rangos de precio.
        """
        # Si no se proporcionan rangos de precio, crear algunos simulados
        if not price_ranges:
            # Simular rangos de precio alrededor del precio medio
            mid_price = 10000.0  # Precio medio simulado
            range_size = 200.0   # Tamaño de cada rango
            
            price_ranges = [
                (mid_price - 2.5 * range_size + i * range_size, mid_price - 2.5 * range_size + (i + 1) * range_size)
                for i in range(5)
            ]
        
        results = {}
        
        # Para cada rango de precio, generar estadísticas simuladas
        for i, (min_price, max_price) in enumerate(price_ranges):
            range_name = f"range_{i+1}_{min_price:.2f}_{max_price:.2f}"
            
            # Simular tiempos de vida diferentes según la distancia al precio medio
            # Las órdenes más cercanas al precio medio tienden a tener tiempos de vida más cortos
            distance_from_mid = abs(((min_price + max_price) / 2) - 10000.0)
            scale_factor = 60 + distance_from_mid / 10  # Más lejos = más tiempo de vida
            
            # Generar tiempos de vida simulados
            bid_count = np.random.randint(50, 150)
            ask_count = np.random.randint(50, 150)
            
            bid_lifetimes = np.random.exponential(scale=scale_factor, size=bid_count)
            ask_lifetimes = np.random.exponential(scale=scale_factor * 0.8, size=ask_count)  # Ventas más rápidas
            
            # Limitar valores extremos
            bid_lifetimes = np.clip(bid_lifetimes, 5, 3600)
            ask_lifetimes = np.clip(ask_lifetimes, 5, 3600)
            
            all_lifetimes = np.concatenate([bid_lifetimes, ask_lifetimes])
            
            if len(all_lifetimes) > 0:
                avg_lifetime = np.mean(all_lifetimes)
                median_lifetime = np.median(all_lifetimes)
                bid_avg = np.mean(bid_lifetimes) if len(bid_lifetimes) > 0 else 0
                ask_avg = np.mean(ask_lifetimes) if len(ask_lifetimes) > 0 else 0
                
                results[range_name] = {
                    'min_price': float(min_price),
                    'max_price': float(max_price),
                    'avg_lifetime_seconds': float(avg_lifetime),
                    'median_lifetime_seconds': float(median_lifetime),
                    'bid_avg_lifetime_seconds': float(bid_avg),
                    'ask_avg_lifetime_seconds': float(ask_avg),
                    'total_orders': int(len(all_lifetimes)),
                    'bid_orders': int(len(bid_lifetimes)),
                    'ask_orders': int(len(ask_lifetimes))
                }
            else:
                results[range_name] = {
                    'min_price': float(min_price),
                    'max_price': float(max_price),
                    'avg_lifetime_seconds': 0.0,
                    'median_lifetime_seconds': 0.0,
                    'bid_avg_lifetime_seconds': 0.0,
                    'ask_avg_lifetime_seconds': 0.0,
                    'total_orders': 0,
                    'bid_orders': 0,
                    'ask_orders': 0
                }
        
        return results


class OrderBookAnalyzer:
    """
    Clase para realizar análisis avanzados sobre los datos del libro de órdenes (LOB).
    Esta versión trabaja con datos simulados para demostración.
    """
    
    def __init__(self, mock_data: MockOrderBookData):
        """
        Inicializa el analizador con datos simulados.
        
        Args:
            mock_data: Objeto con datos simulados del libro de órdenes
        """
        self.data = mock_data
    
    def get_buy_sell_ratio(self, timestamp: Optional[datetime] = None) -> float:
        """
        Calcula el ratio entre el volumen total de órdenes de compra (bids) y de venta (asks).
        
        Args:
            timestamp: Momento específico para el cálculo. Si es None, se usa el último snapshot disponible.
            
        Returns:
            float: Ratio de volumen de compra/venta (>1 indica más presión compradora, <1 más presión vendedora)
        """
        if timestamp is None:
            timestamp = max(self.data.data.keys())
        
        snapshot = self.data.get_snapshot(timestamp)
        if not snapshot:
            raise ValueError(f"No se encontraron datos para el timestamp {timestamp}")
        
        # Calcular el volumen total de bids (compra) y asks (venta)
        total_bid_volume = sum(float(q) for q in snapshot['bids_quantity'])
        total_ask_volume = sum(float(q) for q in snapshot['asks_quantity'])
        
        # Evitar división por cero
        if total_ask_volume == 0:
            return float('inf')  # Infinito si no hay órdenes de venta
        
        # Calcular el ratio
        buy_sell_ratio = total_bid_volume / total_ask_volume
        
        return buy_sell_ratio
    
    def get_buy_sell_ratio_history(self) -> Dict[datetime, float]:
        """
        Calcula el ratio de compra/venta para todos los puntos de tiempo disponibles.
        
        Returns:
            Dict[datetime, float]: Diccionario con timestamps como claves y ratios como valores
        """
        results = {}
        
        for timestamp in self.data.data.keys():
            try:
                ratio = self.get_buy_sell_ratio(timestamp)
                results[timestamp] = ratio
            except ValueError:
                # Si hay algún error, continuamos con el siguiente punto
                pass
        
        return results
    
    def get_imbalance_score(self, timestamp: Optional[datetime] = None, depth_levels: int = 10) -> float:
        """
        Calcula una puntuación de desequilibrio ponderada que da más importancia a las órdenes
        cercanas al precio medio.
        
        Args:
            timestamp: Momento específico para el cálculo
            depth_levels: Número de niveles de profundidad a considerar
            
        Returns:
            float: Puntuación de desequilibrio (-1 a 1, donde valores positivos indican presión compradora)
        """
        if timestamp is None:
            timestamp = max(self.data.data.keys())
        
        snapshot = self.data.get_snapshot(timestamp)
        if not snapshot:
            raise ValueError(f"No se encontraron datos para el timestamp {timestamp}")
        
        # Limitar el análisis a los primeros depth_levels niveles
        bids_quantity = snapshot['bids_quantity'][:depth_levels] if len(snapshot['bids_quantity']) > depth_levels else snapshot['bids_quantity']
        asks_quantity = snapshot['asks_quantity'][:depth_levels] if len(snapshot['asks_quantity']) > depth_levels else snapshot['asks_quantity']
        
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
    
    def get_order_book_change_rate(self, start_time: datetime, end_time: datetime, interval_seconds: int = 60) -> Dict[str, float]:
        """
        Calcula la velocidad de cambio en el libro de órdenes (número de actualizaciones por intervalo).
        
        Args:
            start_time: Tiempo de inicio para el análisis
            end_time: Tiempo de fin para el análisis
            interval_seconds: Intervalo en segundos para calcular la tasa de cambio
            
        Returns:
            Dict[str, float]: Diccionario con métricas de velocidad de cambio
        """
        # Contar el número de actualizaciones en el período especificado
        total_updates = self.data.get_updates_count(start_time, end_time)
        
        if total_updates == 0:
            raise ValueError(f"No se encontraron actualizaciones en el rango de tiempo especificado")
        
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
            interval_count = self.data.get_updates_count(current_start, current_end)
            
            if interval_count > 0:
                interval_stats.append(interval_count)
            
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
    
    def get_order_book_change_rate_history(self, start_time: datetime, end_time: datetime, window_minutes: int = 5) -> Dict[datetime, float]:
        """
        Calcula la velocidad de cambio en el libro de órdenes a lo largo del tiempo.
        
        Args:
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
                updates_count = self.data.get_updates_count(current_time, window_end)
                
                if updates_count > 0:
                    # Calcular la duración real de la ventana en segundos
                    window_duration = (window_end - current_time).total_seconds()
                    
                    # Calcular actualizaciones por segundo en esta ventana
                    updates_per_second = updates_count / window_duration if window_duration > 0 else 0
                    
                    results[current_time] = updates_per_second
            except Exception:
                # Si hay algún error, continuamos con el siguiente punto
                pass
            
            # Avanzar al siguiente punto de tiempo (con un solapamiento del 50%)
            current_time += timedelta(seconds=window_seconds // 2)
        
        return results
    
    def analyze_order_book_volatility(self, start_time: datetime, end_time: datetime, window_minutes: int = 5) -> Dict[str, float]:
        """
        Analiza la volatilidad del libro de órdenes basándose en la variación de la velocidad de cambio.
        
        Args:
            start_time: Tiempo de inicio para el análisis
            end_time: Tiempo de fin para el análisis
            window_minutes: Tamaño de la ventana deslizante en minutos
            
        Returns:
            Dict[str, float]: Diccionario con métricas de volatilidad
        """
        # Obtener el historial de tasas de cambio
        change_rate_history = self.get_order_book_change_rate_history(start_time, end_time, window_minutes)
        
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
    
    def get_average_order_size(self, timestamp: Optional[datetime] = None, side: str = 'both') -> Dict[str, float]:
        """
        Calcula el tamaño medio de las órdenes en el libro de órdenes.
        
        Args:
            timestamp: Momento específico para el cálculo. Si es None, se usa el último snapshot disponible.
            side: Lado del libro a analizar ('bid', 'ask', o 'both')
            
        Returns:
            Dict[str, float]: Diccionario con tamaños medios de órdenes para cada lado
        """
        if timestamp is None:
            timestamp = max(self.data.data.keys())
        
        snapshot = self.data.get_snapshot(timestamp)
        if not snapshot:
            raise ValueError(f"No se encontraron datos para el timestamp {timestamp}")
        
        # Calcular tamaño medio de órdenes de compra (bids)
        bid_avg_size = 0
        if side in ['bid', 'both'] and snapshot['bids_quantity']:
            bid_avg_size = sum(float(q) for q in snapshot['bids_quantity']) / len(snapshot['bids_quantity'])
        
        # Calcular tamaño medio de órdenes de venta (asks)
        ask_avg_size = 0
        if side in ['ask', 'both'] and snapshot['asks_quantity']:
            ask_avg_size = sum(float(q) for q in snapshot['asks_quantity']) / len(snapshot['asks_quantity'])
        
        # Calcular tamaño medio combinado si se solicitan ambos lados
        combined_avg_size = 0
        if side == 'both' and (snapshot['bids_quantity'] or snapshot['asks_quantity']):
            total_quantity = sum(float(q) for q in snapshot['bids_quantity']) + sum(float(q) for q in snapshot['asks_quantity'])
            total_orders = len(snapshot['bids_quantity']) + len(snapshot['asks_quantity'])
            combined_avg_size = total_quantity / total_orders if total_orders > 0 else 0
        
        # Devolver resultados
        return {
            'bid_avg_size': bid_avg_size,
            'ask_avg_size': ask_avg_size,
            'combined_avg_size': combined_avg_size
        }
    
    def get_average_order_size_history(self, side: str = 'both') -> Dict[datetime, Dict[str, float]]:
        """
        Calcula el tamaño medio de las órdenes a lo largo del tiempo.
        
        Args:
            side: Lado del libro a analizar ('bid', 'ask', o 'both')
            
        Returns:
            Dict[datetime, Dict[str, float]]: Diccionario con timestamps como claves y tamaños medios como valores
        """
        results = {}
        
        for timestamp in self.data.data.keys():
            try:
                avg_sizes = self.get_average_order_size(timestamp, side)
                results[timestamp] = avg_sizes
            except ValueError:
                # Si hay algún error, continuamos con el siguiente punto
                pass
        
        return results
    
    def get_market_depth(self, timestamp: Optional[datetime] = None, depth_levels: int = 10) -> Dict[str, float]:
        """
        Calcula la profundidad del mercado, que es el volumen total de órdenes
        en el libro de órdenes a diferentes niveles de precio.
        
        Args:
            timestamp: Momento específico para el cálculo. Si es None, se usa el último snapshot disponible.
            depth_levels: Número de niveles de profundidad a considerar
            
        Returns:
            Dict[str, float]: Diccionario con métricas de profundidad del mercado
        """
        if timestamp is None:
            timestamp = max(self.data.data.keys())
        
        snapshot = self.data.get_snapshot(timestamp)
        if not snapshot:
            raise ValueError(f"No se encontraron datos para el timestamp {timestamp}")
        
        # Limitar el análisis a los primeros depth_levels niveles
        bids_price = snapshot['bids_price'][:depth_levels] if len(snapshot['bids_price']) > depth_levels else snapshot['bids_price']
        bids_quantity = snapshot['bids_quantity'][:depth_levels] if len(snapshot['bids_quantity']) > depth_levels else snapshot['bids_quantity']
        asks_price = snapshot['asks_price'][:depth_levels] if len(snapshot['asks_price']) > depth_levels else snapshot['asks_price']
        asks_quantity = snapshot['asks_quantity'][:depth_levels] if len(snapshot['asks_quantity']) > depth_levels else snapshot['asks_quantity']
        
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
    
    def get_market_depth_history(self, depth_levels: int = 10) -> Dict[datetime, Dict[str, float]]:
        """
        Calcula la profundidad del mercado a lo largo del tiempo.
        
        Args:
            depth_levels: Número de niveles de profundidad a considerar
            
        Returns:
            Dict[datetime, Dict[str, float]]: Diccionario con timestamps como claves y métricas de profundidad como valores
        """
        results = {}
        
        for timestamp in self.data.data.keys():
            try:
                depth_metrics = self.get_market_depth(timestamp, depth_levels)
                
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
                
                results[timestamp] = simplified_metrics
            except ValueError:
                # Si hay algún error, continuamos con el siguiente punto
                pass
        
        return results
    
    def get_order_lifetime(self, symbol: str = None, start_time: datetime = None, end_time: datetime = None,
                          price_levels: int = 10, min_volume_threshold: float = 0.1) -> Dict[str, float]:
        """
        Simula la estimación del tiempo de vida promedio de las órdenes en el libro de órdenes.
        """
        return self.data.estimate_order_lifetime(symbol, start_time, end_time, price_levels, min_volume_threshold)
    
    def get_order_lifetime_by_price_range(self, symbol: str = None, start_time: datetime = None, end_time: datetime = None,
                                        price_ranges: List[Tuple[float, float]] = None) -> Dict[str, Dict[str, float]]:
        """
        Simula el análisis del tiempo de vida de las órdenes por rangos de precio.
        """
        return self.data.get_order_lifetime_by_price_range(symbol, start_time, end_time, price_ranges)


def save_results_to_file(symbol: str, ratio_history: Dict[datetime, float], imbalance: float, 
                        change_rate_stats: Dict[str, float], volatility: Dict[str, float],
                        avg_order_sizes: Dict[str, float] = None, market_depth: Dict[str, float] = None,
                        order_lifetime: Dict[str, float] = None, lifetime_by_price: Dict[str, Dict[str, float]] = None):
    """
    Guarda los resultados del análisis en un archivo JSON.
    
    Args:
        symbol: Símbolo del par de trading
        ratio_history: Historial de ratios de compra/venta
        imbalance: Puntuación de desequilibrio actual
        change_rate_stats: Estadísticas de velocidad de cambio
        volatility: Estadísticas de volatilidad
        avg_order_sizes: Tamaños medios de órdenes (opcional)
        market_depth: Métricas de profundidad del mercado (opcional)
        order_lifetime: Métricas de tiempo de vida de las órdenes (opcional)
        lifetime_by_price: Métricas de tiempo de vida por rango de precio (opcional)
    """
    # Convertir los timestamps a strings para poder serializarlos
    ratio_history_str = {ts.isoformat(): val for ts, val in ratio_history.items()}
    
    results = {
        'symbol': symbol,
        'timestamp': datetime.utcnow().isoformat(),
        'buy_sell_ratio_history': ratio_history_str,
        'current_imbalance_score': imbalance,
        'order_book_change_rate': change_rate_stats,
        'order_book_volatility': volatility
    }
    
    if avg_order_sizes:
        results['average_order_sizes'] = avg_order_sizes
    
    if market_depth:
        # Eliminar los niveles detallados para el archivo JSON (son muy extensos)
        simplified_depth = {k: v for k, v in market_depth.items() if k not in ['bid_depth_levels', 'ask_depth_levels']}
        results['market_depth'] = simplified_depth
    
    if order_lifetime:
        results['order_lifetime'] = order_lifetime
    
    if lifetime_by_price:
        results['order_lifetime_by_price'] = lifetime_by_price
    
    # Crear directorio de resultados si no existe
    os.makedirs('results', exist_ok=True)
    
    # Guardar en un archivo con timestamp
    filename = f"results/analysis_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Resultados guardados en {filename}")


def main():
    parser = argparse.ArgumentParser(description='Análisis del libro de órdenes de Binance')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Símbolo del par de trading')
    parser.add_argument('--days', type=int, default=1, help='Número de días para el análisis histórico')
    args = parser.parse_args()
    
    # Configurar el período de tiempo para el análisis
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=args.days)
    
    print(f"Analizando datos para {args.symbol} desde {start_time} hasta {end_time}")
    
    # Crear datos simulados para la demostración
    mock_data = MockOrderBookData(
        symbol=args.symbol,
        start_time=start_time,
        end_time=end_time,
        interval_minutes=5
    )
    
    # Crear el analizador
    analyzer = OrderBookAnalyzer(mock_data)
    
    # 1. Análisis del Ratio de Órdenes Compra/Venta
    print("\n=== Análisis del Ratio de Órdenes Compra/Venta ===")
    
    # Calcular el ratio actual
    current_ratio = analyzer.get_buy_sell_ratio()
    print(f"Ratio actual de compra/venta: {current_ratio:.4f}")
    
    if current_ratio > 1:
        print(f"Interpretación: Hay {current_ratio:.2f}x más volumen en órdenes de compra que de venta")
        print("Esto indica una presión compradora en el mercado.")
    else:
        print(f"Interpretación: Hay {1/current_ratio:.2f}x más volumen en órdenes de venta que de compra")
        print("Esto indica una presión vendedora en el mercado.")
    
    # Calcular el historial de ratios
    ratio_history = analyzer.get_buy_sell_ratio_history()
    
    # Calcular la puntuación de desequilibrio
    imbalance = analyzer.get_imbalance_score()
    print(f"\nPuntuación de desequilibrio: {imbalance:.4f}")
    print("(Un valor positivo indica presión compradora, negativo indica presión vendedora)")
    
    # Visualizar el historial de ratios
    timestamps = list(ratio_history.keys())
    ratios = list(ratio_history.values())
    
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, ratios, 'b-', label='Ratio Compra/Venta')
    plt.axhline(y=1.0, color='r', linestyle='--', label='Equilibrio')
    plt.title(f'Evolución del Ratio de Órdenes Compra/Venta para {args.symbol}')
    plt.xlabel('Tiempo')
    plt.ylabel('Ratio (Volumen Compra / Volumen Venta)')
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Guardar la gráfica
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/buy_sell_ratio_{args.symbol}.png')
    print(f"Gráfica guardada en results/buy_sell_ratio_{args.symbol}.png")
    
    # 2. Análisis de la Velocidad de Cambio en el Order Book
    print("\n=== Análisis de la Velocidad de Cambio en el Order Book ===")
    
    # Calcular estadísticas de velocidad de cambio
    change_rate_stats = analyzer.get_order_book_change_rate(start_time, end_time)
    
    print(f"Total de actualizaciones: {change_rate_stats['total_updates']}")
    print(f"Duración del análisis: {change_rate_stats['duration_seconds']:.1f} segundos")
    print(f"Actualizaciones por segundo: {change_rate_stats['updates_per_second']:.2f}")
    print(f"Actualizaciones por minuto: {change_rate_stats['updates_per_second'] * 60:.2f}")
    
    print("\nEstadísticas por intervalos de 60 segundos:")
    print(f"Máximo de actualizaciones en un intervalo: {change_rate_stats['max_updates_in_interval']}")
    print(f"Mínimo de actualizaciones en un intervalo: {change_rate_stats['min_updates_in_interval']}")
    print(f"Promedio de actualizaciones por intervalo: {change_rate_stats['avg_updates_in_interval']:.2f}")
    
    # Obtener el historial de tasas de cambio
    window_minutes = 5
    change_rate_history = analyzer.get_order_book_change_rate_history(start_time, end_time, window_minutes)
    
    # Visualizar el historial de tasas de cambio
    cr_timestamps = list(change_rate_history.keys())
    cr_rates = list(change_rate_history.values())
    
    plt.figure(figsize=(12, 6))
    plt.plot(cr_timestamps, cr_rates, 'g-', label='Tasa de Cambio')
    plt.title(f'Evolución de la Velocidad de Cambio en el Order Book para {args.symbol}')
    plt.xlabel('Tiempo')
    plt.ylabel('Actualizaciones por segundo')
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Guardar la gráfica
    plt.savefig(f'results/change_rate_{args.symbol}.png')
    print(f"Gráfica guardada en results/change_rate_{args.symbol}.png")
    
    # Analizar la volatilidad
    volatility = analyzer.analyze_order_book_volatility(start_time, end_time, window_minutes)
    
    print("\nAnálisis de volatilidad del libro de órdenes:")
    print(f"Tasa promedio de actualizaciones: {volatility['avg_updates_per_second']:.2f} por segundo")
    print(f"Desviación estándar: {volatility['std_dev']:.2f}")
    print(f"Coeficiente de variación: {volatility['coefficient_of_variation']:.2f}")
    print(f"Cambio relativo (max/min): {volatility['relative_change']:.2f}x")
    
    # 3. Análisis del Tamaño Medio de Órdenes
    print("\n=== Análisis del Tamaño Medio de Órdenes ===")
    
    # Calcular tamaños medios actuales
    avg_sizes = analyzer.get_average_order_size()
    
    print(f"Tamaño medio de órdenes de compra (bids): {avg_sizes['bid_avg_size']:.2f}")
    print(f"Tamaño medio de órdenes de venta (asks): {avg_sizes['ask_avg_size']:.2f}")
    print(f"Tamaño medio combinado: {avg_sizes['combined_avg_size']:.2f}")
    
    # Calcular el historial de tamaños medios
    size_history = analyzer.get_average_order_size_history()
    
    # Extraer datos para la visualización
    size_timestamps = list(size_history.keys())
    bid_sizes = [data['bid_avg_size'] for data in size_history.values()]
    ask_sizes = [data['ask_avg_size'] for data in size_history.values()]
    
    # Visualizar el historial de tamaños medios
    plt.figure(figsize=(12, 6))
    plt.plot(size_timestamps, bid_sizes, 'g-', label='Tamaño Medio de Compras (Bids)')
    plt.plot(size_timestamps, ask_sizes, 'r-', label='Tamaño Medio de Ventas (Asks)')
    plt.title(f'Evolución del Tamaño Medio de Órdenes para {args.symbol}')
    plt.xlabel('Tiempo')
    plt.ylabel('Tamaño Medio')
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Guardar la gráfica
    plt.savefig(f'results/avg_order_size_{args.symbol}.png')
    print(f"Gráfica guardada en results/avg_order_size_{args.symbol}.png")
    
    # Calcular la relación entre tamaños de compra y venta
    size_ratio = avg_sizes['bid_avg_size'] / avg_sizes['ask_avg_size'] if avg_sizes['ask_avg_size'] > 0 else float('inf')
    print(f"\nRelación entre tamaño medio de compras y ventas: {size_ratio:.2f}")
    
    if size_ratio > 1:
        print(f"Las órdenes de compra son en promedio {size_ratio:.2f}x más grandes que las de venta")
        print("Esto puede indicar que los compradores están más convencidos o son más agresivos.")
    elif size_ratio < 1:
        print(f"Las órdenes de venta son en promedio {1/size_ratio:.2f}x más grandes que las de compra")
        print("Esto puede indicar que los vendedores están más convencidos o son más agresivos.")
    else:
        print("Los tamaños de órdenes de compra y venta son similares.")
    
    # 4. Análisis de la Profundidad del Mercado
    print("\n=== Análisis de la Profundidad del Mercado ===")
    
    # Calcular métricas de profundidad del mercado
    depth_levels = 10
    market_depth = analyzer.get_market_depth(depth_levels=depth_levels)
    
    print(f"Volumen total de órdenes de compra (bids): {market_depth['total_bid_volume']:.2f}")
    print(f"Volumen total de órdenes de venta (asks): {market_depth['total_ask_volume']:.2f}")
    print(f"Volumen total combinado: {market_depth['total_volume']:.2f}")
    
    print(f"\nMejor precio de compra (bid): {market_depth['best_bid']:.2f}")
    print(f"Mejor precio de venta (ask): {market_depth['best_ask']:.2f}")
    print(f"Spread: {market_depth['spread']:.2f} ({market_depth['spread_percentage']:.2f}%)")
    
    print(f"\nPrecio medio ponderado por volumen (VWAP) de compra: {market_depth['bid_vwap']:.2f}")
    print(f"Precio medio ponderado por volumen (VWAP) de venta: {market_depth['ask_vwap']:.2f}")
    
    print(f"\nAsimetría de profundidad: {market_depth['depth_asymmetry']:.4f}")
    print("(Un valor positivo indica mayor volumen en el lado de compra, negativo en el lado de venta)")
    
    # Calcular el historial de profundidad del mercado
    depth_history = analyzer.get_market_depth_history(depth_levels=depth_levels)
    
    # Extraer datos para la visualización
    depth_timestamps = list(depth_history.keys())
    bid_volumes = [data['total_bid_volume'] for data in depth_history.values()]
    ask_volumes = [data['total_ask_volume'] for data in depth_history.values()]
    asymmetry = [data['depth_asymmetry'] for data in depth_history.values()]
    
    # Visualizar el historial de volúmenes
    plt.figure(figsize=(12, 6))
    plt.plot(depth_timestamps, bid_volumes, 'g-', label='Volumen Total de Compras (Bids)')
    plt.plot(depth_timestamps, ask_volumes, 'r-', label='Volumen Total de Ventas (Asks)')
    plt.title(f'Evolución de la Profundidad del Mercado para {args.symbol}')
    plt.xlabel('Tiempo')
    plt.ylabel('Volumen Total')
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Guardar la gráfica
    plt.savefig(f'results/market_depth_volume_{args.symbol}.png')
    print(f"Gráfica guardada en results/market_depth_volume_{args.symbol}.png")
    
    # Visualizar el historial de asimetría de profundidad
    plt.figure(figsize=(12, 6))
    plt.plot(depth_timestamps, asymmetry, 'b-', label='Asimetría de Profundidad')
    plt.axhline(y=0.0, color='r', linestyle='--', label='Equilibrio')
    plt.title(f'Evolución de la Asimetría de Profundidad para {args.symbol}')
    plt.xlabel('Tiempo')
    plt.ylabel('Asimetría (-1 a 1)')
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Guardar la gráfica
    plt.savefig(f'results/market_depth_asymmetry_{args.symbol}.png')
    print(f"Gráfica guardada en results/market_depth_asymmetry_{args.symbol}.png")
    
    # Visualizar la profundidad acumulada (para el último snapshot)
    bid_levels = market_depth['bid_depth_levels']
    ask_levels = market_depth['ask_depth_levels']
    
    bid_prices = [level_data['price'] for level_data in bid_levels.values()]
    bid_cum_volumes = [level_data['cumulative_volume'] for level_data in bid_levels.values()]
    
    ask_prices = [level_data['price'] for level_data in ask_levels.values()]
    ask_cum_volumes = [level_data['cumulative_volume'] for level_data in ask_levels.values()]
    
    plt.figure(figsize=(12, 6))
    plt.step(bid_prices, bid_cum_volumes, 'g-', where='post', label='Profundidad Acumulada de Compras')
    plt.step(ask_prices, ask_cum_volumes, 'r-', where='post', label='Profundidad Acumulada de Ventas')
    plt.title(f'Profundidad Acumulada del Libro de Órdenes para {args.symbol}')
    plt.xlabel('Precio')
    plt.ylabel('Volumen Acumulado')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Guardar la gráfica
    plt.savefig(f'results/market_depth_cumulative_{args.symbol}.png')
    print(f"Gráfica guardada en results/market_depth_cumulative_{args.symbol}.png")
    
    # 5. Análisis del Tiempo de Vida de las Órdenes
    print("\n=== Análisis del Tiempo de Vida de las Órdenes ===")
    
    # Calcular estadísticas de tiempo de vida
    order_lifetime = analyzer.get_order_lifetime()
    
    print(f"Tiempo de vida promedio de las órdenes: {order_lifetime['avg_order_lifetime_seconds']:.2f} segundos")
    print(f"Tiempo de vida mediano: {order_lifetime['median_order_lifetime_seconds']:.2f} segundos")
    print(f"Tiempo de vida máximo: {order_lifetime['max_order_lifetime_seconds']:.2f} segundos")
    print(f"Tiempo de vida mínimo: {order_lifetime['min_order_lifetime_seconds']:.2f} segundos")
    
    print(f"\nTiempo de vida promedio de órdenes de compra: {order_lifetime['bid_avg_lifetime_seconds']:.2f} segundos")
    print(f"Tiempo de vida promedio de órdenes de venta: {order_lifetime['ask_avg_lifetime_seconds']:.2f} segundos")
    
    print(f"\nTotal de órdenes analizadas: {order_lifetime['total_orders_tracked']}")
    print(f"Órdenes de compra analizadas: {order_lifetime['bid_orders_tracked']}")
    print(f"Órdenes de venta analizadas: {order_lifetime['ask_orders_tracked']}")
    
    # Convertir segundos a minutos para mejor visualización
    bid_lifetime_min = order_lifetime['bid_avg_lifetime_seconds'] / 60
    ask_lifetime_min = order_lifetime['ask_avg_lifetime_seconds'] / 60
    
    # Visualizar la comparación de tiempos de vida entre compras y ventas
    plt.figure(figsize=(10, 6))
    bars = plt.bar(['Órdenes de Compra', 'Órdenes de Venta'], 
                  [bid_lifetime_min, ask_lifetime_min],
                  color=['green', 'red'])
    
    # Añadir etiquetas con los valores
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f} min', ha='center', va='bottom')
    
    plt.title(f'Tiempo de Vida Promedio de Órdenes para {args.symbol}')
    plt.ylabel('Tiempo de Vida (minutos)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Guardar la gráfica
    plt.savefig(f'results/order_lifetime_{args.symbol}.png')
    print(f"Gráfica guardada en results/order_lifetime_{args.symbol}.png")
    
    # Analizar el tiempo de vida por rango de precio
    lifetime_by_price = analyzer.get_order_lifetime_by_price_range()
    
    print("\nTiempo de vida por rango de precio:")
    
    # Extraer datos para visualización
    price_ranges = []
    avg_lifetimes = []
    bid_lifetimes = []
    ask_lifetimes = []
    
    for range_name, stats in lifetime_by_price.items():
        price_range_str = f"{stats['min_price']:.0f}-{stats['max_price']:.0f}"
        price_ranges.append(price_range_str)
        
        # Convertir a minutos para mejor visualización
        avg_lifetimes.append(stats['avg_lifetime_seconds'] / 60)
        bid_lifetimes.append(stats['bid_avg_lifetime_seconds'] / 60)
        ask_lifetimes.append(stats['ask_avg_lifetime_seconds'] / 60)
        
        print(f"Rango {price_range_str}: {stats['avg_lifetime_seconds']:.2f} segundos promedio, "
              f"{stats['total_orders']} órdenes analizadas")
    
    # Visualizar el tiempo de vida por rango de precio
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(price_ranges))
    width = 0.25
    
    plt.bar(x - width, bid_lifetimes, width, label='Órdenes de Compra', color='green', alpha=0.7)
    plt.bar(x, avg_lifetimes, width, label='Promedio', color='blue', alpha=0.7)
    plt.bar(x + width, ask_lifetimes, width, label='Órdenes de Venta', color='red', alpha=0.7)
    
    plt.xlabel('Rango de Precio')
    plt.ylabel('Tiempo de Vida (minutos)')
    plt.title(f'Tiempo de Vida de Órdenes por Rango de Precio para {args.symbol}')
    plt.xticks(x, price_ranges, rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Guardar la gráfica
    plt.savefig(f'results/order_lifetime_by_price_{args.symbol}.png')
    print(f"Gráfica guardada en results/order_lifetime_by_price_{args.symbol}.png")
    
    # Guardar todos los resultados en un archivo
    save_results_to_file(
        symbol=args.symbol,
        ratio_history=ratio_history,
        imbalance=imbalance,
        change_rate_stats=change_rate_stats,
        volatility=volatility,
        avg_order_sizes=avg_sizes,
        market_depth=market_depth,
        order_lifetime=order_lifetime,
        lifetime_by_price=lifetime_by_price
    )
    
    print("\nAnálisis completado. Revise los archivos de resultados para más detalles.")
    plt.show()


if __name__ == "__main__":
    main()
