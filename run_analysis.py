import argparse
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from infi.clickhouse_orm.database import Database
from config import CONFIG
from analysis import OrderBookAnalyzer


def main():
    parser = argparse.ArgumentParser(description="Análisis del Libro de Órdenes de Binance")
    parser.add_argument("--symbol", type=str, help="Símbolo a analizar (ej. BTCUSDT)")
    parser.add_argument("--lookback", type=int, default=120, 
                        help="Minutos hacia atrás para analizar (default: 120)")
    parser.add_argument("--interval", type=int, default=5,
                        help="Intervalo en minutos entre cada punto de datos (default: 5)")
    parser.add_argument("--depth", type=int, default=10,
                        help="Niveles de profundidad para el análisis de desequilibrio (default: 10)")
    
    args = parser.parse_args()
    
    # Si no se especifica un símbolo, usar el primero de la configuración
    symbol = args.symbol if args.symbol else CONFIG.symbols[0]
    
    # Conectar a la base de datos
    db = Database(CONFIG.db_name, db_url=f"http://{CONFIG.host_name}:8123/")
    analyzer = OrderBookAnalyzer(db)
    
    # Calcular el ratio actual
    try:
        current_ratio = analyzer.get_buy_sell_ratio(symbol)
        print(f"\nRatio actual de compra/venta para {symbol}: {current_ratio:.4f}")
        
        # Interpretar el resultado
        if current_ratio > 1.0:
            print(f"Hay {current_ratio:.2f} veces más volumen de compra que de venta (presión compradora)")
        else:
            print(f"Hay {1/current_ratio:.2f} veces más volumen de venta que de compra (presión vendedora)")
        
        # Calcular el desequilibrio ponderado
        imbalance = analyzer.get_imbalance_score(symbol, depth_levels=args.depth)
        print(f"Puntuación de desequilibrio: {imbalance:.4f} ({'+' if imbalance > 0 else ''}{imbalance*100:.1f}%)")
        
        # Calcular el historial de ratios
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=args.lookback)
        print(f"\nCalculando historial de ratios desde {start_time} hasta {end_time}...")
        
        ratio_history = analyzer.get_buy_sell_ratio_history(
            symbol, start_time, end_time, interval_minutes=args.interval
        )
        
        # Visualizar el historial de ratios
        if ratio_history:
            times = list(ratio_history.keys())
            ratios = list(ratio_history.values())
            
            print(f"Se encontraron {len(ratio_history)} puntos de datos.")
            
            plt.figure(figsize=(12, 6))
            plt.plot(times, ratios)
            plt.axhline(y=1.0, color='r', linestyle='--', label='Equilibrio')
            plt.title(f"Ratio de Órdenes Compra/Venta para {symbol}")
            plt.xlabel("Tiempo")
            plt.ylabel("Ratio (Compra/Venta)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            
            # Guardar la gráfica
            output_file = f"{symbol}_buy_sell_ratio.png"
            plt.savefig(output_file)
            print(f"\nGráfica guardada como: {output_file}")
            
            # Mostrar estadísticas básicas
            if ratios:
                avg_ratio = sum(ratios) / len(ratios)
                max_ratio = max(ratios)
                min_ratio = min(ratios)
                
                print(f"\nEstadísticas del período analizado:")
                print(f"- Ratio promedio: {avg_ratio:.4f}")
                print(f"- Ratio máximo: {max_ratio:.4f}")
                print(f"- Ratio mínimo: {min_ratio:.4f}")
                
                # Calcular tendencia
                if len(ratios) > 1:
                    first_half = sum(ratios[:len(ratios)//2]) / (len(ratios)//2)
                    second_half = sum(ratios[len(ratios)//2:]) / (len(ratios) - len(ratios)//2)
                    
                    if second_half > first_half:
                        print(f"- Tendencia: ALCISTA (incremento de {(second_half/first_half-1)*100:.1f}%)")
                    else:
                        print(f"- Tendencia: BAJISTA (disminución de {(1-second_half/first_half)*100:.1f}%)")
            
            plt.show()
        else:
            print("No se encontraron suficientes datos para generar el historial.")
    
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error inesperado: {e}")


if __name__ == "__main__":
    main()
