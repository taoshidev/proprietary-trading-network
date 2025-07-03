"""
Test position data for backtest_manager.py

This module contains predefined test positions used for backtesting scenarios.
Separated from the main backtest_manager.py to reduce clutter and improve maintainability.
"""

from vali_objects.enums.order_type_enum import OrderType
from vali_objects.vali_config import TradePair

# Test positions with realistic trading scenarios
TEST_POSITIONS = [
    {'miner_hotkey': '5DaW56UxJ9Dk14mvraGSEZhy1c91WyLuT2JnNrnKrwnzmZxk',
     'position_uuid': '51c02f65-ff69-3180-e035-524d01f178fe', 'open_ms': 1738199964405,
     'trade_pair': TradePair.NZDJPY, 'orders': [
        {'trade_pair': TradePair.NZDJPY, 'order_type': OrderType.SHORT, 'leverage': -5.0, 'price': 87.446,
         'processed_ms': 1738199964405, 'order_uuid': '51c02f65-ff69-3180-e035-524d01f178fe',
         'price_sources': [
             {'source': 'Polygon_ws', 'timespan_ms': 0, 'open': 87.446, 'close': 87.446, 'vwap': 87.446,
              'high': 87.446, 'low': 87.446, 'start_ms': 1738199964000, 'websocket': True, 'lag_ms': 405,
              'volume': 1.0},
             {'source': 'Tiingo_rest', 'timespan_ms': 0, 'open': 87.443, 'close': 87.443, 'vwap': 87.443,
              'high': 87.443, 'low': 87.443, 'start_ms': 1738199966615, 'websocket': True, 'lag_ms': 2210,
              'volume': None}], 'src': 0},
        {'trade_pair': TradePair.NZDJPY, 'order_type': OrderType.FLAT, 'leverage': 0.0, 'price': 87.419,
         'processed_ms': 1738203392320, 'order_uuid': '4463ff0f-beb2-7a24-468d-cb4401647f70',
         'price_sources': [
             {'source': 'Polygon_ws', 'timespan_ms': 0, 'open': 87.419, 'close': 87.419, 'vwap': 87.419,
              'high': 87.419, 'low': 87.419, 'start_ms': 1738203392000, 'websocket': True, 'lag_ms': 320,
              'volume': 1.0},
             {'source': 'Tiingo_rest', 'timespan_ms': 0, 'open': 87.417, 'close': 87.417, 'vwap': 87.417,
              'high': 87.417, 'low': 87.417, 'start_ms': 1738203394533, 'websocket': True, 'lag_ms': 2533,
              'volume': None}], 'src': 0}], 'current_return': 1.0003093170064654, 'close_ms': 1738203392320,
     'return_at_close': 1.0003093170064654, 'net_leverage': 0.0, 'average_entry_price': 87.446,
     'position_type': OrderType.FLAT, 'is_closed_position': True},

    {'miner_hotkey': '5G6CByTKqPZXvJ8dPVaK3HjFu1HySDDKF5D5WiGVCFaRJ7dH',
     'position_uuid': '6b3b7b7e-c7ac-4c5e-81d3-5dc47cdbe34b', 'open_ms': 1737030125304,
     'trade_pair': TradePair.BTCUSD, 'orders': [
        {'trade_pair': TradePair.BTCUSD, 'order_type': OrderType.LONG, 'leverage': 0.25, 'price': 104628.0,
         'processed_ms': 1737030125304, 'order_uuid': '6b3b7b7e-c7ac-4c5e-81d3-5dc47cdbe34b',
         'price_sources': [
             {'source': 'Polygon_ws', 'timespan_ms': 0, 'open': 104628.0, 'close': 104628.0, 'vwap': 104628.0,
              'high': 104628.0, 'low': 104628.0, 'start_ms': 1737030125000, 'websocket': True, 'lag_ms': 304,
              'volume': 1.0},
             {'source': 'Tiingo_rest', 'timespan_ms': 0, 'open': 104625.5, 'close': 104625.5, 'vwap': 104625.5,
              'high': 104625.5, 'low': 104625.5, 'start_ms': 1737030127516, 'websocket': True, 'lag_ms': 2516,
              'volume': None}], 'src': 0},
        {'trade_pair': TradePair.BTCUSD, 'order_type': OrderType.FLAT, 'leverage': 0.0, 'price': 105234.0,
         'processed_ms': 1737123456789, 'order_uuid': 'f1a2b3c4-d5e6-7890-1234-567890abcdef',
         'price_sources': [
             {'source': 'Polygon_ws', 'timespan_ms': 0, 'open': 105234.0, 'close': 105234.0, 'vwap': 105234.0,
              'high': 105234.0, 'low': 105234.0, 'start_ms': 1737123456000, 'websocket': True, 'lag_ms': 789,
              'volume': 1.0}], 'src': 0}], 'current_return': 1.0057923469387755, 'close_ms': 1737123456789,
     'return_at_close': 1.0057923469387755, 'net_leverage': 0.0, 'average_entry_price': 104628.0,
     'position_type': OrderType.FLAT, 'is_closed_position': True},

    {'miner_hotkey': '5H2K9xNfHe6Dh7hfJz8kL9mN0pQ1rS2tT3uU4vV5wW6xX7yY',
     'position_uuid': 'a1b2c3d4-e5f6-7890-1234-567890abcdef', 'open_ms': 1736550000000,
     'trade_pair': TradePair.EURUSD, 'orders': [
        {'trade_pair': TradePair.EURUSD, 'order_type': OrderType.SHORT, 'leverage': -1.5, 'price': 1.0456,
         'processed_ms': 1736550000000, 'order_uuid': 'a1b2c3d4-e5f6-7890-1234-567890abcdef',
         'price_sources': [
             {'source': 'Polygon_ws', 'timespan_ms': 0, 'open': 1.0456, 'close': 1.0456, 'vwap': 1.0456,
              'high': 1.0456, 'low': 1.0456, 'start_ms': 1736550000000, 'websocket': True, 'lag_ms': 100,
              'volume': 1.0}], 'src': 0},
        {'trade_pair': TradePair.EURUSD, 'order_type': OrderType.SHORT, 'leverage': -0.75, 'price': 1.0423,
         'processed_ms': 1736650000000, 'order_uuid': 'b2c3d4e5-f6g7-8901-2345-67890abcdef1',
         'price_sources': [
             {'source': 'Polygon_ws', 'timespan_ms': 0, 'open': 1.0423, 'close': 1.0423, 'vwap': 1.0423,
              'high': 1.0423, 'low': 1.0423, 'start_ms': 1736650000000, 'websocket': True, 'lag_ms': 150,
              'volume': 1.0}], 'src': 0},
        {'trade_pair': TradePair.EURUSD, 'order_type': OrderType.FLAT, 'leverage': 0.0, 'price': 1.0389,
         'processed_ms': 1736750000000, 'order_uuid': 'c3d4e5f6-g7h8-9012-3456-7890abcdef12',
         'price_sources': [
             {'source': 'Polygon_ws', 'timespan_ms': 0, 'open': 1.0389, 'close': 1.0389, 'vwap': 1.0389,
              'high': 1.0389, 'low': 1.0389, 'start_ms': 1736750000000, 'websocket': True, 'lag_ms': 200,
              'volume': 1.0}], 'src': 0}], 'current_return': 1.0123456789, 'close_ms': 1736750000000,
     'return_at_close': 1.0123456789, 'net_leverage': 0.0, 'average_entry_price': 1.0445,
     'position_type': OrderType.FLAT, 'is_closed_position': True},

    {'miner_hotkey': '5F3gHi4jKl5mN6oP7qR8sT9uV0wX1yZ2aB3cD4eF5gH6iJ7k',
     'position_uuid': 'd4e5f6g7-h8i9-0123-4567-890abcdef123', 'open_ms': 1736000000000,
     'trade_pair': TradePair.GBPUSD, 'orders': [
        {'trade_pair': TradePair.GBPUSD, 'order_type': OrderType.LONG, 'leverage': 2.0, 'price': 1.2456,
         'processed_ms': 1736000000000, 'order_uuid': 'd4e5f6g7-h8i9-0123-4567-890abcdef123',
         'price_sources': [
             {'source': 'Polygon_ws', 'timespan_ms': 0, 'open': 1.2456, 'close': 1.2456, 'vwap': 1.2456,
              'high': 1.2456, 'low': 1.2456, 'start_ms': 1736000000000, 'websocket': True, 'lag_ms': 250,
              'volume': 1.0}], 'src': 0},
        {'trade_pair': TradePair.GBPUSD, 'order_type': OrderType.FLAT, 'leverage': 0.0, 'price': 1.2523,
         'processed_ms': 1736100000000, 'order_uuid': 'e5f6g7h8-i9j0-1234-5678-90abcdef1234',
         'price_sources': [
             {'source': 'Polygon_ws', 'timespan_ms': 0, 'open': 1.2523, 'close': 1.2523, 'vwap': 1.2523,
              'high': 1.2523, 'low': 1.2523, 'start_ms': 1736100000000, 'websocket': True, 'lag_ms': 300,
              'volume': 1.0}], 'src': 0}], 'current_return': 1.0107728337, 'close_ms': 1736100000000,
     'return_at_close': 1.0107728337, 'net_leverage': 0.0, 'average_entry_price': 1.2456,
     'position_type': OrderType.FLAT, 'is_closed_position': True},

    {'miner_hotkey': '5E7yZ8aB9cD0eF1gH2iJ3kL4mN5oP6qR7sT8uV9wX0yZ1aB2',
     'position_uuid': 'f6g7h8i9-j0k1-2345-6789-0abcdef12345', 'open_ms': 1735800000000,
     'trade_pair': TradePair.USDJPY, 'orders': [
        {'trade_pair': TradePair.USDJPY, 'order_type': OrderType.SHORT, 'leverage': -1.0, 'price': 157.89,
         'processed_ms': 1735800000000, 'order_uuid': 'f6g7h8i9-j0k1-2345-6789-0abcdef12345',
         'price_sources': [
             {'source': 'Polygon_ws', 'timespan_ms': 0, 'open': 157.89, 'close': 157.89, 'vwap': 157.89,
              'high': 157.89, 'low': 157.89, 'start_ms': 1735800000000, 'websocket': True, 'lag_ms': 180,
              'volume': 1.0}], 'src': 0},
        {'trade_pair': TradePair.USDJPY, 'order_type': OrderType.FLAT, 'leverage': 0.0, 'price': 156.45,
         'processed_ms': 1735900000000, 'order_uuid': 'g7h8i9j0-k1l2-3456-7890-abcdef123456',
         'price_sources': [
             {'source': 'Polygon_ws', 'timespan_ms': 0, 'open': 156.45, 'close': 156.45, 'vwap': 156.45,
              'high': 156.45, 'low': 156.45, 'start_ms': 1735900000000, 'websocket': True, 'lag_ms': 220,
              'volume': 1.0}], 'src': 0}], 'current_return': 1.0091234567, 'close_ms': 1735900000000,
     'return_at_close': 1.0091234567, 'net_leverage': 0.0, 'average_entry_price': 157.89,
     'position_type': OrderType.FLAT, 'is_closed_position': True},

    # Multi-hotkey position example
    {'miner_hotkey': '5HCJ6okRkmCsu7iLEWotBxgcZy11RhbxSzs8MXT4Dei9osUx',
     'position_uuid': '5fcfe64b-b2f0-4060-addc-6379f44cf53c', 'open_ms': 1736151925629,
     'trade_pair': TradePair.EURUSD, 'orders': [
        {'trade_pair': TradePair.EURUSD, 'order_type': OrderType.SHORT, 'leverage': -0.5, 'price': 1.03342,
         'processed_ms': 1736151925629, 'order_uuid': '5fcfe64b-b2f0-4060-addc-6379f44cf53c',
         'price_sources': [], 'src': 0},
        {'trade_pair': TradePair.EURUSD, 'order_type': OrderType.SHORT, 'leverage': -0.25, 'price': 1.0292,
         'processed_ms': 1737102024911, 'order_uuid': '3c5512b5-741c-4d38-9656-6b50f48bfe24',
         'price_sources': [], 'src': 0},
        {'trade_pair': TradePair.EURUSD, 'order_type': OrderType.FLAT, 'leverage': 0.0, 'price': 0.0,
         'processed_ms': 1738011582117, 'order_uuid': 'c35fc44f9736-cdda-0604-0f2b-b46efcf5',
         'price_sources': [], 'src': 1}], 'current_return': 1.0020417642391284, 'close_ms': 1738011582117,
     'return_at_close': 1.0012786740908333, 'net_leverage': 0.0, 'average_entry_price': 1.0320133333333332,
     'position_type': OrderType.FLAT, 'is_closed_position': True},

    {'miner_hotkey': '5HCJ6okRkmCsu7iLEWotBxgcZy11RhbxSzs8MXT4Dei9osUx',
     'position_uuid': '2a003e6f-2835-4e33-aa3f-ea50e0e1ed08', 'open_ms': 1736843416198,
     'trade_pair': TradePair.AUDUSD, 'orders': [
        {'trade_pair': TradePair.AUDUSD, 'order_type': OrderType.SHORT, 'leverage': -0.4, 'price': 0.62034,
         'processed_ms': 1736843416198, 'order_uuid': '2a003e6f-2835-4e33-aa3f-ea50e0e1ed08',
         'price_sources': [], 'src': 0},
        {'trade_pair': TradePair.AUDUSD, 'order_type': OrderType.FLAT, 'leverage': 0.0, 'price': 0.0,
         'processed_ms': 1738011582117, 'order_uuid': '80de1e0e05ae-f3aa-33e4-5382-f6e300a2',
         'price_sources': [], 'src': 1}], 'current_return': 1.0, 'close_ms': 1738011582117,
     'return_at_close': 0.999972, 'net_leverage': 0.0, 'average_entry_price': 0.62034,
     'position_type': OrderType.FLAT, 'is_closed_position': True}
]

def get_test_positions():
    """
    Get the test positions data.
    
    Returns:
        List of test position dictionaries
    """
    return TEST_POSITIONS

def get_test_hotkeys():
    """
    Get unique miner hotkeys from test positions.
    
    Returns:
        List of unique miner hotkeys
    """
    return list(set(pos['miner_hotkey'] for pos in TEST_POSITIONS))

def get_time_range():
    """
    Calculate the time range covered by test positions.
    
    Returns:
        Tuple of (start_time_ms, end_time_ms)
    """
    all_times = []
    for pos in TEST_POSITIONS:
        for order in pos['orders']:
            all_times.append(order['processed_ms'])
    
    return min(all_times), max(all_times)