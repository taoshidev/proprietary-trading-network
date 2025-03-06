from vali_objects.position import Position
from vali_objects.vali_config import TradePair
from vali_objects.enums.order_type_enum import OrderType
positions_to_snap = [
    {'miner_hotkey': '5EWKUhycaBQHiHnfE3i2suZ1BvxAAE3HcsFsp8TaR6mu3JrJ',
     'position_uuid': '5a1498f1-31f5-4261-8718-edd0d6175689', 'open_ms': 1740932673738, 'trade_pair': TradePair.BTCUSD,
     'orders': [{'trade_pair': TradePair.BTCUSD, 'order_type': OrderType.SHORT, 'leverage': -0.5, 'price': 88914.94,
                 'bid': 0.0, 'ask': 0.0, 'slippage': 1e-05, 'processed_ms': 1740932673738,
                 'order_uuid': '5a1498f1-31f5-4261-8718-edd0d6175689', 'price_sources': [
             {'source': 'Polygon_ws', 'timespan_ms': 0, 'open': 88914.94, 'close': 88914.94, 'vwap': 88914.94,
              'high': 88914.94, 'low': 88914.94, 'start_ms': 1740932674000, 'websocket': True, 'lag_ms': 262,
              'volume': 1.104e-05},
             {'source': 'Tiingo_gdax_rest', 'timespan_ms': 0, 'open': 88971.9, 'close': 88971.9, 'vwap': 88971.9,
              'high': 88971.9, 'low': 88971.9, 'start_ms': 1740932676265, 'websocket': True, 'lag_ms': 2527,
              'volume': None}], 'quote_sources': [
             {'source': 'Polygon_rest', 'timestamp_ms': 0, 'bid': 0.0, 'ask': 0.0, 'websocket': False,
              'lag_ms': 1740932673738}], 'src': 0},

            {'trade_pair': TradePair.BTCUSD, 'order_type': OrderType.FLAT, 'leverage': 0.5, 'price': 89271.58,
             'bid': 0.0, 'ask': 0.0, 'slippage': 1e-05, 'processed_ms': 1740933125000,
             'order_uuid': 'ba6ccb69-b6b7-4514-b891-23dd22a1b5d0', 'price_sources': [], 'quote_sources': [], 'src': 0}],
     'current_return': 0.9979844475548514,
     'close_ms': 1740933125000, 'net_leverage': 0.0, 'return_at_close': 0.9979844475548514,
     'average_entry_price': 88914.0508506, 'position_type': OrderType.FLAT, 'is_closed_position': True}
]

if __name__ == "__main__":
    for position_json in positions_to_snap:
        pos = Position(**position_json)
        pos.rebuild_position_with_updated_orders()
        print(pos.to_copyable_str())