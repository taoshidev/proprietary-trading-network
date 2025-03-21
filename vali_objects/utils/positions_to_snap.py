from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position
from vali_objects.vali_config import TradePair

positions_to_snap = [
    {'miner_hotkey': '5EWKUhycaBQHiHnfE3i2suZ1BvxAAE3HcsFsp8TaR6mu3JrJ',
     'position_uuid': 'bbc676e5-9ab3-4c11-8be4-5f2022ae9208', 'open_ms': 1742357759047, 'trade_pair': TradePair.BTCUSD,
     'orders': [{'trade_pair': TradePair.BTCUSD, 'order_type': OrderType.SHORT, 'leverage': -0.5, 'price': 82872.75,
                 'bid': 82880.06, 'ask': 82880.07, 'slippage': 1e-05, 'processed_ms': 1742357759047,
                 'order_uuid': 'bbc676e5-9ab3-4c11-8be4-5f2022ae9208', 'price_sources': [
             {'ask': 82880.07, 'bid': 82880.06, 'close': 82872.75, 'high': 82872.75, 'lag_ms': 1348, 'low': 82872.75,
              'open': 82872.75, 'source': 'Tiingo_gdax_rest', 'start_ms': 1742357760395, 'timespan_ms': 0,
              'vwap': 82872.75, 'websocket': True},
             {'ask': 0.0, 'bid': 0.0, 'close': 82869.24, 'high': 82880.07, 'lag_ms': 2048, 'low': 82869.24,
              'open': 82880.07, 'source': 'Polygon_rest', 'start_ms': 1742357756000, 'timespan_ms': 1000,
              'vwap': 82875.8692, 'websocket': False},
             {'ask': 0.0, 'bid': 0.0, 'close': 82047.16, 'high': 82047.16, 'lag_ms': 26043047, 'low': 82047.16,
              'open': 82047.16, 'source': 'Polygon_ws', 'start_ms': 1742331716000, 'timespan_ms': 0, 'vwap': 82047.16,
              'websocket': True}], 'src': 0},
                {'trade_pair': TradePair.BTCUSD, 'order_type': OrderType.FLAT, 'leverage': 0.5, 'price': 83202.0,
                 'bid': 83202.0, 'ask': 83202.0, 'slippage': 1e-05, 'processed_ms': 1742362860000,
                 'order_uuid': 'd6e1e768-9024-4cc5-84d7-d66691d82061', 'price_sources': [], 'src': 0}],
     'current_return': 0.9980034808990859, 'close_ms': 1742362860000, 'net_leverage': 0.0,
     'return_at_close': 0.9980034808990859, 'average_entry_price': 82871.92127250001, 'position_type': OrderType.FLAT,
     'is_closed_position': True}
]

if __name__ == "__main__":
    for position_json in positions_to_snap:
        pos = Position(**position_json)
        pos.rebuild_position_with_updated_orders()
        print(pos.to_copyable_str())
