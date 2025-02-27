from vali_objects.vali_config import TradePair
from vali_objects.enums.order_type_enum import OrderType
positions_to_snap = [
    {'miner_hotkey': '5EWKUhycaBQHiHnfE3i2suZ1BvxAAE3HcsFsp8TaR6mu3JrJ',
     'position_uuid': '22be58e8-749a-4a3b-a2ff-0cc4bf7cf063', 'open_ms': 1739470155445, 'trade_pair': TradePair.XRPUSD,
     'orders': [{'trade_pair': TradePair.XRPUSD, 'order_type': OrderType.SHORT, 'leverage': -0.5, 'price': 2.4236,
                 'processed_ms': 1739470155445, 'order_uuid': '22be58e8-749a-4a3b-a2ff-0cc4bf7cf063', 'price_sources': [
             {'source': 'Polygon_ws', 'timespan_ms': 0, 'open': 2.4236, 'close': 2.4236, 'vwap': 2.4236, 'high': 2.4236,
              'low': 2.4236, 'start_ms': 1739470155000, 'websocket': True, 'lag_ms': 445, 'volume': 19.616137},
             {'source': 'Tiingo_gdax_rest', 'timespan_ms': 0, 'open': 2.4238, 'close': 2.4238, 'vwap': 2.4238,
              'high': 2.4238, 'low': 2.4238, 'start_ms': 1739470156086, 'websocket': True, 'lag_ms': 641,
              'volume': None}], 'src': 0},
                {'trade_pair': TradePair.XRPUSD, 'order_type': OrderType.FLAT, 'leverage': 0.05, 'price': 2.425,
                 'processed_ms': 1739471831000, 'order_uuid': '98dfdc19-40ff-43a6-bdbd-86139c29fdf7',
                 'price_sources': [],
                 'src': 0}], 'current_return': 0.9997111734609672, 'close_ms': 1739471831,
     'return_at_close': 0.9992113178742368, 'net_leverage': 0.0, 'average_entry_price': 2.4236,
     'position_type': OrderType.FLAT, 'is_closed_position': True},

    {'miner_hotkey': '5HYBzAsTcxDXxHNXBpUJAQ9ZwmaGTwTb24ZBGJpELpG7LPGf',
     'position_uuid': 'ad94908b-0dac-44a2-9817-418a57e1d382', 'open_ms': 1739464206269, 'trade_pair': TradePair.BTCUSD,
     'orders': [{'trade_pair': TradePair.BTCUSD, 'order_type': OrderType.SHORT, 'leverage': -0.5, 'price': 95444.22,
                 'processed_ms': 1739464206269, 'order_uuid': 'ad94908b-0dac-44a2-9817-418a57e1d382', 'price_sources': [
             {'source': 'Polygon_ws', 'timespan_ms': 0, 'open': 95444.22, 'close': 95444.22, 'vwap': 95444.22,
              'high': 95444.22, 'low': 95444.22, 'start_ms': 1739464206000, 'websocket': True, 'lag_ms': 269,
              'volume': 8.27e-06},
             {'source': 'Tiingo_gdax_rest', 'timespan_ms': 0, 'open': 95444.22, 'close': 95444.22, 'vwap': 95444.22,
              'high': 95444.22, 'low': 95444.22, 'start_ms': 1739464205568, 'websocket': True, 'lag_ms': 701,
              'volume': None}], 'src': 0},
                {'trade_pair': TradePair.BTCUSD, 'order_type': OrderType.FLAT, 'leverage': 0.0, 'price': 96886.23,
                 'processed_ms': 1739492888000, 'order_uuid': 'ad94908b-0dac-44a2-9817-418a57e1d38x',
                 'price_sources': [], 'src': 0}], 'current_return': 0.9924457971367989, 'close_ms': 1739492888000,
     'return_at_close': 0.9918970466103956, 'net_leverage': 0.0, 'average_entry_price': 95444.22,
     'position_type': OrderType.FLAT, 'is_closed_position': True}

]