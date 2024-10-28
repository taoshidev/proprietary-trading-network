from unittest.mock import patch

from tests.vali_tests.base_objects.test_base import TestBase
from time_util.time_util import TimeUtil
from vali_objects.vali_config import TradePair
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position
from vali_objects.vali_dataclasses.order import Order
from vali_objects.vali_dataclasses.perf_ledger import PerfLedgerManager

class TestPerfLedgers(TestBase):

    def setUp(self):
        super().setUp()
        self.DEFAULT_MINER_HOTKEY = "test_miner"
        self.DEFAULT_POSITION_UUID = "test_position"
        self.DEFAULT_ORDER_UUID = "test_order"
        self.DEFAULT_OPEN_MS = 1718071209000
        self.DEFAULT_TRADE_PAIR = TradePair.BTCUSD
        self.default_order = Order(price=60000, processed_ms=self.DEFAULT_OPEN_MS, order_uuid=self.DEFAULT_ORDER_UUID, trade_pair=self.DEFAULT_TRADE_PAIR,
                                     order_type=OrderType.LONG, leverage=1)

        self.default_open_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid=self.DEFAULT_POSITION_UUID,
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            orders=[self.default_order],
            position_type=OrderType.LONG
        )

    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_basic(self, mock_unified_candle_fetcher):
        mock_unified_candle_fetcher.return_value = {}

        perf_ledger_manager = PerfLedgerManager(metagraph=None, running_unit_tests=True)
        hotkey_to_positions = {self.DEFAULT_MINER_HOTKEY: [self.default_open_position]}
        ans = perf_ledger_manager.generate_perf_ledgers_for_analysis(hotkey_to_positions)
        for x in ans[self.DEFAULT_MINER_HOTKEY].cps:
            last_update_formated = TimeUtil.millis_to_timestamp(x.last_update_ms)
            print(x, last_update_formated)
        print('max_perf_ledger_return:', ans[self.DEFAULT_MINER_HOTKEY].max_return)
        assert len(ans) == 1, ans

