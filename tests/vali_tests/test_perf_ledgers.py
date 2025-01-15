from unittest.mock import patch

from tests.vali_tests.base_objects.test_base import TestBase
from time_util.time_util import TimeUtil
from vali_objects.utils.elimination_manager import EliminationManager
from vali_objects.utils.position_manager import PositionManager
from vali_objects.vali_config import TradePair
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position
from vali_objects.vali_dataclasses.order import Order
from vali_objects.vali_dataclasses.perf_ledger import PerfLedgerManager, TP_ID_PORTFOLIO

class TestPerfLedgers(TestBase):

    def setUp(self):
        super().setUp()
        self.DEFAULT_MINER_HOTKEY = "test_miner"
        self.DEFAULT_OPEN_MS = 1718071209000
        self.DEFAULT_TRADE_PAIR = TradePair.BTCUSD
        self.default_btc_order = Order(price=60000, processed_ms=self.DEFAULT_OPEN_MS, order_uuid="test_order_btc", trade_pair=self.DEFAULT_TRADE_PAIR,
                                     order_type=OrderType.LONG, leverage=.5)
        self.default_nvda_order = Order(price=100, processed_ms=self.DEFAULT_OPEN_MS + 1000 * 60 * 60 * 24 * 5, order_uuid="test_order_nvda", trade_pair=TradePair.NVDA,
                                     order_type=OrderType.LONG, leverage=1)
        self.default_usdjpy_order = Order(price=156, processed_ms=self.DEFAULT_OPEN_MS + 1000 * 60 * 60 * 24 * 10, order_uuid="test_order_usdjpy",
                                          trade_pair=TradePair.USDJPY, order_type=OrderType.LONG, leverage=1)

        self.default_btc_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid="test_position_btc",
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            orders=[self.default_btc_order],
            position_type=OrderType.LONG
        )
        self.default_nvda_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid="test_position_nvda",
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=TradePair.NVDA,
            orders=[self.default_nvda_order],
            position_type=OrderType.LONG
        )
        self.default_usdjpy_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid="test_position_usdjpy",
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=TradePair.USDJPY,
            orders=[self.default_usdjpy_order],
            position_type=OrderType.LONG
        )

        elimination_manager = EliminationManager(None, None, None)
        position_manager = PositionManager(metagraph=None, running_unit_tests=True, elimination_manager=elimination_manager)
        self.perf_ledger_manager = PerfLedgerManager(metagraph=None, running_unit_tests=True, position_manager=position_manager)

    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_basic(self, mock_unified_candle_fetcher):
        mock_unified_candle_fetcher.return_value = {}
        hotkey_to_positions = {self.DEFAULT_MINER_HOTKEY: [self.default_btc_position]}
        ans = self.perf_ledger_manager.generate_perf_ledgers_for_analysis(hotkey_to_positions)
        for hk, dat in ans.items():
            for tp_id, pl in dat.items():
                print('-----------', tp_id, '-----------')
                for idx, x in enumerate(pl.cps):
                    last_update_formatted = TimeUtil.millis_to_timestamp(x.last_update_ms)
                    if idx == 0 or idx == len(pl.cps) - 1:
                        print(x, last_update_formatted)
                print(tp_id, 'max_perf_ledger_return:', pl.max_return)

        assert len(ans) == 1, ans

    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_multiple_tps(self, mock_unified_candle_fetcher):
        mock_unified_candle_fetcher.return_value = {}
        hotkey_to_positions = {self.DEFAULT_MINER_HOTKEY:
                                   [self.default_btc_position, self.default_nvda_position, self.default_usdjpy_position]}
        ans = self.perf_ledger_manager.generate_perf_ledgers_for_analysis(hotkey_to_positions)
        for hk, dat in ans.items():
            for tp_id, pl in dat.items():
                print('-----------', tp_id, '-----------')
                for idx, x in enumerate(pl.cps):
                    last_update_formatted = TimeUtil.millis_to_timestamp(x.last_update_ms)
                    if idx == 0 or idx == len(pl.cps) - 1:
                        print(x, last_update_formatted)
                print(tp_id, 'max_perf_ledger_return:', pl.max_return)

        tp_to_position_start_time = {}
        for position in hotkey_to_positions[self.DEFAULT_MINER_HOTKEY]:
            if position.trade_pair == TradePair.BTCUSD:
                tp_to_position_start_time[position.trade_pair.trade_pair_id] = self.default_btc_position.open_ms
            elif position.trade_pair == TradePair.NVDA:
                tp_to_position_start_time[position.trade_pair.trade_pair_id] = self.default_nvda_position.open_ms
            elif position.trade_pair == TradePair.USDJPY:
                tp_to_position_start_time[position.trade_pair] = self.default_usdjpy_position.open_ms

        self.assertEqual(len(ans), 1)
        self.assertEqual(len(ans[self.DEFAULT_MINER_HOTKEY]), 4)
        self.assertIn(TP_ID_PORTFOLIO, ans[self.DEFAULT_MINER_HOTKEY])
        for tp_id in tp_to_position_start_time:
            bundle = ans[self.DEFAULT_MINER_HOTKEY]
            self.assertIn(tp_id, ans[self.DEFAULT_MINER_HOTKEY])
            self.assertEqual(tp_to_position_start_time[tp_id], bundle[tp_id].cps[0].last_update_ms, tp_id)

        assert len(ans) == 1, ans

