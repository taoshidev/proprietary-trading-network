from tests.shared_objects.mock_classes import MockLivePriceFetcher, MockPriceSlippageModel
from tests.vali_tests.base_objects.test_base import TestBase
from time_util.time_util import TimeUtil
from vali_objects.enums.order_type_enum import OrderType
# from vali_objects.utils.live_price_fetcher import LivePriceFetcher
from vali_objects.utils.price_slippage_model import PriceSlippageModel
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair
from vali_objects.vali_dataclasses.order import Order


class TestPriceSlippageModel(TestBase):
    def setUp(self):
        super().setUp()
        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        self.live_price_fetcher = MockLivePriceFetcher(secrets=secrets, disable_ws=True)
        self.psm = MockPriceSlippageModel(live_price_fetcher=self.live_price_fetcher)
        self.psm.refresh_features_daily(write_to_disk=False)

        self.DEFAULT_MINER_HOTKEY = "test_miner"
        self.DEFAULT_POSITION_UUID = "test_position"
        self.DEFAULT_ORDER_UUID = "test_order"
        self.DEFAULT_OPEN_MS = TimeUtil.now_in_millis()  # 1718071209000
        self.default_bid = 80
        self.default_ask = 100


    def test_equities_slippage(self):
        """
        test buy and sell order slippage, using slippage model
        """
        self.equities_order_buy = Order(price=1, processed_ms=self.DEFAULT_OPEN_MS,
                                            order_uuid=self.DEFAULT_ORDER_UUID,
                                            trade_pair=TradePair.NVDA,
                                            order_type=OrderType.LONG, leverage=1, side="buy")
        slippage_buy = PriceSlippageModel.calculate_slippage(self.default_bid, self.default_ask, self.equities_order_buy)

        self.equities_order_sell = Order(price=1, processed_ms=self.DEFAULT_OPEN_MS,
                                            order_uuid=self.DEFAULT_ORDER_UUID,
                                            trade_pair=TradePair.NVDA,
                                            order_type=OrderType.SHORT, leverage=1, side="sell")
        slippage_sell = PriceSlippageModel.calculate_slippage(self.default_bid, self.default_ask, self.equities_order_sell)

        ## assert slippage is proportional to order size
        self.equities_order_buy_large = Order(price=1, processed_ms=self.DEFAULT_OPEN_MS,
                                        order_uuid=self.DEFAULT_ORDER_UUID,
                                        trade_pair=TradePair.NVDA,
                                        order_type=OrderType.LONG, leverage=3, side="buy")
        large_slippage_buy = PriceSlippageModel.calculate_slippage(self.default_bid, self.default_ask, self.equities_order_buy_large)
        assert large_slippage_buy > slippage_buy

        self.equities_order_sell_small = Order(price=1, processed_ms=self.DEFAULT_OPEN_MS,
                                         order_uuid=self.DEFAULT_ORDER_UUID,
                                         trade_pair=TradePair.NVDA,
                                         order_type=OrderType.SHORT, leverage=0.1, side="sell")
        small_slippage_sell = PriceSlippageModel.calculate_slippage(self.default_bid, self.default_ask,
                                                              self.equities_order_sell_small)
        assert small_slippage_sell < slippage_sell

        ##

    def test_forex_slippage(self):
        """
        test buy and sell order slippage, using BB+ model
        """
        self.forex_order_buy = Order(price=1, processed_ms=self.DEFAULT_OPEN_MS,
                                        order_uuid=self.DEFAULT_ORDER_UUID,
                                        trade_pair=TradePair.USDCAD,
                                        order_type=OrderType.LONG, leverage=1, side="buy")
        slippage_buy = PriceSlippageModel.calculate_slippage(self.default_bid, self.default_ask,
                                                             self.forex_order_buy)

        self.forex_order_sell = Order(price=1, processed_ms=self.DEFAULT_OPEN_MS,
                                         order_uuid=self.DEFAULT_ORDER_UUID,
                                         trade_pair=TradePair.USDCAD,
                                         order_type=OrderType.SHORT, leverage=1, side="sell")
        slippage_sell = PriceSlippageModel.calculate_slippage(self.default_bid, self.default_ask,
                                                              self.forex_order_sell)

        ## assert slippage is proportional to order size
        self.forex_order_buy_large = Order(price=1, processed_ms=self.DEFAULT_OPEN_MS,
                                              order_uuid=self.DEFAULT_ORDER_UUID,
                                              trade_pair=TradePair.USDCAD,
                                              order_type=OrderType.LONG, leverage=3, side="buy")
        large_slippage_buy = PriceSlippageModel.calculate_slippage(self.default_bid, self.default_ask,
                                                                   self.forex_order_buy_large)
        assert large_slippage_buy > slippage_buy

        self.forex_order_sell_small = Order(price=1, processed_ms=self.DEFAULT_OPEN_MS,
                                               order_uuid=self.DEFAULT_ORDER_UUID,
                                               trade_pair=TradePair.USDCAD,
                                               order_type=OrderType.SHORT, leverage=0.1, side="sell")
        small_slippage_sell = PriceSlippageModel.calculate_slippage(self.default_bid, self.default_ask,
                                                                    self.forex_order_sell_small)
        assert small_slippage_sell < slippage_sell

    def test_crypto_slippage(self):
        """
        test buy and sell order slippage
        """
        self.crypto_order_buy = Order(price=1, processed_ms=self.DEFAULT_OPEN_MS,
                                     order_uuid=self.DEFAULT_ORDER_UUID,
                                     trade_pair=TradePair.BTCUSD,
                                     order_type=OrderType.LONG, leverage=0.25, side="buy")
        slippage_buy = PriceSlippageModel.calculate_slippage(self.default_bid, self.default_ask,
                                                             self.crypto_order_buy)

        self.crypto_order_sell = Order(price=1, processed_ms=self.DEFAULT_OPEN_MS,
                                      order_uuid=self.DEFAULT_ORDER_UUID,
                                      trade_pair=TradePair.SOLUSD,
                                      order_type=OrderType.SHORT, leverage=0.25, side="sell")
        slippage_sell = PriceSlippageModel.calculate_slippage(self.default_bid, self.default_ask,
                                                              self.crypto_order_sell)

        ## assert slippage is proportional to order size
        self.crypto_order_buy_large = Order(price=1, processed_ms=self.DEFAULT_OPEN_MS,
                                           order_uuid=self.DEFAULT_ORDER_UUID,
                                           trade_pair=TradePair.BTCUSD,
                                           order_type=OrderType.LONG, leverage=0.5, side="buy")
        large_slippage_buy = PriceSlippageModel.calculate_slippage(self.default_bid, self.default_ask,
                                                                   self.crypto_order_buy_large)
        ## crypto slippage does not depend on size
        assert large_slippage_buy == slippage_buy

        self.crypto_order_sell_small = Order(price=1, processed_ms=self.DEFAULT_OPEN_MS,
                                            order_uuid=self.DEFAULT_ORDER_UUID,
                                            trade_pair=TradePair.SOLUSD,
                                            order_type=OrderType.SHORT, leverage=0.1, side="sell")
        small_slippage_sell = PriceSlippageModel.calculate_slippage(self.default_bid, self.default_ask,
                                                                    self.crypto_order_sell_small)
        assert small_slippage_sell == slippage_sell



