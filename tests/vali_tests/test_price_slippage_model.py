from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.utils.price_slippage_model import PriceSlippageModel
from vali_objects.vali_config import TradePair


class TestPriceSlippageModel(TestBase):
    def setUp(self):
        super().setUp()
        self.psm = PriceSlippageModel()

    # TODO: test each, also outside of traded hours
    # TODO: test each with large and small size

    def test_equities_slippage(self):
        """
        test buy and sell order slippage
        """
        slippage_buy = self.psm.calculate_slippage(TradePair.AAPL, "buy", 75000, 1737655200000)
        print(slippage_buy)
        slippage_sell = self.psm.calculate_slippage(TradePair.AAPL, "sell", 275000, 1737655200000)

    def test_equities_slippage_size(self):
        """
        larger size should equal larger slippage
        smaller size should equal smaller slippage
        """
        slippage_less = self.psm.calculate_slippage(TradePair.AAPL, "buy", 75000, 1737655200000)
        slippage_more = self.psm.calculate_slippage(TradePair.AAPL, "buy", 275000, 1737655200000)
        assert slippage_less < slippage_more

        slippage_less = self.psm.calculate_slippage(TradePair.AAPL, "sell", 75000, 1737655200000)
        slippage_more = self.psm.calculate_slippage(TradePair.AAPL, "sell", 275000, 1737655200000)
        assert slippage_less < slippage_more

    def test_forex_slippage(self):
        """
        test buy and sell order slippage
        """
        slippage = self.psm.calculate_slippage(TradePair.USDJPY, "buy", 75000, 1737655200000)
        print(slippage)
        slippage = self.psm.calculate_slippage(TradePair.USDJPY, "sell", 275000, 1737655200000)

    def test_forex_slippage_size(self):
        """
        larger size should equal larger slippage
        smaller size should equal smaller slippage
        """
        slippage_less = self.psm.calculate_slippage(TradePair.USDJPY, "buy", 75000, 1737655200000)
        slippage_more = self.psm.calculate_slippage(TradePair.USDJPY, "buy", 275000, 1737655200000)
        assert slippage_less < slippage_more

        slippage_less = self.psm.calculate_slippage(TradePair.USDJPY, "sell", 75000, 1737655200000)
        slippage_more = self.psm.calculate_slippage(TradePair.USDJPY, "sell", 275000, 1737655200000)
        assert slippage_less < slippage_more

    def test_crypto_slippage(self):
        """
        test buy and sell order slippage
        """
        slippage = self.psm.calculate_slippage(TradePair.BTCUSD, "buy", 75000, 1737655200000)
        print(slippage)
        slippage = self.psm.calculate_slippage(TradePair.BTCUSD, "sell", 275000, 1737655200000)

    def test_crypto_slippage_asset(self):
        """
        btcusd and ethusd have less slippage than solusd, xrpusd, and dogeusd
        crypto slippage is constant below 50k
        """
        slippage_btc = self.psm.calculate_slippage(TradePair.BTCUSD, "buy", 7500, 1737655200000)
        slippage_sol = self.psm.calculate_slippage(TradePair.SOLUSD, "sell", 7500, 1737655200000)
        assert slippage_btc < slippage_sol

    # TODO: test a buy and sell order with slippage, compare to without
