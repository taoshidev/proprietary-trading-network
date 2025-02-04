from tests.shared_objects.mock_classes import MockLivePriceFetcher, MockPriceSlippageModel
from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair


class TestPriceSlippageModel(TestBase):
    def setUp(self):
        super().setUp()
        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        self.live_price_fetcher = MockLivePriceFetcher(secrets=secrets, disable_ws=True)
        self.psm = MockPriceSlippageModel(live_price_fetcher=self.live_price_fetcher)

    # TODO: test each, also outside of traded hours
    # TODO: test each with large and small size

    def test_equities_slippage(self):
        """
        test buy and sell order slippage
        """
        slippage_buy = self.psm.calculate_slippage(TradePair.AAPL, "buy", 75000, 1737655200000)
        print(slippage_buy)
        slippage_sell = self.psm.calculate_slippage(TradePair.AAPL, "sell", 275000, 1737655200000)
        print(slippage_sell)

    def test_equities_slippage_size(self):
        """
        larger size should equal larger slippage
        smaller size should equal smaller slippage
        """
        slippage_less = self.psm.calculate_slippage(TradePair.AAPL, "buy", 5000, 1737655200000)
        slippage_more = self.psm.calculate_slippage(TradePair.AAPL, "buy", 275000, 1737655200000)
        assert slippage_less < slippage_more

        slippage_less = self.psm.calculate_slippage(TradePair.AAPL, "sell", 75000, 1737655200000)
        slippage_more = self.psm.calculate_slippage(TradePair.AAPL, "sell", 275000, 1737655200000)
        assert slippage_less < slippage_more

    def test_forex_slippage(self):
        """
        test buy and sell order slippage
        """
        slippage = self.psm.calculate_slippage(TradePair.EURCAD, "buy", 75000, 1737655200000)
        print(slippage)
        slippage = self.psm.calculate_slippage(TradePair.USDJPY, "sell", 275000, 1737655200000)
        print(slippage)

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
        slippage_btc_buy_small = self.psm.calculate_slippage(TradePair.BTCUSD, "buy", 25000, 1737655200000)
        slippage_btc_buy_large = self.psm.calculate_slippage(TradePair.BTCUSD, "buy", 50000, 1737655200000)
        slippage_btc_sell_small = self.psm.calculate_slippage(TradePair.BTCUSD, "sell", 25000, 1737655200000)
        slippage_btc_sell_large = self.psm.calculate_slippage(TradePair.BTCUSD, "sell", 50000, 1737655200000)

        assert slippage_btc_buy_small < slippage_btc_buy_large
        assert slippage_btc_sell_small < slippage_btc_sell_large

    def test_crypto_slippage_asset(self):
        """
        btcusd and ethusd have less slippage than solusd, xrpusd, and dogeusd
        crypto slippage is constant below 50k
        """
        slippage_btc = self.psm.calculate_slippage(TradePair.BTCUSD, "buy", 7500, 1737655200000)
        slippage_sol = self.psm.calculate_slippage(TradePair.SOLUSD, "sell", 7500, 1737655200000)
        assert slippage_btc < slippage_sol

    def test_commodities_slippage(self):
        """
        commodities are treated as forex
        """
        slippage_gold_buy_small = self.psm.calculate_slippage(TradePair.XAUUSD, "buy", 25000, 1737655200000)
        slippage_gold_buy_large = self.psm.calculate_slippage(TradePair.XAUUSD, "buy", 350000, 1737655200000)
        slippage_gold_sell_small = self.psm.calculate_slippage(TradePair.XAUUSD, "sell", 25000, 1737655200000)
        slippage_gold_sell_large = self.psm.calculate_slippage(TradePair.XAUUSD, "sell", 350000, 1737655200000)
        assert slippage_gold_buy_small < slippage_gold_buy_large
        assert slippage_gold_sell_small < slippage_gold_sell_large

        slippage_silver_buy_small = self.psm.calculate_slippage(TradePair.XAGUSD, "buy", 25000, 1737655200000)
        slippage_silver_buy_large = self.psm.calculate_slippage(TradePair.XAGUSD, "buy", 350000, 1737655200000)
        slippage_silver_sell_small = self.psm.calculate_slippage(TradePair.XAGUSD, "sell", 25000, 1737655200000)
        slippage_silver_sell_large = self.psm.calculate_slippage(TradePair.XAGUSD, "sell", 350000, 1737655200000)
        assert slippage_silver_buy_small < slippage_silver_buy_large
        assert slippage_silver_sell_small < slippage_silver_sell_large

