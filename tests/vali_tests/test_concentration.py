import copy
from copy import deepcopy
from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.utils.ledger_utils import LedgerUtils
from vali_objects.vali_config import ValiConfig, TradePair

from vali_objects.utils.ledger_utils import LedgerUtils
from vali_objects.utils.position_penalties import PositionPenalties

from vali_objects.utils.functional_utils import FunctionalUtils
from tests.shared_objects.test_utilities import generate_ledger, ledger_generator, position_generator

class TestConcentration(TestBase):
    def setUp(self):
        super().setUp()

    ## Testing the daily log returns concentration metrics
    # 1. Concentration with no values works
    # 2. Concentration with a single value works
    # 3. Similar length - low concentration is lower than high concentration
    # 4. Monotomic characteristic - addition of new values decreases concentration
    def test_concentration(self):
        # Daily returns
        daily_log_returns = []
        concentration = FunctionalUtils.concentration(daily_log_returns)
        self.assertEqual(concentration, 1)

    def test_concentration_single_value(self):
        # Daily returns
        daily_log_returns = [0.1]
        concentration = FunctionalUtils.concentration(daily_log_returns)
        self.assertEqual(concentration, 1)

    def test_concentration_similar_length(self):
        # Daily returns
        daily_low = [0.1, 0.1, 0.1, 0.1, 0.1]
        daily_high = [0.0, 0.0, 0.0, 0.5, 0.0]
        concentration_low = FunctionalUtils.concentration(daily_low)
        concentration_high = FunctionalUtils.concentration(daily_high)
        self.assertLess(concentration_low, concentration_high)

    def test_concentration_monotomic(self):
        # Daily returns
        daily_log_returns = [0.1, 0.1, 0.1, 0.1, 0.1]
        concentration = FunctionalUtils.concentration(daily_log_returns)
        daily_log_returns.append(0.1)
        concentration_new = FunctionalUtils.concentration(daily_log_returns)
        self.assertLess(concentration_new, concentration)

    def test_concentration_sign_invariant(self):
        # Daily returns
        daily_log_returns = [0.1, 0.1, 0.1, 0.1, 0.1]
        concentration = FunctionalUtils.concentration(daily_log_returns)
        daily_log_returns.append(-0.1)
        concentration_new = FunctionalUtils.concentration(daily_log_returns)
        self.assertLess(concentration_new, concentration)  # This should still be less
        
    ## Now moving to daily return specific concentration
    def test_daily_log_returns_concentration(self):
        # Daily returns with no elements
        sample_ledger = ledger_generator(checkpoints=[])
        log_returns = LedgerUtils.daily_return_log(sample_ledger.cps)
        concentration_penalty = FunctionalUtils.concentration(log_returns)
        self.assertEqual(concentration_penalty, 1)

        # Daily returns with a single element
        sample_ledger = generate_ledger(gain=0.1, loss=0.0, nterms=1)
        log_returns = LedgerUtils.daily_return_log(sample_ledger.cps)
        concentration_penalty = FunctionalUtils.concentration(log_returns)
        self.assertEqual(concentration_penalty, 1)

        # Daily returns with multiple elements
        sample_ledger = generate_ledger(gain=0.1, loss=0.0, nterms=50)
        sample_concentrated = copy.deepcopy(sample_ledger)
        sample_concentrated.cps[-1].gain = 0.5

        log_returns = LedgerUtils.daily_return_log(sample_ledger.cps)
        log_returns_concentrated = LedgerUtils.daily_return_log(sample_concentrated.cps)

        concentration_penalty = FunctionalUtils.concentration(log_returns)
        concentration_penalty_concentrated = FunctionalUtils.concentration(log_returns_concentrated)
        self.assertLess(concentration_penalty, concentration_penalty_concentrated)

    def test_positional_concentration(self):
        # Daily returns with no elements
        positional_returns = [1.01, 1.01, 1.01, 1.01, 1.01]
        positions = []

        for positional_return in positional_returns:
            sample_position = position_generator(
                open_time_ms=0,
                close_time_ms=100,
                trade_pair=TradePair.BTCUSD,
                return_at_close=positional_return
            )
            positions.append(sample_position)

        positional_returns = [position.return_at_close for position in positions]
        concentration_penalty = FunctionalUtils.concentration(positional_returns)

        self.assertGreater(concentration_penalty, 0)
        self.assertLess(concentration_penalty, 1)

