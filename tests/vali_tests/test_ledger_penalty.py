import copy
from copy import deepcopy
from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.utils.ledger_utils import LedgerUtils

from vali_config import ValiConfig

from tests.shared_objects.test_utilities import generate_ledger
import random


class TestLedgerPenalty(TestBase):
    """
    This class will only test the positions and the consistency metrics associated with positions.
    """

    def setUp(self):
        super().setUp()
        # seeding
        random.seed(0)

        self.START_TIME = 0
        self.END_TIME = ValiConfig.TARGET_LEDGER_WINDOW_MS

        self.DEFAULT_LEDGER = generate_ledger(0.1, start_time=self.START_TIME, end_time=self.END_TIME)

    # Want to test the penalties on each of the ledgers
    # 1. Daily Consistency
    # 2. Biweekly Consistency
    # 3. Max Drawdown Threshold

    def test_daily_consistency(self):
        daily_window = ValiConfig.DAILY_CHECKPOINTS
        biweekly_window = ValiConfig.BIWEEKLY_CHECKPOINTS

        l1 = deepcopy(self.DEFAULT_LEDGER)
        l2 = deepcopy(self.DEFAULT_LEDGER)

        # Consistent growth
        l3 = generate_ledger(
            gain=0.2,
            loss=-0.1,
            start_time=self.START_TIME,
            end_time=self.END_TIME
        )
        l4 = copy.deepcopy(l3)

        l1_cps = l1.cps
        l2_cps = l2.cps
        l3_cps = l3.cps
        l4_cps = l4.cps

        # Inconsistent growth period - one interval
        l2_cps[len(l2_cps) // 2].gain = 1.0

        # Total growth for consistent ledger
        l3_return = sum([cp.gain + cp.loss for cp in l3_cps])
        l3_return_inconsistent_period = l3_return * 10

        l4_midpoint = len(l4_cps) // 2

        # Huge growth at the tail period, but spread over many days
        for i in range(l4_midpoint, len(l4_cps)):
            l4_cps[i].gain = l3_return_inconsistent_period

        # Many days, but all in one week
        l5 = deepcopy(self.DEFAULT_LEDGER)
        l5_cps = l5.cps

        # Set all gains to 1.0 for these days
        n_days_in_biweekly = biweekly_window // daily_window

        for i in range(0, biweekly_window, daily_window):
            l5_cps[min(i, len(l5_cps) - 1)].gain = 1.0

        # Daily consistency penalties and ratios
        self.assertAlmostEqual(LedgerUtils.daily_consistency_penalty(l1_cps), 0.0)
        self.assertEqual(LedgerUtils.daily_consistency_ratio(l1_cps), 1.0)

        self.assertAlmostEqual(LedgerUtils.daily_consistency_penalty(l2_cps), 0.0)
        self.assertEqual(LedgerUtils.daily_consistency_ratio(l2_cps), 1.0)

        self.assertAlmostEqual(LedgerUtils.daily_consistency_penalty(l3_cps), 1.0, places=3)  # should have small penalty
        self.assertGreater(LedgerUtils.daily_consistency_ratio(l3_cps), 0.0)

        self.assertLess(LedgerUtils.daily_consistency_ratio(l3_cps), 0.5)  # Less than half of returns come from a day
        self.assertAlmostEqual(LedgerUtils.daily_consistency_ratio(l3_cps), 1 / len(l3_cps))  # More than a perfect ledger ratio, as we sum the days
        self.assertGreater(LedgerUtils.daily_consistency_penalty(l3_cps), 0.0)
        self.assertLess(LedgerUtils.daily_consistency_penalty(l3_cps), 1.0)

        self.assertLess(LedgerUtils.daily_consistency_ratio(l4_cps), 0.5)  # Less than half of returns come from a day
        self.assertGreater(LedgerUtils.daily_consistency_ratio(l4_cps), 1 / len(l4_cps))  # More than a perfect ledger ratio, as we sum the days
        self.assertGreater(LedgerUtils.daily_consistency_penalty(l4_cps), 0.0)
        self.assertLess(LedgerUtils.daily_consistency_penalty(l4_cps), 1.0)

        self.assertGreater(LedgerUtils.daily_consistency_penalty(l5_cps), 0.0)
        self.assertLess(LedgerUtils.daily_consistency_penalty(l5_cps), 1.0)
        self.assertLess(LedgerUtils.daily_consistency_ratio(l5_cps), 1.0)
        self.assertGreater(LedgerUtils.daily_consistency_ratio(l5_cps), 0.0)
        self.assertAlmostEqual(LedgerUtils.daily_consistency_ratio(l5_cps), 1 / n_days_in_biweekly)

    def test_biweekly_consistency(self):
        daily_window = ValiConfig.DAILY_CHECKPOINTS
        biweekly_window = ValiConfig.BIWEEKLY_CHECKPOINTS

        l1 = generate_ledger(0.1, start_time=0)
        l2 = generate_ledger(0.1)

        # Consistent growth
        l3 = generate_ledger(
            gain=0.2,
            loss=-0.1
        )
        l4 = copy.deepcopy(l3)

        l1_cps = l1.cps
        l2_cps = l2.cps
        l3_cps = l3.cps
        l4_cps = l4.cps

        # Inconsistent growth period - one interval
        l2_cps[len(l2_cps) // 2].gain = 1.0

        # Total growth for consistent ledger
        l3_return = sum([cp.gain + cp.loss for cp in l3_cps])
        l3_return_inconsistent_period = l3_return * 10

        # Huge growth at the most recent interval
        tail_tart = len(l4_cps) - (len(l4_cps) // 5)
        for i in range(tail_tart, len(l4_cps)):
            l4_cps[i].gain = l3_return_inconsistent_period

        # Many days, but all in one week
        l5 = generate_ledger(0.1)
        l5_cps = l5.cps

        # Set all gains to 1.0 for these days
        for i in range(0, biweekly_window, daily_window):
            l5_cps[min(i, len(l5_cps) - 1)].gain = 1.0

        # Daily consistency penalties and ratios
        self.assertAlmostEqual(LedgerUtils.biweekly_consistency_penalty(l1_cps), 0.0)
        self.assertEqual(LedgerUtils.biweekly_consistency_ratio(l1_cps), 1.0)

        self.assertAlmostEqual(LedgerUtils.biweekly_consistency_penalty(l2_cps), 0.0)
        self.assertEqual(LedgerUtils.biweekly_consistency_ratio(l2_cps), 1.0)

        self.assertLess(LedgerUtils.biweekly_consistency_penalty(l3_cps), 1.0)  # should have small penalty
        self.assertGreater(LedgerUtils.biweekly_consistency_ratio(l3_cps), 0.0)

        self.assertLess(LedgerUtils.biweekly_consistency_ratio(l3_cps),0.5)  # Less than half of returns come from a day
        self.assertGreater(LedgerUtils.biweekly_consistency_ratio(l3_cps), 1 / len(l3_cps))  # More than a perfect ledger ratio, as we sum the days
        self.assertGreater(LedgerUtils.biweekly_consistency_penalty(l3_cps), 0.0)
        self.assertLess(LedgerUtils.biweekly_consistency_penalty(l3_cps), 1.0)

        # Ratio between the largest interval and overall is pretty high, almost 1.0
        self.assertGreater(
            LedgerUtils.biweekly_consistency_ratio(l4_cps),
            0.9
        )  # Less than half of returns come from a day

        self.assertGreater(LedgerUtils.biweekly_consistency_ratio(l4_cps), 1 / len(l4_cps))  # More than a perfect ledger ratio, as we sum the days
        self.assertGreater(LedgerUtils.biweekly_consistency_penalty(l4_cps), 0.0)
        self.assertLess(LedgerUtils.biweekly_consistency_penalty(l4_cps), 1.0)

        # Now it should fail the biweekly consistency for l5
        self.assertAlmostEqual(LedgerUtils.biweekly_consistency_ratio(l5_cps), 1.0)
        self.assertLess(LedgerUtils.biweekly_consistency_penalty(l5_cps), 1.0)
        self.assertAlmostEqual(LedgerUtils.biweekly_consistency_penalty(l5_cps), 0.0)

    def test_max_drawdown_threshold(self):
        l1 = generate_ledger(0.1, mdd=0.99)  # 1% drawdown
        l1_cps = l1.cps
        l2 = copy.deepcopy(l1)
        l2_cps = l2.cps
        l2_cps[-1].mdd = 0.8  # 20% drawdown only on the most recent checkpoint

        l3 = copy.deepcopy(l1)
        l3_cps = l3.cps
        l3_cps[0].mdd = 0.8  # 20% drawdown only on the first checkpoint

        l4 = generate_ledger(0.1, mdd=0.8)  # 20% drawdown
        l4_cps = l4.cps

        self.assertEqual(LedgerUtils.max_drawdown_threshold_penalty(l1_cps), 1.0)
        self.assertEqual(LedgerUtils.max_drawdown_threshold_penalty(l2_cps), 0.0)
        self.assertEqual(LedgerUtils.max_drawdown_threshold_penalty(l3_cps), 1.0)
        self.assertEqual(LedgerUtils.max_drawdown_threshold_penalty(l4_cps), 0.0)
