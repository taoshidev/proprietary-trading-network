import copy
from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.utils.ledger_utils import LedgerUtils

from vali_objects.vali_config import ValiConfig
from vali_objects.vali_dataclasses.perf_ledger import TP_ID_PORTFOLIO
from tests.shared_objects.test_utilities import generate_ledger, checkpoint_generator
import random


class TestLedgerUtils(TestBase):
    """
    This class will only test the positions and the consistency metrics associated with positions.
    """

    def setUp(self):
        super().setUp()
        # seeding
        random.seed(0)

        self.DEFAULT_LEDGER = generate_ledger(0.1, mdd=0.99)

    def test_daily_return_log(self):
        """
        should bucket the checkpoint returns by full days
        """
        self.assertEqual(LedgerUtils.daily_return_log([]), [])
        checkpoints = self.DEFAULT_LEDGER[TP_ID_PORTFOLIO].cps

        # One checkpoint shouldn't be enough since full day is required
        self.assertEqual(len(LedgerUtils.daily_return_log([checkpoints[0]])), 0)

        # Two checkpoints with one not having enough accumulation time doesn't count as a full day
        self.assertEqual(len(LedgerUtils.daily_return_log(checkpoints[:2])), 0)

        # Three checkpoints but no two starting on the same day that have full accumulation time doesn't count
        self.assertEqual(len(LedgerUtils.daily_return_log(checkpoints[:3])), 0)

        # Single day should return a value
        # 2 is used in the product since the first full day of checkpoints is the second day here
        num_checkpoints = ValiConfig.DAILY_CHECKPOINTS
        self.assertEqual(len(LedgerUtils.daily_return_log(checkpoints[:num_checkpoints * 2])), 1)
        self.assertEqual(LedgerUtils.daily_return_log(checkpoints[:num_checkpoints * 2])[0], 0)

        self.assertEqual(len(LedgerUtils.daily_return_log(checkpoints)), 89)
        l1 = generate_ledger(0.1, start_time=10, end_time=ValiConfig.TARGET_LEDGER_WINDOW_MS, mdd=0.99)
        l1_cps = l1[TP_ID_PORTFOLIO].cps
        self.assertEqual(len(LedgerUtils.daily_return_log(l1_cps)), 88)

    def test_daily_return(self):
        """
        exponentiate daily return log and convert to percentage
        """
        # Base case
        self.assertEqual(len(LedgerUtils.daily_return_percentage([])), 0)

        # No returns
        l1 = generate_ledger(0.1)
        l1_cps = l1[TP_ID_PORTFOLIO].cps
        self.assertEqual(LedgerUtils.daily_return_percentage(l1_cps)[0], 0)
        # Simple returns >= log returns
        self.assertGreaterEqual(LedgerUtils.daily_return_percentage(l1_cps)[0], LedgerUtils.daily_return_log(l1_cps)[0] * 100)

        # Negative returns
        l1 = generate_ledger(0.1, gain=0.1, loss=-0.2)
        l1_cps = l1[TP_ID_PORTFOLIO].cps
        self.assertLess(LedgerUtils.daily_return_percentage(l1_cps)[0], 0)
        # Simple returns >= log returns
        self.assertGreaterEqual(LedgerUtils.daily_return_percentage(l1_cps)[0], LedgerUtils.daily_return_log(l1_cps)[0] * 100)

        # Positive returns
        l1 = generate_ledger(0.1, gain=0.2, loss=-0.1)
        l1_cps = l1[TP_ID_PORTFOLIO].cps
        self.assertGreater(LedgerUtils.daily_return_percentage(l1_cps)[0], 0)
        # Simple returns >= log returns
        self.assertGreaterEqual(LedgerUtils.daily_return_percentage(l1_cps)[0], LedgerUtils.daily_return_log(l1_cps)[0] * 100)

    # Want to test the individual functions inputs and outputs
    def test_recent_drawdown(self):
        l1 = generate_ledger(0.1, mdd=0.99)
        l1_cps = l1[TP_ID_PORTFOLIO].cps

        self.assertEqual(LedgerUtils.recent_drawdown([]), 1)

        LedgerUtils.recent_drawdown(l1_cps)
        self.assertEqual(LedgerUtils.recent_drawdown(l1_cps), 0.99)

        l2 = generate_ledger(0.1, mdd=0.95)
        l2_cps = l2[TP_ID_PORTFOLIO].cps
        self.assertEqual(LedgerUtils.recent_drawdown(l2_cps), 0.95)

        l3 = generate_ledger(0.1, mdd=0.99)
        l3_cps = l3[TP_ID_PORTFOLIO].cps
        l3_cps[-1].mdd = 0.5
        self.assertEqual(LedgerUtils.recent_drawdown(l3_cps), 0.5)

        l4 = generate_ledger(0.1, mdd=0.99)
        l4_cps = l4[TP_ID_PORTFOLIO].cps
        l4_cps[0].mdd = 0.5
        self.assertEqual(LedgerUtils.recent_drawdown(l4_cps), 0.99)

        for element in [l1_cps, l2_cps, l3_cps, l4_cps]:
            self.assertGreaterEqual(LedgerUtils.recent_drawdown(element), 0)
            self.assertLessEqual(LedgerUtils.recent_drawdown(element), 1)

        # Recent drawdown should work even if there is only one checkpoint
        drawdowns = [0.99, 0.98]
        checkpoints = [checkpoint_generator(mdd=mdd) for mdd in drawdowns]
        self.assertEqual(LedgerUtils.recent_drawdown(checkpoints), 0.98)

    def test_drawdown_percentage(self):
        self.assertAlmostEqual(LedgerUtils.drawdown_percentage(1), 0)
        self.assertAlmostEqual(LedgerUtils.drawdown_percentage(0), 100)
        self.assertAlmostEqual(LedgerUtils.drawdown_percentage(-0.1), 100)
        self.assertAlmostEqual(LedgerUtils.drawdown_percentage(0.99), 1)
        self.assertAlmostEqual(LedgerUtils.drawdown_percentage(0.95), 5)
        self.assertAlmostEqual(LedgerUtils.drawdown_percentage(0.5), 50)

    # Test mdd_lower_augmentation
    def test_mdd_lower_augmentation(self):
        self.assertEqual(LedgerUtils.mdd_lower_augmentation(-1), 0)
        self.assertEqual(LedgerUtils.mdd_lower_augmentation(0), 0)
        self.assertEqual(LedgerUtils.mdd_lower_augmentation(ValiConfig.DRAWDOWN_MINVALUE_PERCENTAGE - 0.01), 0)
        self.assertEqual(LedgerUtils.mdd_lower_augmentation(ValiConfig.DRAWDOWN_MINVALUE_PERCENTAGE + 0.01), 1)

    # Test mdd_upper_augmentation
    def test_mdd_upper_augmentation(self):
        # Test at the upper boundary plus a small increment (should still return 1)
        self.assertEqual(LedgerUtils.mdd_upper_augmentation(ValiConfig.DRAWDOWN_MAXVALUE_PERCENTAGE + 0.01), 0)

        # Test just below the upper boundary (should return 0)
        self.assertGreater(LedgerUtils.mdd_upper_augmentation(ValiConfig.DRAWDOWN_MAXVALUE_PERCENTAGE - 0.01), 0)

        # Test at the midpoint of the penalty region
        midpoint_drawdown = (ValiConfig.DRAWDOWN_MAXVALUE_PERCENTAGE / 2)
        penalty_midpoint = LedgerUtils.mdd_upper_augmentation(midpoint_drawdown)
        self.assertGreater(penalty_midpoint, 0)
        self.assertLess(penalty_midpoint, 100)

        # Test slightly below the max value (should return a value between 0 and 1)
        near_max_drawdown = ValiConfig.DRAWDOWN_MAXVALUE_PERCENTAGE * 0.9
        penalty_near_max = LedgerUtils.mdd_upper_augmentation(near_max_drawdown)
        self.assertGreater(penalty_near_max, 0)
        self.assertLess(penalty_near_max, 1)

        # Test slightly above 0 but within the region (should return a value between 0 and 1)
        near_zero_drawdown = ValiConfig.DRAWDOWN_MAXVALUE_PERCENTAGE * 0.99
        penalty_near_zero = LedgerUtils.mdd_upper_augmentation(near_zero_drawdown)
        self.assertGreater(penalty_near_zero, 0)
        self.assertLess(penalty_near_zero, 1)

    # Test mdd_base_augmentation
    def test_mdd_base_augmentation(self):
        self.assertEqual(LedgerUtils.mdd_base_augmentation(0.5), 2)
        self.assertEqual(LedgerUtils.mdd_base_augmentation(1), 1)

    # Test mdd_augmentation
    def test_mdd_augmentation(self):
        self.assertEqual(LedgerUtils.mdd_augmentation(-1), 0)
        self.assertEqual(LedgerUtils.mdd_augmentation(1.1), 0)
        self.assertEqual(LedgerUtils.mdd_augmentation(0), 0)
        self.assertEqual(LedgerUtils.mdd_augmentation(1.01), 0)
        self.assertEqual(LedgerUtils.mdd_augmentation(0.5), 0)

        self.assertAlmostEqual(LedgerUtils.mdd_augmentation(0.99), 1)  # Assuming drawdown_percentage = 99% and falls within limits
        self.assertAlmostEqual(LedgerUtils.mdd_augmentation(0.98), 0.5)

        # this should be something like 0.995, so subtracting will "increase" drawdown
        lower_percent = ValiConfig.DRAWDOWN_MINVALUE_PERCENTAGE
        lower_limit = 1 - (lower_percent / 100)
        self.assertAlmostEqual(LedgerUtils.mdd_augmentation(lower_limit + 0.01), 0)

        self.assertAlmostEqual(LedgerUtils.mdd_base_augmentation(lower_percent), 1 / lower_percent)
        self.assertAlmostEqual(LedgerUtils.mdd_lower_augmentation(lower_percent + 0.01), 1)
        self.assertAlmostEqual(LedgerUtils.mdd_upper_augmentation(lower_percent + 0.01), 1)

        self.assertEqual(LedgerUtils.mdd_augmentation(lower_limit + 0.01), 0)

        self.assertAlmostEqual(LedgerUtils.mdd_augmentation(1 - (ValiConfig.DRAWDOWN_MAXVALUE_PERCENTAGE / 100)), 0)
        self.assertEqual(LedgerUtils.mdd_augmentation(1 - (ValiConfig.DRAWDOWN_MAXVALUE_PERCENTAGE / 100) - 0.001), 0)

        self.assertAlmostEqual(LedgerUtils.mdd_augmentation(1 - (ValiConfig.DRAWDOWN_MINVALUE_PERCENTAGE / 100) + 0.001), 0)

    # Test max_drawdown_threshold_penalty
    def test_max_drawdown_threshold_penalty(self):
        checkpoints = self.DEFAULT_LEDGER[TP_ID_PORTFOLIO].cps
        self.assertEqual(LedgerUtils.max_drawdown_threshold_penalty([]), 0)
        self.assertEqual(LedgerUtils.max_drawdown_threshold_penalty(checkpoints), 1)

        l1 = copy.deepcopy(self.DEFAULT_LEDGER)
        l1_cps = l1[TP_ID_PORTFOLIO].cps
        l1_cps[-1].mdd = 0.8

        self.assertEqual(LedgerUtils.max_drawdown_threshold_penalty(l1_cps), 0)

    # Test approximate_drawdown
    def test_approximate_drawdown(self):
        checkpoints = self.DEFAULT_LEDGER[TP_ID_PORTFOLIO].cps
        self.assertEqual(LedgerUtils.approximate_drawdown([]), 0)
        self.assertLessEqual(LedgerUtils.approximate_drawdown(checkpoints), 1)

        l1 = generate_ledger(0.1, mdd=0.99)  # 1% drawdown
        l1_cps = l1[TP_ID_PORTFOLIO].cps

        for i in range(0, len(l1_cps) - len(l1_cps)//4):
            l1_cps[i].mdd = (random.random() / 10) + 0.9

        self.assertLess(LedgerUtils.approximate_drawdown(l1_cps), 0.99)
        self.assertGreater(LedgerUtils.approximate_drawdown(l1_cps), 0.8)

        l2 = generate_ledger(0.1, mdd=0.99)  # 1% drawdown
        l2_cps = l2[TP_ID_PORTFOLIO].cps
        l2_cps[-1].mdd = 0.8  # 20% drawdown only on the most recent checkpoint
        self.assertLessEqual(LedgerUtils.approximate_drawdown(l2_cps), 0.99)
        self.assertGreater(LedgerUtils.approximate_drawdown(l2_cps), 0.8)

    # Test effective_drawdown
    def test_effective_drawdown(self):
        self.assertEqual(LedgerUtils.effective_drawdown(0, 0.5), 0)
        self.assertEqual(LedgerUtils.effective_drawdown(0.5, 0), 0)
        self.assertEqual(LedgerUtils.effective_drawdown(0.5, 0.5), 0.5)
        self.assertEqual(LedgerUtils.effective_drawdown(0.9, 0.95), 0.9)

    # Test mean_drawdown
    def test_mean_drawdown(self):
        drawdowns = [0.98]
        checkpoints = [checkpoint_generator(mdd=x) for x in drawdowns]

        self.assertEqual(LedgerUtils.mean_drawdown([]), 0)
        # With one checkpoint, mean drawdown should be mdd
        self.assertEqual(LedgerUtils.mean_drawdown(checkpoints), 0.98)

        drawdowns = [0.98, 1.0]
        checkpoints = [checkpoint_generator(mdd=mdd) for mdd in drawdowns]

        # Should be the average in the general case
        self.assertEqual(LedgerUtils.mean_drawdown(checkpoints), 0.99)

        drawdowns = [1.1, 1.3, 0.99]
        checkpoints = [checkpoint_generator(mdd=mdd) for mdd in drawdowns]

        # Should be set to 1 if average is over 1
        self.assertEqual(LedgerUtils.mean_drawdown(checkpoints), 1.0)

        drawdowns = [-0.1, -0.3, -0.9]
        checkpoints = [checkpoint_generator(mdd=mdd) for mdd in drawdowns]

        # Should be set 0 if drawdowns are somehow negative
        self.assertEqual(LedgerUtils.mean_drawdown(checkpoints), 0)

        self.assertEqual(LedgerUtils.mean_drawdown(self.DEFAULT_LEDGER[TP_ID_PORTFOLIO].cps), 0.99)

    # Test risk_normalization
    def test_risk_normalization(self):
        checkpoints = self.DEFAULT_LEDGER[TP_ID_PORTFOLIO].cps
        self.assertEqual(LedgerUtils.risk_normalization([]), 0)
        self.assertLessEqual(LedgerUtils.risk_normalization(checkpoints), 1)

