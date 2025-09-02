import copy
import math
import random
from datetime import date as date_type
from datetime import datetime, timezone

from tests.shared_objects.test_utilities import (
    checkpoint_generator,
    generate_ledger,
    ledger_generator,
)
from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.utils.ledger_utils import LedgerUtils
from vali_objects.vali_config import ValiConfig
from vali_objects.vali_dataclasses.perf_ledger import (
    TP_ID_PORTFOLIO,
    PerfCheckpoint,
    PerfLedger,
)


class TestLedgerUtils(TestBase):
    """
    This class will only test the positions and the consistency metrics associated with positions.
    """

    def setUp(self):
        super().setUp()
        # seeding
        random.seed(0)

        self.DEFAULT_LEDGER = generate_ledger(0.1, mdd=0.99)[TP_ID_PORTFOLIO]

    def test_daily_return_log(self):
        """
        should bucket the checkpoint returns by full days
        """
        # Test with empty ledger
        empty_ledger = PerfLedger()
        self.assertEqual(LedgerUtils.daily_return_log(empty_ledger), [])

        ledger = self.DEFAULT_LEDGER
        checkpoints = ledger.cps

        # One checkpoint shouldn't be enough since full day is required
        single_cp_ledger = ledger_generator(checkpoints=[checkpoints[0]])
        self.assertEqual(len(LedgerUtils.daily_return_log(single_cp_ledger)), 0)

        # Two checkpoints with one not having enough accumulation time doesn't count as a full day
        two_cp_ledger = ledger_generator(checkpoints=checkpoints[:2])
        self.assertEqual(len(LedgerUtils.daily_return_log(two_cp_ledger)), 0)

        # Test: Only full cells count towards the daily returns
        cp1 = copy.deepcopy(checkpoints[0])
        cp2 = copy.deepcopy(checkpoints[1])

        # Set invalid accum_ms to prevent a full day - only full cells should count
        cp1.accum_ms = ValiConfig.TARGET_CHECKPOINT_DURATION_MS  # Full duration
        cp2.accum_ms = ValiConfig.TARGET_CHECKPOINT_DURATION_MS // 2  # Half duration (not full)

        # Set timestamps to be on the same day (UTC)
        base_ts = 1672531200000  # 2023-01-01 00:00:00 UTC
        cp1.last_update_ms = base_ts + 43200000  # 12:00:00 (noon)
        cp2.last_update_ms = base_ts + 64800000  # 18:00:00 (evening)

        partial_day_ledger = ledger_generator(checkpoints=[cp1, cp2])
        self.assertEqual(len(LedgerUtils.daily_return_log(partial_day_ledger)), 0,
                        "Day with not enough full cells should return 0 days")

        # We need to mock checkpoints that will pass the full_cell check
        # and have exact timestamps that will be categorized by day

        # Create a ledger with 4 checkpoints (assuming DAILY_CHECKPOINTS = 2)
        # Day 1: 2 checkpoints at 00:00 and 12:00
        # Day 2: 2 checkpoints at 00:00 and 12:00
        day1_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
        day2_date = datetime(2023, 1, 2, tzinfo=timezone.utc)

        checkpoint_duration = ValiConfig.TARGET_CHECKPOINT_DURATION_MS

        # Make sure that last_update_ms - accum_ms will be on the correct day
        day1_midnight_ms = int(day1_date.timestamp() * 1000) + checkpoint_duration  # ensures start time is on day1
        day1_noon_ms = day1_midnight_ms + (12 * 60 * 60 * 1000)  # 12 hours later, still on day1
        day2_midnight_ms = int(day2_date.timestamp() * 1000) + checkpoint_duration  # ensures start time is on day2
        day2_noon_ms = day2_midnight_ms + (12 * 60 * 60 * 1000)  # 12 hours later, still on day2

        # Create explicit checkpoints with controlled timestamps
        mock_checkpoints = [
            # Day 1 checkpoints
            PerfCheckpoint(
                last_update_ms=day1_midnight_ms,
                accum_ms=checkpoint_duration,
                open_ms=day1_midnight_ms - checkpoint_duration,
                gain=0.1,
                loss=-0.05,
                prev_portfolio_ret=1.0,
                n_updates=1,
                mdd=0.99,
            ),
            PerfCheckpoint(
                last_update_ms=day1_noon_ms,
                accum_ms=checkpoint_duration,
                open_ms=day1_noon_ms - checkpoint_duration,
                gain=0.1,
                loss=-0.05,
                prev_portfolio_ret=1.0,
                n_updates=1,
                mdd=0.99,
            ),
            # Day 2 checkpoints
            PerfCheckpoint(
                last_update_ms=day2_midnight_ms,
                accum_ms=checkpoint_duration,
                open_ms=day2_midnight_ms - checkpoint_duration,
                gain=0.2,
                loss=-0.1,
                prev_portfolio_ret=1.0,
                n_updates=1,
                mdd=0.99,
            ),
            PerfCheckpoint(
                last_update_ms=day2_noon_ms,
                accum_ms=checkpoint_duration,
                open_ms=day2_noon_ms - checkpoint_duration,
                gain=0.2,
                loss=-0.1,
                prev_portfolio_ret=1.0,
                n_updates=1,
                mdd=0.99,
            ),
        ]

        # Filter only day 1 checkpoints
        day1_checkpoints = mock_checkpoints[:2]
        day1_ledger = ledger_generator(checkpoints=day1_checkpoints)
        day1_returns = LedgerUtils.daily_return_log(day1_ledger)
        self.assertEqual(len(day1_returns), 1,
                        "A full day of checkpoints should return 1 day")

        # Filter only day 2 checkpoints
        day2_checkpoints = mock_checkpoints[2:4]
        day2_ledger = ledger_generator(checkpoints=day2_checkpoints)
        day2_returns = LedgerUtils.daily_return_log(day2_ledger)
        self.assertEqual(len(day2_returns), 1,
                        "A full day of checkpoints should return 1 day")

        # Test with both days
        two_day_ledger = ledger_generator(checkpoints=mock_checkpoints)
        daily_returns = LedgerUtils.daily_return_log(two_day_ledger)
        self.assertEqual(len(daily_returns), 2,
                        "Two complete days should return 2 days of returns")

        # Check date-based returns
        date_returns = LedgerUtils.daily_return_log_by_date(two_day_ledger)
        self.assertEqual(len(date_returns), 2,
                        "Two complete days should return 2 dates with returns")

        dates = list(date_returns.keys())
        self.assertEqual(dates[0], day1_date.date(), "First date should be 2023-01-01")
        self.assertEqual(dates[1], day2_date.date(), "Second date should be 2023-01-02")

        # Verify return values
        day1_expected_return = 2 * (0.1 - 0.05)  # 2 checkpoints with 0.1 gain, -0.05 loss each
        day2_expected_return = 2 * (0.2 - 0.1)   # 2 checkpoints with 0.2 gain, -0.1 loss each

        self.assertAlmostEqual(daily_returns[0], day1_expected_return,
                             msg="Day 1 return should match the expected value")
        self.assertAlmostEqual(daily_returns[1], day2_expected_return,
                             msg="Day 2 return should match the expected value")

        # Test the problematic day-boundary case by creating checkpoints spanning from day 1 to day 2
        # This checkpoint starts at end of day 1 and ends at beginning of day 2
        # Create a day1 checkpoint that will match the exact format expected
        # Looking at the _daily_return_log_by_date code:
        # 1. It calculates start_time = (cp.last_update_ms - cp.accum_ms)
        # 2. It groups by dates from that start_time
        # This checkpoint must have:
        # - Full accum_ms to pass the full_cell check
        # - correct start time to be on day1

        # Add exactly enough checkpoints to form a complete day
        num_cp_needed = int(ValiConfig.DAILY_CHECKPOINTS)
        day1_checkpoints = []

        # Create multiple checkpoints for day1 that all have the proper characteristics
        for i in range(num_cp_needed):
            hour = 6 + i * (24 // num_cp_needed)  # Space them out across the day
            checkpoint_start = int(day1_date.timestamp() * 1000) + (hour * 3600 * 1000)
            day1_checkpoints.append(
                PerfCheckpoint(
                    last_update_ms=checkpoint_start + checkpoint_duration,
                    accum_ms=checkpoint_duration,
                    open_ms=checkpoint_start,
                    gain=0.1,
                    loss=-0.05,
                    prev_portfolio_ret=1.0,
                    n_updates=1,
                    mdd=0.99,
                ),
            )

        full_day1_ledger = ledger_generator(checkpoints=day1_checkpoints)
        date_returns = LedgerUtils.daily_return_log_by_date(full_day1_ledger)

        # Now we should have a full day's worth of data
        self.assertIn(day1_date.date(), date_returns.keys(),
                    "Day with enough checkpoints should be included")
        self.assertNotIn(day2_date.date(), date_returns.keys(),
                      "No checkpoints were on day 2")

        # Verify our understanding of the ledger_utils.py algorithm:
        # 1. It groups checkpoints by date (the date of the start time)
        # 2. It only includes days with EXACTLY n_checkpoints_per_day checkpoints

        # Test with one checkpoint less than required - should NOT have any daily returns
        insufficient_day_cps = day1_checkpoints[:-1]  # One less than required
        insufficient_ledger = ledger_generator(checkpoints=insufficient_day_cps)
        self.assertEqual(len(LedgerUtils.daily_return_log(insufficient_ledger)), 0,
                       "Day with less than required checkpoints should return 0 days")

        # Verify we have the right number of checkpoints per day for this test
        self.assertEqual(len(day1_checkpoints), int(ValiConfig.DAILY_CHECKPOINTS),
                       "Test setup should have the exact number of required checkpoints")

        # The key observation about the ledger_utils.py implementation:
        # It requires EXACTLY the number of checkpoints per day - not more, not less

        # Add one more checkpoint - this actually breaks the day count because
        # the implementation requires EXACTLY the right number
        hour = 18  # 6 PM
        checkpoint_start = int(day1_date.timestamp() * 1000) + (hour * 3600 * 1000)
        extra_cp = PerfCheckpoint(
            last_update_ms=checkpoint_start + checkpoint_duration,
            accum_ms=checkpoint_duration,
            open_ms=checkpoint_start,
            gain=0.1,
            loss=-0.05,
            prev_portfolio_ret=1.0,
            n_updates=1,
            mdd=0.99,
        )

        extra_day_cps = day1_checkpoints + [extra_cp]
        extra_ledger = ledger_generator(checkpoints=extra_day_cps)

        # According to the current implementation, we should have 0 daily returns
        # because the day has MORE than n_checkpoints_per_day checkpoints
        self.assertEqual(len(LedgerUtils.daily_return_log(extra_ledger)), 0,
                       "Day with more than required checkpoints should return 0 days")

    def test_daily_return(self):
        """
        exponentiate daily return log and convert to percentage
        """
        # Base case
        empty_ledger = PerfLedger()
        self.assertEqual(len(LedgerUtils.daily_returns(empty_ledger)), 0)

        # No returns
        l1 = generate_ledger(0.1)[TP_ID_PORTFOLIO]
        l1_ledger = l1
        self.assertEqual(LedgerUtils.daily_returns(l1_ledger)[0], 0)
        # Simple returns >= log returns
        self.assertGreaterEqual(LedgerUtils.daily_returns(l1_ledger)[0], LedgerUtils.daily_return_log(l1_ledger)[0] * 100)

        # Negative returns
        l1 = generate_ledger(0.1, gain=0.1, loss=-0.2)[TP_ID_PORTFOLIO]
        l1_ledger = l1
        self.assertLess(LedgerUtils.daily_returns(l1_ledger)[0], 0)
        # Simple returns >= log returns
        self.assertGreaterEqual(LedgerUtils.daily_returns(l1_ledger)[0], LedgerUtils.daily_return_log(l1_ledger)[0] * 100)

        # Positive returns
        l1 = generate_ledger(0.1, gain=0.2, loss=-0.1)[TP_ID_PORTFOLIO]
        l1_ledger = l1
        self.assertGreater(LedgerUtils.daily_returns(l1_ledger)[0], 0)
        # Simple returns >= log returns
        self.assertGreaterEqual(LedgerUtils.daily_returns(l1_ledger)[0], LedgerUtils.daily_return_log(l1_ledger)[0] * 100)

    # Want to test the individual functions inputs and outputs
    def test_instantaneous_max_drawdown(self):
        """Test instantaneous_max_drawdown function"""
        # Empty ledger test
        empty_ledger = PerfLedger()
        self.assertEqual(LedgerUtils.instantaneous_max_drawdown(empty_ledger), 0.0)

        # Valid ledger tests
        l1 = generate_ledger(0.1, mdd=0.99)[TP_ID_PORTFOLIO]
        self.assertEqual(LedgerUtils.instantaneous_max_drawdown(l1), 0.99)

        l2 = generate_ledger(0.1, mdd=0.95)[TP_ID_PORTFOLIO]
        self.assertEqual(LedgerUtils.instantaneous_max_drawdown(l2), 0.95)

        # Test with varying drawdowns - should return the minimum (worst) drawdown
        l3 = generate_ledger(0.1, mdd=0.99)[TP_ID_PORTFOLIO]
        l3_cps = l3.cps
        l3_cps[-1].mdd = 0.5  # Worse drawdown
        self.assertEqual(LedgerUtils.instantaneous_max_drawdown(l3), 0.5)

        l4 = generate_ledger(0.1, mdd=0.99)[TP_ID_PORTFOLIO]
        l4_cps = l4.cps
        l4_cps[0].mdd = 0.5  # Worse drawdown at start
        self.assertEqual(LedgerUtils.instantaneous_max_drawdown(l4), 0.5)

        # Test bounds - should always be between 0 and 1
        for element in [l1, l2, l3, l4]:
            result = LedgerUtils.instantaneous_max_drawdown(element)
            self.assertGreaterEqual(result, 0)
            self.assertLessEqual(result, 1)

        # Test with a minimal ledger containing just a few checkpoints
        drawdowns = [0.99, 0.98, 0.97]
        checkpoints = [checkpoint_generator(mdd=mdd) for mdd in drawdowns]
        minimal_ledger = ledger_generator(checkpoints=checkpoints)
        self.assertEqual(LedgerUtils.instantaneous_max_drawdown(minimal_ledger), 0.97)

        # Test with single checkpoint
        single_checkpoint = [checkpoint_generator(mdd=0.85)]
        single_ledger = ledger_generator(checkpoints=single_checkpoint)
        self.assertEqual(LedgerUtils.instantaneous_max_drawdown(single_ledger), 0.85)

        # Test with drawdown values that need clipping
        extreme_drawdowns = [1.1, -0.1, 0.3]  # Values outside [0, 1]
        extreme_checkpoints = [checkpoint_generator(mdd=mdd) for mdd in extreme_drawdowns]
        extreme_ledger = ledger_generator(checkpoints=extreme_checkpoints)
        result = LedgerUtils.instantaneous_max_drawdown(extreme_ledger)
        self.assertGreaterEqual(result, 0)
        self.assertLessEqual(result, 1)

    def test_daily_max_drawdown(self):
        """Test daily_max_drawdown function"""
        # Empty ledger test
        empty_ledger = PerfLedger()
        self.assertEqual(LedgerUtils.daily_max_drawdown(empty_ledger), 0.0)

        # Test with ledger that has no complete daily returns
        # Single checkpoint won't create complete daily returns
        single_checkpoint = [checkpoint_generator(gain=0.05, loss=-0.03)]
        single_ledger = ledger_generator(checkpoints=single_checkpoint)
        single_result = LedgerUtils.daily_max_drawdown(single_ledger)
        self.assertEqual(single_result, 0.0)  # No complete daily returns = 0

        # Test with ledger that would create complete daily returns
        # We need to create a ledger that will pass the daily_return_log requirements
        # Create checkpoints that will form complete days
        day1_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
        day2_date = datetime(2023, 1, 2, tzinfo=timezone.utc)
        day3_date = datetime(2023, 1, 3, tzinfo=timezone.utc)
        
        checkpoint_duration = ValiConfig.TARGET_CHECKPOINT_DURATION_MS
        num_cp_per_day = int(ValiConfig.DAILY_CHECKPOINTS)
        
        # Create checkpoints for 3 complete days with specific return patterns
        def create_day_checkpoints(date: datetime, gain: float, loss: float) -> list:
            checkpoints = []
            for i in range(num_cp_per_day):
                hour = 6 + i * (12 // num_cp_per_day)  # Space them out
                checkpoint_start = int(date.timestamp() * 1000) + (hour * 3600 * 1000)
                checkpoints.append(
                    PerfCheckpoint(
                        last_update_ms=checkpoint_start + checkpoint_duration,
                        accum_ms=checkpoint_duration,
                        open_ms=checkpoint_start,
                        gain=gain,
                        loss=loss,
                        prev_portfolio_ret=1.0,
                        n_updates=1,
                        mdd=0.99
                    )
                )
            return checkpoints
        
        # Test with positive returns only (no drawdown expected)
        positive_checkpoints = (
            create_day_checkpoints(day1_date, 0.1, 0.0) +    # +0.1 per checkpoint
            create_day_checkpoints(day2_date, 0.05, 0.0) +   # +0.05 per checkpoint
            create_day_checkpoints(day3_date, 0.08, 0.0)     # +0.08 per checkpoint
        )
        positive_ledger = ledger_generator(checkpoints=positive_checkpoints)
        positive_result = LedgerUtils.daily_max_drawdown(positive_ledger)
        
        # With only positive returns, drawdown should be 1.0 (no drawdown)
        self.assertEqual(positive_result, 1.0)
        
        # Test with pattern: up, down, up (recovery scenario)
        recovery_checkpoints = (
            create_day_checkpoints(day1_date, 0.1, 0.0) +    # Up day: +0.1 per checkpoint
            create_day_checkpoints(day2_date, 0.0, -0.15) +  # Down day: -0.15 per checkpoint
            create_day_checkpoints(day3_date, 0.12, 0.0)     # Recovery day: +0.12 per checkpoint
        )
        recovery_ledger = ledger_generator(checkpoints=recovery_checkpoints)
        recovery_result = LedgerUtils.daily_max_drawdown(recovery_ledger)
        
        # Should have some drawdown (less than 1.0) due to the down day
        self.assertGreaterEqual(recovery_result, 0)
        self.assertLess(recovery_result, 1.0)
        
        # Test with consistently negative returns (significant drawdown expected)
        negative_checkpoints = (
            create_day_checkpoints(day1_date, 0.0, -0.1) +   # Down day: -0.1 per checkpoint
            create_day_checkpoints(day2_date, 0.0, -0.05) +  # Down day: -0.05 per checkpoint
            create_day_checkpoints(day3_date, 0.0, -0.08)    # Down day: -0.08 per checkpoint
        )
        negative_ledger = ledger_generator(checkpoints=negative_checkpoints)
        negative_result = LedgerUtils.daily_max_drawdown(negative_ledger)
        
        # Should have significant drawdown (much less than 1.0)
        self.assertGreaterEqual(negative_result, 0)
        self.assertLess(negative_result, 1.0)
        
        # Negative returns should result in lower drawdown values than positive
        self.assertLess(negative_result, positive_result)
        
        # Test with zero returns (no change in value)
        zero_checkpoints = (
            create_day_checkpoints(day1_date, 0.0, 0.0) +    # Flat day
            create_day_checkpoints(day2_date, 0.0, 0.0)      # Flat day
        )
        zero_ledger = ledger_generator(checkpoints=zero_checkpoints)
        zero_result = LedgerUtils.daily_max_drawdown(zero_ledger)
        
        # With zero returns, cumulative values stay at 1.0, no drawdown
        self.assertEqual(zero_result, 1.0)
        
        # Test mathematical correctness with known values
        # Day 1: +0.1 per checkpoint * 2 checkpoints = 0.2 total return -> cumulative value = exp(0.2) ≈ 1.221
        # Day 2: -0.2 per checkpoint * 2 checkpoints = -0.4 total return -> cumulative value = exp(0.2-0.4) = exp(-0.2) ≈ 0.819
        # Day 3: +0.15 per checkpoint * 2 checkpoints = 0.3 total return -> cumulative value = exp(0.2-0.4+0.3) = exp(0.1) ≈ 1.105
        # Running max: [1.221, 1.221, 1.221]
        # Drawdown on day 2: (0.819 - 1.221) / 1.221 ≈ -0.33
        # Min drawdown_numeric: 1 + (-0.33) ≈ 0.67
        
        math_test_checkpoints = (
            create_day_checkpoints(day1_date, 0.1, 0.0) +    # Net +0.1 per checkpoint
            create_day_checkpoints(day2_date, 0.0, -0.2) +   # Net -0.2 per checkpoint  
            create_day_checkpoints(day3_date, 0.15, 0.0)     # Net +0.15 per checkpoint
        )
        math_test_ledger = ledger_generator(checkpoints=math_test_checkpoints)
        math_result = LedgerUtils.daily_max_drawdown(math_test_ledger)
        
        # The result should be around 0.67 based on the calculation above
        self.assertGreaterEqual(math_result, 0.65)
        self.assertLessEqual(math_result, 0.7)

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
        # self.assertAlmostEqual(LedgerUtils.mdd_augmentation(lower_limit + 0.01), 0)

        self.assertAlmostEqual(LedgerUtils.mdd_base_augmentation(lower_percent), 1 / lower_percent)
        self.assertAlmostEqual(LedgerUtils.mdd_lower_augmentation(lower_percent + 0.01), 1)
        self.assertAlmostEqual(LedgerUtils.mdd_upper_augmentation(lower_percent + 0.01), 1)

        self.assertEqual(LedgerUtils.mdd_augmentation(lower_limit + 0.01), 0)

        # self.assertAlmostEqual(LedgerUtils.mdd_augmentation(1 - (ValiConfig.DRAWDOWN_MAXVALUE_PERCENTAGE / 100)), 0)
        self.assertEqual(LedgerUtils.mdd_augmentation(1 - (ValiConfig.DRAWDOWN_MAXVALUE_PERCENTAGE / 100) - 0.001), 0)

        # self.assertAlmostEqual(LedgerUtils.mdd_augmentation(1 - (ValiConfig.DRAWDOWN_MINVALUE_PERCENTAGE / 100) + 0.001), 0)

    # Test risk_normalization
    def test_risk_normalization(self):
        # Test with empty list
        self.assertEqual(LedgerUtils.risk_normalization(PerfLedger()), 0)

        # Test with default ledger
        ledger = self.DEFAULT_LEDGER
        self.assertLessEqual(LedgerUtils.risk_normalization(ledger), 1)

        # Test with empty ledger
        empty_ledger = PerfLedger()
        self.assertEqual(LedgerUtils.risk_normalization(empty_ledger), 0)

    def test_daily_return_log_by_date(self):
        """Test the daily return log by date function"""
        # Test with empty ledger
        empty_ledger = PerfLedger()
        self.assertEqual(LedgerUtils.daily_return_log_by_date(empty_ledger), {})

        # Test with ledger containing checkpoints
        ledger = self.DEFAULT_LEDGER
        result = LedgerUtils.daily_return_log_by_date(ledger)

        # Should return a dictionary mapping dates to returns
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), len(LedgerUtils.daily_return_log(ledger)))

        # Each key should be a date
        for date in result.keys():
            self.assertIsInstance(date, date_type)

        # Values should be floats (returns)
        for value in result.values():
            self.assertIsInstance(value, float)

    def test_daily_returns_by_date(self):
        """Test the daily returns by date function"""
        # Test with empty ledger
        empty_ledger = PerfLedger()
        self.assertEqual(LedgerUtils.daily_returns_by_date(empty_ledger), {})

        # Test with ledger containing checkpoints
        ledger = self.DEFAULT_LEDGER
        log_results = LedgerUtils.daily_return_log_by_date(ledger)
        percentage_results = LedgerUtils.daily_returns_by_date(ledger)

        # Should have same dates as keys
        self.assertEqual(set(log_results.keys()), set(percentage_results.keys()))

        # Values should be percentages (exp(log_return) - 1) * 100
        for date, log_return in log_results.items():
            expected_percentage = (math.exp(log_return) - 1) * 100
            self.assertAlmostEqual(percentage_results[date], expected_percentage)

    def test_daily_returns_by_date_json(self):
        """Test the JSON-compatible daily returns by date function"""
        # Test with empty ledger
        empty_ledger = PerfLedger()
        self.assertEqual(LedgerUtils.daily_returns_by_date_json(empty_ledger), {})

        # Test with ledger containing checkpoints
        ledger = self.DEFAULT_LEDGER
        regular_results = LedgerUtils.daily_returns_by_date(ledger)
        json_results = LedgerUtils.daily_returns_by_date_json(ledger)

        # Should have same number of entries
        self.assertEqual(len(regular_results), len(json_results))

        # Keys in json_results should be strings in ISO format (YYYY-MM-DD)
        for key in json_results.keys():
            self.assertIsInstance(key, str)
            # Verify it's in ISO format by parsing it back to a date
            parsed_date = date_type.fromisoformat(key)
            self.assertIsInstance(parsed_date, date_type)

        # Values should match between regular and JSON formats
        for date, value in regular_results.items():
            date_str = date.isoformat()
            self.assertIn(date_str, json_results)
            self.assertEqual(json_results[date_str], value)

        # Test JSON serializability
        import json
        try:
            json_string = json.dumps(json_results)
        # Parsing back should give the same data
            parsed_json = json.loads(json_string)
            self.assertEqual(parsed_json, json_results)
        except TypeError:
            self.fail("daily_returns_by_date_json results should be JSON serializable")
    
    def test_is_valid_trading_day_forex_saturday_exclusion(self):
        """Test that forex trade pairs correctly exclude Saturdays as invalid trading days"""
        # Test data: asset_id, expected_saturday_result, expected_monday_result
        test_cases = [
            ("EURUSD", False, True),    # Forex - Saturday closed, Monday open
            ("GBPUSD", False, True),    # Forex - Saturday closed, Monday open
            ("USDJPY", False, True),    # Forex - Saturday closed, Monday open
            ("BTCUSD", True, True),     # Crypto - Always open
            (TP_ID_PORTFOLIO, True, True),  # Portfolio - Always valid
        ]
        
        saturday_date = date_type(2023, 1, 7)  # Saturday
        monday_date = date_type(2023, 1, 9)    # Monday
        
        for asset_id, expected_saturday, expected_monday in test_cases:
            ledger = generate_ledger(0.1)[TP_ID_PORTFOLIO]
            ledger.tp_id = asset_id
            
            saturday_result = LedgerUtils.is_valid_trading_day(ledger, saturday_date)
            monday_result = LedgerUtils.is_valid_trading_day(ledger, monday_date)
            
            self.assertEqual(
                saturday_result, expected_saturday,
                f"{asset_id} Saturday result should be {expected_saturday}"
            )
            self.assertEqual(
                monday_result, expected_monday,
                f"{asset_id} Monday result should be {expected_monday}"
            )
    
    def test_daily_return_log_by_date_forex_saturday_exclusion(self):
        """Test that daily_return_log_by_date excludes Saturdays for forex pairs"""
        # Create ledgers spanning a weekend using generate_ledger
        friday_start = int(datetime(2023, 1, 6, tzinfo=timezone.utc).timestamp() * 1000)  # Friday
        monday_end = int(datetime(2023, 1, 9, 23, 59, 59, tzinfo=timezone.utc).timestamp() * 1000)  # Monday
        
        # Create forex and crypto ledgers with same time range
        forex_ledger = generate_ledger(0.1, start_time=friday_start, end_time=monday_end)[TP_ID_PORTFOLIO]
        forex_ledger.tp_id = "EURUSD"
        
        crypto_ledger = generate_ledger(0.1, start_time=friday_start, end_time=monday_end)[TP_ID_PORTFOLIO]
        crypto_ledger.tp_id = "BTCUSD"
        
        # Get daily returns for both ledgers
        forex_daily_returns = LedgerUtils.daily_return_log_by_date(forex_ledger)
        crypto_daily_returns = LedgerUtils.daily_return_log_by_date(crypto_ledger)
        
        # Crypto should have more days than forex (Saturday should be excluded for forex)
        self.assertLess(len(forex_daily_returns), len(crypto_daily_returns), 
                       "Forex should have fewer trading days than crypto due to Saturday exclusion")
        
        # Check that Saturday (2023-01-07) is missing from forex but present in crypto
        saturday_date_obj = date_type(2023, 1, 7)
        self.assertNotIn(saturday_date_obj, forex_daily_returns, "Forex should NOT have Saturday returns")
        self.assertIn(saturday_date_obj, crypto_daily_returns, "Crypto should have Saturday returns")
    
    def test_is_valid_trading_day_error_handling(self):
        """Test error handling for is_valid_trading_day function"""
        # Test with None ledger
        self.assertFalse(LedgerUtils.is_valid_trading_day(None, date_type(2023, 1, 1)))
        
        # Test with None date
        ledger = generate_ledger(0.1)[TP_ID_PORTFOLIO]
        ledger.tp_id = "EURUSD"
        self.assertFalse(LedgerUtils.is_valid_trading_day(ledger, None))
        
        # Test with invalid date type
        self.assertFalse(LedgerUtils.is_valid_trading_day(ledger, "2023-01-01"))
        self.assertFalse(LedgerUtils.is_valid_trading_day(ledger, 20230101))
        
        # Test with invalid asset_id (should return False due to None trade_pair)
        invalid_ledger = generate_ledger(0.1)[TP_ID_PORTFOLIO]
        invalid_ledger.tp_id = "INVALID_PAIR"
        self.assertFalse(LedgerUtils.is_valid_trading_day(invalid_ledger, date_type(2023, 1, 1)))
        
        # Test portfolio ledger should always return True (except for error cases)
        portfolio_ledger = generate_ledger(0.1)[TP_ID_PORTFOLIO]
        portfolio_ledger.tp_id = TP_ID_PORTFOLIO
        self.assertTrue(LedgerUtils.is_valid_trading_day(portfolio_ledger, date_type(2023, 1, 7)))  # Saturday

    def test_circuit_subnet_aggregation_parity(self):
        """
        Test that subnet's daily_return_log matches circuit's expected aggregation logic.
        This test verifies the fix for the circuit vs subnet metrics discrepancy.
        """
        # Create checkpoints with mixed complete/incomplete days to test filtering behavior
        checkpoint_duration = ValiConfig.TARGET_CHECKPOINT_DURATION_MS
        daily_checkpoints = ValiConfig.DAILY_CHECKPOINTS
        
        # Day 1: Complete day (exactly DAILY_CHECKPOINTS checkpoints)
        day1_base = datetime(2023, 1, 1, tzinfo=timezone.utc)
        day1_start_ms = int(day1_base.timestamp() * 1000)
        
        # Day 2: Incomplete day (only 1 checkpoint)  
        day2_base = datetime(2023, 1, 2, tzinfo=timezone.utc)
        day2_start_ms = int(day2_base.timestamp() * 1000)
        
        # Day 3: Complete day (exactly DAILY_CHECKPOINTS checkpoints)
        day3_base = datetime(2023, 1, 3, tzinfo=timezone.utc)
        day3_start_ms = int(day3_base.timestamp() * 1000)
        
        checkpoints = [
            # Day 1: 2 complete checkpoints (should be included)
            PerfCheckpoint(
                last_update_ms=day1_start_ms + checkpoint_duration,
                accum_ms=checkpoint_duration,
                gain=0.1, loss=-0.05,
                prev_portfolio_ret=1.0, n_updates=1, mdd=0.99
            ),
            PerfCheckpoint(
                last_update_ms=day1_start_ms + (2 * checkpoint_duration),  
                accum_ms=checkpoint_duration,
                gain=0.05, loss=-0.02,
                prev_portfolio_ret=1.05, n_updates=1, mdd=0.99
            ),
            # Day 2: 1 incomplete checkpoint (should be excluded by current logic)
            PerfCheckpoint(
                last_update_ms=day2_start_ms + checkpoint_duration,
                accum_ms=checkpoint_duration,
                gain=0.2, loss=-0.1,
                prev_portfolio_ret=1.08, n_updates=1, mdd=0.99  
            ),
            # Day 3: 2 complete checkpoints (should be included)
            PerfCheckpoint(
                last_update_ms=day3_start_ms + checkpoint_duration,
                accum_ms=checkpoint_duration,
                gain=0.03, loss=-0.01,
                prev_portfolio_ret=1.18, n_updates=1, mdd=0.99
            ),
            PerfCheckpoint(
                last_update_ms=day3_start_ms + (2 * checkpoint_duration),
                accum_ms=checkpoint_duration, 
                gain=0.02, loss=-0.04,
                prev_portfolio_ret=1.2, n_updates=1, mdd=0.99
            ),
        ]
        
        ledger = ledger_generator(checkpoints=checkpoints)
        ledger.tp_id = TP_ID_PORTFOLIO  # Portfolio ledger
        
        # Get subnet's daily returns
        daily_returns = LedgerUtils.daily_return_log(ledger)
        
        # Verify current behavior: only complete days are included
        expected_complete_days = 2  # Day 1 and Day 3 have exactly 2 checkpoints each
        self.assertEqual(len(daily_returns), expected_complete_days, 
                        "Subnet should only include days with exactly DAILY_CHECKPOINTS checkpoints")
        
        # Verify the actual returns match expectations  
        day1_return = (0.1 - 0.05) + (0.05 - 0.02)  # Sum of both day 1 checkpoints
        day3_return = (0.03 - 0.01) + (0.02 - 0.04)  # Sum of both day 3 checkpoints
        
        expected_returns = [day1_return, day3_return]
        self.assertEqual(len(daily_returns), len(expected_returns))
        
        for i, expected_return in enumerate(expected_returns):
            self.assertAlmostEqual(daily_returns[i], expected_return, places=6,
                                 msg=f"Daily return {i} should match expected calculation")
        
        # Document the behavior: This test verifies that the subnet correctly filters
        # to only include "complete" trading days with exactly DAILY_CHECKPOINTS checkpoints.
        # The circuit was updated to match this behavior to resolve metrics discrepancies.
        print(f"Subnet aggregation test: {len(checkpoints)} checkpoints -> {len(daily_returns)} daily returns")
        print(f"Complete days only (DAILY_CHECKPOINTS={daily_checkpoints}): {expected_complete_days}")
        print(f"Total return: {sum(daily_returns):.6f}")

    def test_proof_of_portfolio_metrics_parity(self):
        """
        Integration test that verifies proof-of-portfolio circuit produces the same metrics 
        as the subnet's calculation for identical data.
        """
        # Skip test if proof_of_portfolio is not available
        try:
            from proof_of_portfolio.proof_generator import generate_proof
        except ImportError:
            self.skipTest("proof_of_portfolio package not available")
        
        # Create test data with known metrics
        checkpoint_duration = ValiConfig.TARGET_CHECKPOINT_DURATION_MS
        daily_checkpoints = ValiConfig.DAILY_CHECKPOINTS
        
        # Create multiple complete days with varying returns
        base_time = datetime(2023, 1, 1, tzinfo=timezone.utc)
        
        test_checkpoints = []
        expected_daily_returns = []
        
        # Generate 5 complete days of data
        for day in range(5):
            day_start_ms = int((base_time.replace(day=day+1)).timestamp() * 1000)
            
            # Each day gets exactly DAILY_CHECKPOINTS checkpoints
            day_total_return = 0
            for cp_num in range(daily_checkpoints):
                # Vary the returns to create realistic data
                gain = 0.01 * (day + 1) * (cp_num + 1)  # Positive trend
                loss = -0.005 * (day + 1) * (cp_num + 0.5)  # Some losses
                
                checkpoint = PerfCheckpoint(
                    last_update_ms=day_start_ms + ((cp_num + 1) * checkpoint_duration),
                    accum_ms=checkpoint_duration,
                    gain=gain,
                    loss=loss,
                    prev_portfolio_ret=1.0 + (day * 0.01),
                    n_updates=1,
                    mdd=0.99
                )
                test_checkpoints.append(checkpoint)
                day_total_return += (gain + loss)
            
            expected_daily_returns.append(day_total_return)
        
        # Create ledger and get subnet metrics
        ledger = ledger_generator(checkpoints=test_checkpoints)
        ledger.tp_id = TP_ID_PORTFOLIO
        
        from vali_objects.utils.metrics import Metrics
        subnet_daily_returns = LedgerUtils.daily_return_log(ledger)
        subnet_sharpe = Metrics.sharpe(subnet_daily_returns, ledger=ledger)
        
        # Prepare data for circuit
        circuit_data = {
            "perf_ledgers": {
                "test_hotkey": ledger.to_dict()
            },
            "positions": {
                "test_hotkey": {"positions": []}  # Empty positions for this test
            }
        }
        
        # Call circuit (skip actual proof generation to avoid timeouts)
        try:
            result = generate_proof(
                data=circuit_data,
                miner_hotkey="test_hotkey", 
                verbose=False,
                daily_checkpoints=daily_checkpoints
            )
            
            if result and result.get("status") == "success":
                circuit_metrics = result.get("portfolio_metrics", {})
                circuit_sharpe = circuit_metrics.get("sharpe_ratio_scaled", 0)
                
                # Compare key metrics
                self.assertEqual(len(subnet_daily_returns), len(expected_daily_returns),
                               "Subnet should process all complete days")
                
                # Verify daily returns match expectations
                for i, expected in enumerate(expected_daily_returns):
                    self.assertAlmostEqual(subnet_daily_returns[i], expected, places=6,
                                         msg=f"Subnet daily return {i} should match expected")
                
                # The critical test: circuit and subnet Sharpe should be close
                sharpe_diff = abs(circuit_sharpe - subnet_sharpe)
                self.assertLess(sharpe_diff, 0.1, 
                              f"Circuit Sharpe ({circuit_sharpe:.6f}) should closely match "
                              f"subnet Sharpe ({subnet_sharpe:.6f}). Diff: {sharpe_diff:.6f}")
                
                print(f"Metrics parity test passed:")
                print(f"  Days processed: {len(subnet_daily_returns)}")
                print(f"  Subnet Sharpe: {subnet_sharpe:.6f}")
                print(f"  Circuit Sharpe: {circuit_sharpe:.6f}")
                print(f"  Difference: {sharpe_diff:.6f}")
                
            else:
                self.fail(f"Circuit failed to generate results: {result}")
                
        except Exception as e:
            # If circuit fails due to missing dependencies, skip gracefully
            if "nargo" in str(e).lower() or "barretenberg" in str(e).lower():
                self.skipTest(f"Circuit dependencies not available: {e}")
            else:
                raise

