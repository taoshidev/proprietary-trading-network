from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.utils.ledger_utils import LedgerUtils
from vali_objects.utils.orthogonality import Orthogonality

from vali_objects.vali_config import ValiConfig
from vali_objects.vali_dataclasses.perf_ledger import TP_ID_PORTFOLIO, PerfLedger, PerfCheckpoint
from tests.shared_objects.test_utilities import generate_ledger, checkpoint_generator, ledger_generator
import random
from datetime import datetime, timezone, date as date_type
import math
import copy


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
                mdd=0.99
            ),
            PerfCheckpoint(
                last_update_ms=day1_noon_ms,
                accum_ms=checkpoint_duration,
                open_ms=day1_noon_ms - checkpoint_duration,
                gain=0.1,
                loss=-0.05,
                prev_portfolio_ret=1.0,
                n_updates=1,
                mdd=0.99
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
                mdd=0.99
            ),
            PerfCheckpoint(
                last_update_ms=day2_noon_ms,
                accum_ms=checkpoint_duration,
                open_ms=day2_noon_ms - checkpoint_duration,
                gain=0.2,
                loss=-0.1,
                prev_portfolio_ret=1.0,
                n_updates=1,
                mdd=0.99
            )
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
                    mdd=0.99
                )
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
            mdd=0.99
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
    def test_max_drawdown(self):
        l1 = generate_ledger(0.1, mdd=0.99)[TP_ID_PORTFOLIO]
        
        # Empty ledger test
        empty_ledger = PerfLedger()
        self.assertEqual(LedgerUtils.max_drawdown(empty_ledger), 0.0)

        # Valid ledger tests
        self.assertEqual(LedgerUtils.max_drawdown(l1), 0.99)

        l2 = generate_ledger(0.1, mdd=0.95)[TP_ID_PORTFOLIO]
        self.assertEqual(LedgerUtils.max_drawdown(l2), 0.95)

        l3 = generate_ledger(0.1, mdd=0.99)[TP_ID_PORTFOLIO]
        l3_cps = l3.cps
        l3_cps[-1].mdd = 0.5
        self.assertEqual(LedgerUtils.max_drawdown(l3), 0.5)

        l4 = generate_ledger(0.1, mdd=0.99)[TP_ID_PORTFOLIO]
        l4_cps = l4.cps
        l4_cps[0].mdd = 0.5
        self.assertEqual(LedgerUtils.max_drawdown(l4), 0.5)

        for element in [l1, l2, l3, l4]:
            self.assertGreaterEqual(LedgerUtils.max_drawdown(element), 0)
            self.assertLessEqual(LedgerUtils.max_drawdown(element), 1)

        # Test with a minimal ledger containing just a few checkpoints
        drawdowns = [0.99, 0.98]
        checkpoints = [checkpoint_generator(mdd=mdd) for mdd in drawdowns]
        minimal_ledger = ledger_generator(checkpoints=checkpoints)
        self.assertEqual(LedgerUtils.max_drawdown(minimal_ledger), 0.98)

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


    def _create_test_ledger(self, returns_pattern):
        """Helper to create a test ledger with specific return pattern."""
        checkpoints = []

        for i, ret in enumerate(returns_pattern):
            checkpoint = PerfCheckpoint(
                last_update_ms=1742323014691 + i * 12 * 60 * 60 * 1000,  # Daily
                prev_portfolio_ret=1.0 + ret,  # Convert return to portfolio value
                gain=ret if ret > 0 else 0.0,
                loss=ret if ret < 0 else 0.0,
                accum_ms=12 * 60 * 60 * 1000,
                open_ms=12 * 60 * 60 * 1000
            )
            checkpoints.append(checkpoint)

        return ledger_generator(checkpoints=checkpoints)

    def test_basic_correlation_penalty_example(self):
        """Basic correlation-based orthogonality penalty."""
        # Create miners with different strategies
        # Need enough data points to meet statistical confidence minimum

        # Miner 1: Steady positive returns
        miner1_returns = [0.01, 0.015, 0.008, 0.012, 0.009] * 50  # 125 days

        # Miner 2: Similar to miner 1 (high correlation - should be penalized more)
        miner2_returns = [0.011, 0.016, 0.007, 0.013, 0.008] * 50

        # Miner 3: Opposite strategy (negative correlation)
        miner3_returns = [-0.01, -0.015, -0.008, -0.012, -0.009] * 50

        # Miner 4: Volatile, different pattern (low correlation)
        miner4_returns = [0.05, -0.03, 0.02, -0.01, 0.04] * 50

        # Create ledgers
        ledgers = {
            'miner1': self._create_test_ledger(miner1_returns),
            'miner2': self._create_test_ledger(miner2_returns),
            'miner3': self._create_test_ledger(miner3_returns),
            'miner4': self._create_test_ledger(miner4_returns),
        }

        # Calculate orthogonality penalties
        penalties = LedgerUtils.orthogonality_penalty(ledgers)

        print("\\nBasic Correlation Penalty Example:")
        for miner, penalty in penalties.items():
            print(f"  {miner}: {penalty:.4f}")

        # Assertions
        self.assertEqual(len(penalties), 4)

        # All penalties should be non-negative (exact values depend on data meeting thresholds)
        for penalty in penalties.values():
            self.assertGreaterEqual(penalty, 0.0)

    def test_complete_orthogonality_workflow_example(self):
        # Create miners with different strategies and capital
        strategies = {
            'trend_follower_1': [0.02, 0.015, -0.01, 0.025, 0.01] * 25,  # Trend following
            'trend_follower_2': [0.018, 0.013, -0.008, 0.022, 0.009] * 25,  # Similar trend following
            'mean_reverter': [-0.015, 0.025, -0.02, 0.03, -0.01] * 25,  # Mean reversion
            'volatility_trader': [0.05, -0.04, 0.03, -0.02, 0.06] * 25,  # High volatility
        }

        ledgers = {
            miner: self._create_test_ledger(returns)
            for miner, returns in strategies.items()
        }

        # Calculate correlation matrix
        miner_returns = {}
        for miner, ledger in ledgers.items():
            returns = LedgerUtils.daily_return_log(ledger)
            miner_returns[miner] = returns

        corr_matrix, mean_correlations = Orthogonality.correlation_matrix(miner_returns)

        print("Correlation matrix:")
        if not corr_matrix.empty:
            print(corr_matrix.round(3))

        print("\\nMean pairwise correlations:")
        for miner, corr in mean_correlations.items():
            print(f"  {miner}: {corr:.3f}")

        # Calculate final orthogonality penalties
        penalties = LedgerUtils.orthogonality_penalty(ledgers)

        print("\\nFinal orthogonality penalties:")
        for miner, penalty in sorted(penalties.items(), key=lambda x: x[1], reverse=True):
            print(f"{miner}: {penalty:.4f}")

        # Validate results
        self.assertEqual(len(penalties), 4)

        # All penalties should be non-negative
        for penalty in penalties.values():
            self.assertGreaterEqual(penalty, 0.0)

        # trend_follower_1 and trend_follower_2 should have higher penalties
        # due to similar strategies (high correlation)
        print("\\nAnalysis:")
        print("Trend followers have higher correlation and should receive higher penalties")
        print("Mean reverter and volatility trader should have lower penalties due to different strategies")

