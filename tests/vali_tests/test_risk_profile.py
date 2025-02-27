import unittest
from copy import deepcopy
import numpy as np

from tests.shared_objects.test_utilities import add_orders_to_position
from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.vali_config import TradePair, ValiConfig
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position
from vali_objects.utils.risk_profiling import RiskProfiling


class TestRiskProfile(TestBase):
    """
    This class tests the risk profiling functionality
    """

    def setUp(self):
        super().setUp()

        self.DEFAULT_MINER_HOTKEY = "test_miner"
        self.DEFAULT_POSITION_UUID = "test_position"
        self.DEFAULT_OPEN_MS = 1000
        self.DEFAULT_CLOSE_MS = 2000
        self.DEFAULT_ORDER_MS = 1000
        self.MS_IN_DAY = 86400000
        self.DEFAULT_TRADE_PAIR = TradePair.BTCUSD
        self.default_position = Position(
            position_type=OrderType.LONG,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid=self.DEFAULT_POSITION_UUID,
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=self.DEFAULT_TRADE_PAIR,
        )

    def tearDown(self):
        super().tearDown()

    def test_monotonic_positions(self):
        """Test the monotonically increasing leverage detection with various edge cases"""
        # Test with empty position (no orders)
        position = deepcopy(self.default_position)
        result = RiskProfiling.monotonic_positions(position)
        self.assertEqual(len(result.orders), 0, "Empty position should result in empty monotonic position")

        # Test with single order
        position = deepcopy(self.default_position)
        add_orders_to_position(
            position=position,
            order_type=OrderType.LONG,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            leverages=[0.1],
            prices=[100],
            times=[self.DEFAULT_ORDER_MS]
        )
        result = RiskProfiling.monotonic_positions(position)
        self.assertEqual(len(result.orders), 0, "Position with single order should result in empty monotonic position")

        # Test with winning positions only (should not flag)
        position = deepcopy(self.default_position)
        add_orders_to_position(
            position=position,
            order_type=OrderType.LONG,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            leverages=[0.1, 0.1, 0.1],
            prices=[100, 110, 120],
            times=[self.DEFAULT_ORDER_MS, self.DEFAULT_ORDER_MS + 100, self.DEFAULT_ORDER_MS + 200]
        )
        result = RiskProfiling.monotonic_positions(position)
        self.assertEqual(len(result.orders), 0, "Winning positions should not be flagged")

        # Test with losing positions but decreasing additional leverage
        position = deepcopy(self.default_position)
        add_orders_to_position(
            position=position,
            order_type=OrderType.LONG,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            leverages=[0.3, 0.2, 0.1],
            prices=[100, 90, 80],
            times=[self.DEFAULT_ORDER_MS, self.DEFAULT_ORDER_MS + 100, self.DEFAULT_ORDER_MS + 200]
        )
        result = RiskProfiling.monotonic_positions(position)
        self.assertEqual(len(result.orders), 2, "Losing positions with increasing total leverage should be flagged")

        # Test with losing positions and incrementally decreasing leverage (negative increments)
        position = deepcopy(self.default_position)
        add_orders_to_position(
            position=position,
            order_type=OrderType.LONG,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            leverages=[0.3, -0.1, -0.1],
            prices=[100, 90, 80],
            times=[self.DEFAULT_ORDER_MS, self.DEFAULT_ORDER_MS + 100, self.DEFAULT_ORDER_MS + 200]
        )
        result = RiskProfiling.monotonic_positions(position)
        self.assertEqual(len(result.orders), 0, "Losing positions with decreasing leverage should not be flagged")
        
        # Test SHORT positions with increasing leverage on losing trade
        position = deepcopy(self.default_position)
        position.position_type = OrderType.SHORT
        add_orders_to_position(
            position=position,
            order_type=OrderType.SHORT,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            leverages=[-0.1, -0.1, -0.1],
            prices=[100, 90, 80],
            times=[self.DEFAULT_ORDER_MS, self.DEFAULT_ORDER_MS + 100, self.DEFAULT_ORDER_MS + 200]
        )
        result = RiskProfiling.monotonic_positions(position)
        self.assertEqual(len(result.orders), 0, "Winning SHORT positions should not be flagged")

        # Test SHORT positions with increasing leverage on losing trade
        position = deepcopy(self.default_position)
        position.position_type = OrderType.SHORT
        add_orders_to_position(
            position=position,
            order_type=OrderType.SHORT,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            leverages=[-0.1, -0.1, -0.1],
            prices=[100, 110, 120],
            times=[self.DEFAULT_ORDER_MS, self.DEFAULT_ORDER_MS + 100, self.DEFAULT_ORDER_MS + 200]
        )
        result = RiskProfiling.monotonic_positions(position)
        self.assertEqual(len(result.orders), 2, "Losing SHORT positions with increasing leverage should be flagged")

        # Test with closed position (final order should be ignored)
        position = deepcopy(self.default_position)
        add_orders_to_position(
            position=position,
            order_type=OrderType.LONG,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            leverages=[0.1, 0.1, 0.1, -0.3],
            prices=[100, 90, 80, 70],
            times=[self.DEFAULT_ORDER_MS, self.DEFAULT_ORDER_MS + 100, self.DEFAULT_ORDER_MS + 200, self.DEFAULT_ORDER_MS + 300]
        )
        position.is_closed_position = True
        result = RiskProfiling.monotonic_positions(position)
        self.assertEqual(len(result.orders), 2, "Should only count orders before closing order")

    def test_risk_assessment_steps_utilization(self):
        """Test the steps utilization function"""
        # Test with empty position (no orders)
        position = deepcopy(self.default_position)
        result = RiskProfiling.risk_assessment_steps_utilization(position)
        self.assertEqual(result, 0, "Empty position should have 0 steps")

        # Test with single order
        position = deepcopy(self.default_position)
        add_orders_to_position(
            position=position,
            order_type=OrderType.LONG,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            leverages=[0.1],
            prices=[100],
            times=[self.DEFAULT_ORDER_MS]
        )
        result = RiskProfiling.risk_assessment_steps_utilization(position)
        self.assertEqual(result, 0, "Position with single order should have 0 steps")

        # Test with winning positions only
        position = deepcopy(self.default_position)
        add_orders_to_position(
            position=position,
            order_type=OrderType.LONG,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            leverages=[0.1, 0.1, 0.1],
            prices=[100, 110, 120],
            times=[self.DEFAULT_ORDER_MS, self.DEFAULT_ORDER_MS + 100, self.DEFAULT_ORDER_MS + 200]
        )
        result = RiskProfiling.risk_assessment_steps_utilization(position)
        self.assertEqual(result, 0, "Winning positions should have 0 steps")

        # Test with losing positions and decreasing leverage (negative increments)
        position = deepcopy(self.default_position)
        add_orders_to_position(
            position=position,
            order_type=OrderType.LONG,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            leverages=[0.3, -0.1, -0.1],
            prices=[100, 90, 80],
            times=[self.DEFAULT_ORDER_MS, self.DEFAULT_ORDER_MS + 100, self.DEFAULT_ORDER_MS + 200]
        )
        result = RiskProfiling.risk_assessment_steps_utilization(position)
        self.assertEqual(result, 0, "Losing positions with decreasing leverage should have 0 steps")

        # Test with losing positions and increasing leverage
        position = deepcopy(self.default_position)
        add_orders_to_position(
            position=position,
            order_type=OrderType.LONG,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            leverages=[0.1, 0.1, 0.1],
            prices=[100, 90, 80],
            times=[self.DEFAULT_ORDER_MS, self.DEFAULT_ORDER_MS + 100, self.DEFAULT_ORDER_MS + 200]
        )
        result = RiskProfiling.risk_assessment_steps_utilization(position)
        self.assertEqual(result, 2, "Losing positions with increasing leverage should have 2 steps")
        
        # Test with SHORT positions on losing trade
        position = deepcopy(self.default_position)
        position.position_type = OrderType.SHORT
        add_orders_to_position(
            position=position,
            order_type=OrderType.SHORT,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            leverages=[-0.1, -0.1, -0.1],
            prices=[100, 110, 120],
            times=[self.DEFAULT_ORDER_MS, self.DEFAULT_ORDER_MS + 100, self.DEFAULT_ORDER_MS + 200]
        )

        self.assertEqual(len(position.orders), 3, "Position should have 3 orders")
        result = RiskProfiling.risk_assessment_steps_utilization(position)
        self.assertEqual(result, 2, "Losing SHORT positions with increasing leverage should have 2 steps")

        # Test with closed position
        position = deepcopy(self.default_position)
        add_orders_to_position(
            position=position,
            order_type=OrderType.LONG,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            leverages=[0.1, 0.1, 0.1, -0.3],
            prices=[100, 90, 80, 70],
            times=[self.DEFAULT_ORDER_MS, self.DEFAULT_ORDER_MS + 100, self.DEFAULT_ORDER_MS + 200, self.DEFAULT_ORDER_MS + 300]
        )
        position.is_closed_position = True
        result = RiskProfiling.risk_assessment_steps_utilization(position)
        self.assertEqual(result, 2, "Should only count steps before closing order")

    def test_risk_assessment_monotonic_utilization(self):
        """Test the monotome utilization function"""
        # Test with empty position
        position = deepcopy(self.default_position)
        result = RiskProfiling.risk_assessment_monotonic_utilization(position)
        self.assertEqual(result, 0, "Empty position should have 0 monotonic utilization")

        # Test with single order
        position = deepcopy(self.default_position)
        add_orders_to_position(
            position=position,
            order_type=OrderType.LONG,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            leverages=[0.1],
            prices=[100],
            times=[self.DEFAULT_ORDER_MS]
        )
        result = RiskProfiling.risk_assessment_monotonic_utilization(position)
        self.assertEqual(result, 0, "Position with single order should have 0 monotonic utilization")

        # Test with winning positions only
        position = deepcopy(self.default_position)
        add_orders_to_position(
            position=position,
            order_type=OrderType.LONG,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            leverages=[0.1, 0.1, 0.1],
            prices=[100, 110, 120],
            times=[self.DEFAULT_ORDER_MS, self.DEFAULT_ORDER_MS + 100, self.DEFAULT_ORDER_MS + 200]
        )
        result = RiskProfiling.risk_assessment_monotonic_utilization(position)
        self.assertEqual(result, 0, "Winning positions should have 0 monotonic utilization")

        # Test with losing positions and monotonically increasing leverage
        position = deepcopy(self.default_position)
        add_orders_to_position(
            position=position,
            order_type=OrderType.LONG,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            leverages=[0.1, 0.1, 0.2],
            prices=[100, 90, 80],
            times=[self.DEFAULT_ORDER_MS, self.DEFAULT_ORDER_MS + 100, self.DEFAULT_ORDER_MS + 200]
        )
        result = RiskProfiling.risk_assessment_monotonic_utilization(position)
        self.assertEqual(result, 2, "Losing positions with increasing leverage should have 2 monotonic utilization")

    def test_risk_assessment_margin_utilization(self):
        """Test the margin utilization function"""
        # Test with position having low margin utilization
        position = deepcopy(self.default_position)
        add_orders_to_position(
            position=position,
            order_type=OrderType.LONG,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            leverages=[0.01, 0.01, 0.01],
            prices=[100, 110, 120],
            times=[self.DEFAULT_ORDER_MS, self.DEFAULT_ORDER_MS + 100, self.DEFAULT_ORDER_MS + 200]
        )
        result = RiskProfiling.risk_assessment_margin_utilization(position)
        self.assertLess(result, 0.1, "Low leverage should result in low margin utilization")

        # Test with position having high margin utilization
        position = deepcopy(self.default_position)
        add_orders_to_position(
            position=position,
            order_type=OrderType.LONG,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            leverages=[0.4, 0.05, 0.02],
            prices=[100, 110, 120],
            times=[self.DEFAULT_ORDER_MS, self.DEFAULT_ORDER_MS + 100, self.DEFAULT_ORDER_MS + 200]
        )
        result = RiskProfiling.risk_assessment_margin_utilization(position)
        self.assertGreater(result, 0.8, "High leverage should result in high margin utilization")

        # Test with SHORT position
        position = deepcopy(self.default_position)
        position.position_type = OrderType.SHORT
        add_orders_to_position(
            position=position,
            order_type=OrderType.SHORT,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            leverages=[-0.4, -0.05, -0.02],
            prices=[100, 90, 80],
            times=[self.DEFAULT_ORDER_MS, self.DEFAULT_ORDER_MS + 100, self.DEFAULT_ORDER_MS + 200]
        )
        result = RiskProfiling.risk_assessment_margin_utilization(position)
        self.assertGreater(result, 0.8, "High leverage SHORT position should result in high margin utilization")

    def test_risk_assessment_leverage_advancement_utilization(self):
        """Test the leverage advancement utilization function"""
        # Test with position having low leverage advancement
        position = deepcopy(self.default_position)
        add_orders_to_position(
            position=position,
            order_type=OrderType.LONG,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            leverages=[0.1, 0.01, 0.01],
            prices=[100, 110, 120],
            times=[self.DEFAULT_ORDER_MS, self.DEFAULT_ORDER_MS + 100, self.DEFAULT_ORDER_MS + 200]
        )
        result = RiskProfiling.risk_assessment_leverage_advancement_utilization(position)
        self.assertLess(result, 1.5, "Position with small leverage advancement should have low utilization")
        
        # Test with position having zero initial leverage (edge case)
        position = deepcopy(self.default_position)
        add_orders_to_position(
            position=position,
            order_type=OrderType.LONG,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            leverages=[0.05, 0.1, 0.15],
            prices=[100, 110, 120],
            times=[self.DEFAULT_ORDER_MS, self.DEFAULT_ORDER_MS + 100, self.DEFAULT_ORDER_MS + 200]
        )
        result = RiskProfiling.risk_assessment_leverage_advancement_utilization(position)
        self.assertGreaterEqual(result, 1.0, "Position with zero initial leverage should return at least 1.0")

        # Test with position having high leverage advancement
        position = deepcopy(self.default_position)
        add_orders_to_position(
            position=position,
            order_type=OrderType.LONG,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            leverages=[0.1, 0.2, 0.3],
            prices=[100, 110, 120],
            times=[self.DEFAULT_ORDER_MS, self.DEFAULT_ORDER_MS + 100, self.DEFAULT_ORDER_MS + 200]
        )
        result = RiskProfiling.risk_assessment_leverage_advancement_utilization(position)
        self.assertGreaterEqual(result, 6.0, "Position with large leverage advancement should have high utilization")

    def test_risk_assessment_time_utilization(self):
        """Test the time utilization function"""
        # Test with position having fewer than 3 orders
        position = deepcopy(self.default_position)
        add_orders_to_position(
            position=position,
            order_type=OrderType.LONG,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            leverages=[0.1, 0.1],
            prices=[100, 110],
            times=[self.DEFAULT_ORDER_MS, self.DEFAULT_ORDER_MS + 100]
        )
        result = RiskProfiling.risk_assessment_time_utilization(position)
        self.assertEqual(result, 0.0, "Position with fewer than 3 orders should have 0 time utilization")
        
        # Test with empty position
        position = deepcopy(self.default_position)
        result = RiskProfiling.risk_assessment_time_utilization(position)
        self.assertEqual(result, 0.0, "Empty position should have 0 time utilization")

        # Test with position having perfectly even time intervals
        position = deepcopy(self.default_position)
        add_orders_to_position(
            position=position,
            order_type=OrderType.LONG,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            leverages=[0.1, 0.1, 0.1],
            prices=[100, 110, 120],
            times=[self.DEFAULT_ORDER_MS, self.DEFAULT_ORDER_MS + 100, self.DEFAULT_ORDER_MS + 200]
        )
        result = RiskProfiling.risk_assessment_time_utilization(position)
        self.assertEqual(result, 0.0, "Position with even time intervals should have 0 time utilization")

        # Test with position having uneven time intervals
        position = deepcopy(self.default_position)
        add_orders_to_position(
            position=position,
            order_type=OrderType.LONG,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            leverages=[0.1, 0.1, 0.1],
            prices=[100, 110, 120],
            times=[self.DEFAULT_ORDER_MS, self.DEFAULT_ORDER_MS + 50, self.DEFAULT_ORDER_MS + 200]
        )
        result = RiskProfiling.risk_assessment_time_utilization(position)
        self.assertGreater(result, 0.0, "Position with uneven time intervals should have positive time utilization")

        # Test with position having orders with zero time interval
        position = deepcopy(self.default_position)
        add_orders_to_position(
            position=position,
            order_type=OrderType.LONG,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            leverages=[0.1, 0.1, 0.1],
            prices=[100, 110, 120],
            times=[self.DEFAULT_ORDER_MS, self.DEFAULT_ORDER_MS, self.DEFAULT_ORDER_MS]
        )
        result = RiskProfiling.risk_assessment_time_utilization(position)
        self.assertEqual(result, 0.0, "Position with zero time intervals should handle the edge case")

    def test_risk_profile_single(self):
        """Test the risk profile single function"""
        # Create a position that should trigger risk flags
        position = deepcopy(self.default_position)
        add_orders_to_position(
            position=position,
            order_type=OrderType.LONG,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            leverages=[0.1, 0.2, 0.3, 0.4],
            prices=[100, 90, 80, 70],
            times=[self.DEFAULT_ORDER_MS, self.DEFAULT_ORDER_MS + 50, self.DEFAULT_ORDER_MS + 100, self.DEFAULT_ORDER_MS + 200]
        )
        position.return_at_close = 0.9  # 10% loss
        
        result = RiskProfiling.risk_profile_single(position)
        
        # Verify the keys in the result
        expected_keys = [
            "position_return", "relative_weighting_strength", "overall_flag",
            "steps_utilization", "steps_criteria", 
            "monotonic_utilization", "monotonic_criteria",
            "margin_utilization", "margin_criteria",
            "leverage_advancement_utilization", "leverage_advancement_criteria",
            "time_utilization", "time_criteria"
        ]
        for key in expected_keys:
            self.assertIn(key, result, f"Key {key} missing from risk profile result")
        
        # Verify types of values
        self.assertIsInstance(result["position_return"], float)
        self.assertIsInstance(result["relative_weighting_strength"], float)
        self.assertIsInstance(result["overall_flag"], int)
        self.assertIsInstance(result["steps_utilization"], int)
        self.assertIsInstance(result["steps_criteria"], int)

    def test_risk_profile_reporting(self):
        """Test the risk profile reporting function"""
        # Test with empty list
        result = RiskProfiling.risk_profile_reporting([])
        self.assertEqual(result, {}, "Empty position list should return empty dict")
        
        # Test with single position
        position = deepcopy(self.default_position)
        add_orders_to_position(
            position=position,
            order_type=OrderType.LONG,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            leverages=[0.1, 0.1, 0.1],
            prices=[100, 90, 80],
            times=[self.DEFAULT_ORDER_MS, self.DEFAULT_ORDER_MS + 100, self.DEFAULT_ORDER_MS + 200]
        )
        
        result = RiskProfiling.risk_profile_reporting([position])
        self.assertEqual(len(result), 1, "Should contain one entry for one position")
        self.assertIn(position.position_uuid, result, "Position UUID should be in result")
        
        # Test with multiple positions
        position2 = deepcopy(self.default_position)
        position2.position_uuid = "test_position_2"
        add_orders_to_position(
            position=position2,
            order_type=OrderType.SHORT,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            leverages=[-0.1, -0.1, -0.1],
            prices=[100, 110, 120],
            times=[self.DEFAULT_ORDER_MS, self.DEFAULT_ORDER_MS + 100, self.DEFAULT_ORDER_MS + 200]
        )
        
        result = RiskProfiling.risk_profile_reporting([position, position2])
        self.assertEqual(len(result), 2, "Should contain two entries for two positions")
        self.assertIn(position.position_uuid, result, "First position UUID should be in result")
        self.assertIn(position2.position_uuid, result, "Second position UUID should be in result")

    def test_risk_profile_score_list(self):
        """Test the risk profile score list function with numerical stability cases"""
        # Test with empty list
        result = RiskProfiling.risk_profile_score_list([])
        self.assertEqual(result, 0.0, "Empty position list should return 0.0")
        
        # Test with extremely small return values (to check numerical stability)
        small_return_position = deepcopy(self.default_position)
        add_orders_to_position(
            position=small_return_position,
            order_type=OrderType.LONG,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            leverages=[0.1, 0.1, 0.1],
            prices=[100, 90, 80],
            times=[self.DEFAULT_ORDER_MS, self.DEFAULT_ORDER_MS + 100, self.DEFAULT_ORDER_MS + 200]
        )
        # Set a very small return value to test numerical stability
        small_return_position.return_at_close = 0.0001  # 99.99% loss
        
        # Configure for guaranteed risk flagging
        original_steps = ValiConfig.RISK_PROFILING_STEPS_CRITERIA
        original_time = ValiConfig.RISK_PROFILING_TIME_CRITERIA
        ValiConfig.RISK_PROFILING_STEPS_CRITERIA = 1
        ValiConfig.RISK_PROFILING_TIME_CRITERIA = 0.0

        # The function should handle this edge case without numerical errors
        result = RiskProfiling.risk_profile_score_list([small_return_position])
        self.assertGreaterEqual(result, 0.0, "Function should handle extremely small return values")
        self.assertLessEqual(result, 1.0, "Function should handle extremely small return values")
        
        # Restore original values
        ValiConfig.RISK_PROFILING_STEPS_CRITERIA = original_steps
        ValiConfig.RISK_PROFILING_TIME_CRITERIA = original_time
        
        # Test with single non-risky position
        safe_position = deepcopy(self.default_position)
        add_orders_to_position(
            position=safe_position,
            order_type=OrderType.LONG,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            leverages=[0.05, 0.03, 0.01],
            prices=[100, 110, 120],
            times=[self.DEFAULT_ORDER_MS, self.DEFAULT_ORDER_MS + 100, self.DEFAULT_ORDER_MS + 200]
        )
        safe_position.return_at_close = 1.1  # 10% gain
        
        # Save original thresholds to ensure position is safe
        original_steps = ValiConfig.RISK_PROFILING_STEPS_CRITERIA
        original_monotonic = ValiConfig.RISK_PROFILING_MONOTONIC_CRITERIA
        original_margin = ValiConfig.RISK_PROFILING_MARGIN_CRITERIA
        original_leverage = ValiConfig.RISK_PROFILING_LEVERAGE_ADVANCE
        original_time = ValiConfig.RISK_PROFILING_TIME_CRITERIA
        
        ValiConfig.RISK_PROFILING_STEPS_CRITERIA = 2
        ValiConfig.RISK_PROFILING_MONOTONIC_CRITERIA = 2
        ValiConfig.RISK_PROFILING_MARGIN_CRITERIA = 0.2
        ValiConfig.RISK_PROFILING_LEVERAGE_ADVANCE = 2.0
        ValiConfig.RISK_PROFILING_TIME_CRITERIA = 0.0
        
        result = RiskProfiling.risk_profile_score_list([safe_position])
        self.assertEqual(result, 0.0, "Non-risky position should have score 0.0")
        
        # Test with single risky position
        risky_position = deepcopy(self.default_position)
        add_orders_to_position(
            position=risky_position,
            order_type=OrderType.LONG,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            leverages=[0.1, 0.1, 0.2],
            prices=[100, 90, 80],
            times=[self.DEFAULT_ORDER_MS, self.DEFAULT_ORDER_MS + 50, self.DEFAULT_ORDER_MS + 200]
        )
        risky_position.return_at_close = 0.9  # 10% loss
        
        result = RiskProfiling.risk_profile_score_list([risky_position])
        self.assertEqual(result, 1.0, "Risky position should have score 1.0")
        
        # Test with mix of risky and non-risky positions
        result = RiskProfiling.risk_profile_score_list([safe_position, risky_position])
        
        # The weighted average should consider return_at_close with the RISK_PROFILING_SCOPING_MECHANIC power
        # Safe position has 10% gain, risky position has 10% loss
        # Weight for safe position: 1.1^100
        # Weight for risky position: 0.9^100
        # Risky has score 1, safe has score 0
        # Should be heavily weighted towards safe position's score (0)
        self.assertLess(result, 0.01, "Weighted score should be closer to safe position's score")
        
        # Restore original thresholds
        ValiConfig.RISK_PROFILING_STEPS_CRITERIA = original_steps
        ValiConfig.RISK_PROFILING_MONOTONIC_CRITERIA = original_monotonic
        ValiConfig.RISK_PROFILING_MARGIN_CRITERIA = original_margin
        ValiConfig.RISK_PROFILING_LEVERAGE_ADVANCE = original_leverage
        ValiConfig.RISK_PROFILING_TIME_CRITERIA = original_time

    def test_risk_profile_score(self):
        """Test the risk profile score function with multiple miners"""
        # Create positions for two miners
        safe_position = deepcopy(self.default_position)
        add_orders_to_position(
            position=safe_position,
            order_type=OrderType.LONG,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            leverages=[0.05, 0.03, 0.01],
            prices=[100, 110, 120],
            times=[self.DEFAULT_ORDER_MS, self.DEFAULT_ORDER_MS + 100, self.DEFAULT_ORDER_MS + 200]
        )
        safe_position.return_at_close = 1.1  # 10% gain
        
        risky_position = deepcopy(self.default_position)
        risky_position.miner_hotkey = "risky_miner"
        add_orders_to_position(
            position=risky_position,
            order_type=OrderType.LONG,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            leverages=[0.1, 0.1, 0.2],
            prices=[100, 90, 80],
            times=[self.DEFAULT_ORDER_MS, self.DEFAULT_ORDER_MS + 50, self.DEFAULT_ORDER_MS + 200]
        )
        risky_position.return_at_close = 0.9  # 10% loss
        
        # Make the risky position actually trigger risk criteria
        original_steps = ValiConfig.RISK_PROFILING_STEPS_CRITERIA
        original_monotonic = ValiConfig.RISK_PROFILING_MONOTONIC_CRITERIA
        original_margin = ValiConfig.RISK_PROFILING_MARGIN_CRITERIA
        original_leverage = ValiConfig.RISK_PROFILING_LEVERAGE_ADVANCE
        original_time = ValiConfig.RISK_PROFILING_TIME_CRITERIA
        
        ValiConfig.RISK_PROFILING_STEPS_CRITERIA = 2
        ValiConfig.RISK_PROFILING_MONOTONIC_CRITERIA = 2
        ValiConfig.RISK_PROFILING_MARGIN_CRITERIA = 0.2
        ValiConfig.RISK_PROFILING_TIME_CRITERIA = 0.1
        
        miner_positions = {
            "test_miner": [safe_position],
            "risky_miner": [risky_position]
        }
        
        result = RiskProfiling.risk_profile_score(miner_positions)
        
        self.assertEqual(len(result), 2, "Should have scores for two miners")
        self.assertIn("test_miner", result, "test_miner should be in results")
        self.assertIn("risky_miner", result, "risky_miner should be in results")
        
        self.assertEqual(result["test_miner"], 0.0, "Safe miner should have score 0.0")
        self.assertEqual(result["risky_miner"], 1.0, "Risky miner should have score 1.0")
        
        # Restore original values
        ValiConfig.RISK_PROFILING_STEPS_CRITERIA = original_steps
        ValiConfig.RISK_PROFILING_MONOTONIC_CRITERIA = original_monotonic
        ValiConfig.RISK_PROFILING_MARGIN_CRITERIA = original_margin
        ValiConfig.RISK_PROFILING_LEVERAGE_ADVANCE = original_leverage
        ValiConfig.RISK_PROFILING_TIME_CRITERIA = original_time

    def test_risk_profile_penalty(self):
        """Test the risk profile penalty function"""
        # Test with empty dict
        result = RiskProfiling.risk_profile_penalty({})
        self.assertEqual(result, {}, "Empty dict should return empty dict")
        
        # Create positions for two miners
        safe_position = deepcopy(self.default_position)
        add_orders_to_position(
            position=safe_position,
            order_type=OrderType.LONG,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            leverages=[0.05, 0.03, 0.01],
            prices=[100, 110, 120],
            times=[self.DEFAULT_ORDER_MS, self.DEFAULT_ORDER_MS + 100, self.DEFAULT_ORDER_MS + 200]
        )
        safe_position.return_at_close = 1.1  # 10% gain
        
        risky_position = deepcopy(self.default_position)
        risky_position.miner_hotkey = "risky_miner"
        add_orders_to_position(
            position=risky_position,
            order_type=OrderType.LONG,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            leverages=[0.1, 0.1, 0.2],
            prices=[100, 90, 80],
            times=[self.DEFAULT_ORDER_MS, self.DEFAULT_ORDER_MS + 50, self.DEFAULT_ORDER_MS + 200]
        )
        risky_position.return_at_close = 0.9  # 10% loss
        
        # Make the risky position actually trigger risk criteria
        original_steps = ValiConfig.RISK_PROFILING_STEPS_CRITERIA
        original_monotonic = ValiConfig.RISK_PROFILING_MONOTONIC_CRITERIA
        original_margin = ValiConfig.RISK_PROFILING_MARGIN_CRITERIA
        original_leverage = ValiConfig.RISK_PROFILING_LEVERAGE_ADVANCE
        original_time = ValiConfig.RISK_PROFILING_TIME_CRITERIA
        
        ValiConfig.RISK_PROFILING_STEPS_CRITERIA = 2
        ValiConfig.RISK_PROFILING_MONOTONIC_CRITERIA = 2
        ValiConfig.RISK_PROFILING_MARGIN_CRITERIA = 0.2
        ValiConfig.RISK_PROFILING_TIME_CRITERIA = 0.1
        
        miner_positions = {
            "test_miner": [safe_position],
            "risky_miner": [risky_position]
        }
        
        result = RiskProfiling.risk_profile_penalty(miner_positions)
        
        self.assertEqual(len(result), 2, "Should have penalties for two miners")
        self.assertIn("test_miner", result, "test_miner should be in results")
        self.assertIn("risky_miner", result, "risky_miner should be in results")
        
        # Safe miner should have penalty close to 1 (no penalty)
        self.assertGreater(result["test_miner"], 0.98, "Safe miner should have minimal penalty")
        
        # Risky miner should have significant penalty (less than 0.5)
        self.assertLess(result["risky_miner"], 0.5, "Risky miner should have significant penalty")
        
        # Restore original values
        ValiConfig.RISK_PROFILING_STEPS_CRITERIA = original_steps
        ValiConfig.RISK_PROFILING_MONOTONIC_CRITERIA = original_monotonic
        ValiConfig.RISK_PROFILING_MARGIN_CRITERIA = original_margin
        ValiConfig.RISK_PROFILING_LEVERAGE_ADVANCE = original_leverage
        ValiConfig.RISK_PROFILING_TIME_CRITERIA = original_time

    def test_integration_complete_risk_assessment(self):
        """Integration test for full risk assessment workflow"""
        # Create a diverse set of positions
        positions = []
        
        # Position 1: Safe position with good returns
        pos1 = deepcopy(self.default_position)
        pos1.position_uuid = "pos1"
        add_orders_to_position(
            position=pos1,
            order_type=OrderType.LONG,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            leverages=[0.05, 0.03, 0.01],
            prices=[100, 110, 120],
            times=[self.DEFAULT_ORDER_MS, self.DEFAULT_ORDER_MS + 100, self.DEFAULT_ORDER_MS + 200]
        )
        pos1.return_at_close = 1.2  # 20% gain
        positions.append(pos1)
        
        # Position 2: Risky position with increasing leverage on losing trade
        pos2 = deepcopy(self.default_position)
        pos2.position_uuid = "pos2"
        add_orders_to_position(
            position=pos2,
            order_type=OrderType.LONG,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            leverages=[0.1, 0.1, 0.2],
            prices=[100, 90, 80],
            times=[self.DEFAULT_ORDER_MS, self.DEFAULT_ORDER_MS + 50, self.DEFAULT_ORDER_MS + 200]
        )
        pos2.return_at_close = 0.8  # 20% loss
        positions.append(pos2)
        
        # Position 3: Risky SHORT position with high leverage
        pos3 = deepcopy(self.default_position)
        pos3.position_uuid = "pos3"
        pos3.position_type = OrderType.SHORT
        add_orders_to_position(
            position=pos3,
            order_type=OrderType.SHORT,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            leverages=[-0.1, -0.1, -0.3],
            prices=[100, 110, 120],
            times=[self.DEFAULT_ORDER_MS, self.DEFAULT_ORDER_MS + 100, self.DEFAULT_ORDER_MS + 200]
        )
        pos3.return_at_close = 0.9  # 10% loss
        positions.append(pos3)
        
        # Position 4: Zero-variance position with all orders at same time
        pos4 = deepcopy(self.default_position)
        pos4.position_uuid = "pos4"
        add_orders_to_position(
            position=pos4,
            order_type=OrderType.LONG,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            leverages=[0.1, 0.1, 0.1],
            prices=[100, 100, 100],
            times=[self.DEFAULT_ORDER_MS, self.DEFAULT_ORDER_MS, self.DEFAULT_ORDER_MS]
        )
        pos4.return_at_close = 1.0  # 0% gain/loss
        positions.append(pos4)
        
        # Make sure the risky positions actually trigger risk criteria
        original_steps = ValiConfig.RISK_PROFILING_STEPS_CRITERIA
        original_monotonic = ValiConfig.RISK_PROFILING_MONOTONIC_CRITERIA
        original_margin = ValiConfig.RISK_PROFILING_MARGIN_CRITERIA
        original_leverage = ValiConfig.RISK_PROFILING_LEVERAGE_ADVANCE
        original_time = ValiConfig.RISK_PROFILING_TIME_CRITERIA
        
        ValiConfig.RISK_PROFILING_STEPS_CRITERIA = 2
        ValiConfig.RISK_PROFILING_MONOTONIC_CRITERIA = 2
        ValiConfig.RISK_PROFILING_MARGIN_CRITERIA = 0.4
        ValiConfig.RISK_PROFILING_LEVERAGE_ADVANCE = 3.0
        ValiConfig.RISK_PROFILING_TIME_CRITERIA = 0.1
        
        # Test individual profiles
        for pos in positions:
            profile = RiskProfiling.risk_profile_single(pos)
            self.assertIsInstance(profile, dict, f"Profile for {pos.position_uuid} should be a dict")
            self.assertIn("overall_flag", profile, f"Profile for {pos.position_uuid} missing overall_flag")
            
            # Verify flag states match our expectations
            if pos.position_uuid == "pos1":
                self.assertEqual(profile["overall_flag"], 0, "Safe position should not be flagged")
            elif pos.position_uuid == "pos2":
                self.assertEqual(profile["overall_flag"], 1, "Risky position should be flagged")
        
        # Test full risk profile report
        report = RiskProfiling.risk_profile_reporting(positions)
        self.assertEqual(len(report), 4, "Report should contain 4 positions")
        
        # Test with multiple miners
        miner_positions = {
            "safe_miner": [pos1, pos4],
            "risky_miner": [pos2, pos3]
        }
        
        # Test scores
        scores = RiskProfiling.risk_profile_score(miner_positions)
        self.assertEqual(len(scores), 2, "Should have scores for 2 miners")
        self.assertLess(scores["safe_miner"], scores["risky_miner"], "Safe miner should have lower risk score")
        
        # Test penalties
        penalties = RiskProfiling.risk_profile_penalty(miner_positions)
        self.assertEqual(len(penalties), 2, "Should have penalties for 2 miners")
        self.assertGreater(penalties["safe_miner"], penalties["risky_miner"], "Safe miner should have higher penalty multiplier")
        
        # Test edge case with empty miner
        empty_miner_positions = {
            "empty_miner": [],
            "risky_miner": [pos2, pos3]
        }
        
        scores = RiskProfiling.risk_profile_score(empty_miner_positions)
        self.assertEqual(scores["empty_miner"], 0.0, "Empty miner should have 0.0 score")
        
        penalties = RiskProfiling.risk_profile_penalty(empty_miner_positions)
        self.assertGreater(penalties["empty_miner"], 0.98, "Empty miner should have minimal penalty")
        
        # Test random positions don't trigger penalties
        # Generate 10 "random" positions with reasonable patterns
        np.random.seed(42)  # For reproducibility
        random_positions = []
        
        for i in range(10):
            pos = deepcopy(self.default_position)
            pos.position_uuid = f"random_{i}"
            
            # Reasonable number of orders (2-5)
            n_orders = np.random.randint(2, 6)
            
            # Order type randomly chosen
            order_type = np.random.choice([OrderType.LONG, OrderType.SHORT])
            pos.position_type = order_type
            
            # For short positions, make leverages negative
            leverage_sign = -1 if order_type == OrderType.SHORT else 1
            
            # Generate reasonable leverages (0.01-0.2)
            leverages = [leverage_sign * np.random.uniform(0.01, 0.2) for _ in range(n_orders)]
            
            # Generate reasonable prices with some trend and noise
            base_price = 100
            trend = np.random.uniform(-0.2, 0.2)  # -20% to +20% trend
            prices = []
            for j in range(n_orders):
                price = base_price * (1 + trend * j/n_orders + np.random.uniform(-0.05, 0.05))
                prices.append(price)
            
            # Generate evenly spaced timestamps
            times = [self.DEFAULT_ORDER_MS + j * 100 for j in range(n_orders)]
            
            # Add some noise to timestamps for realism
            times = [t + np.random.randint(-10, 10) for t in times]
            
            add_orders_to_position(
                position=pos,
                order_type=order_type,
                trade_pair=self.DEFAULT_TRADE_PAIR,
                leverages=leverages,
                prices=prices,
                times=times
            )
            
            # Set a reasonable return
            pos.return_at_close = 1.0 + np.random.uniform(-0.1, 0.2)  # -10% to +20%
            
            random_positions.append(pos)
        
        # Check if any random positions are flagged as risky
        risky_count = 0
        for pos in random_positions:
            if RiskProfiling.risk_profile_full_criteria(pos):
                risky_count += 1
        
        # We expect most randomly generated positions to not be risky
        self.assertLess(risky_count, 3, "Less than 30% of random positions should be flagged as risky")
        
        # Restore original values
        ValiConfig.RISK_PROFILING_STEPS_CRITERIA = original_steps
        ValiConfig.RISK_PROFILING_MONOTONIC_CRITERIA = original_monotonic
        ValiConfig.RISK_PROFILING_MARGIN_CRITERIA = original_margin
        ValiConfig.RISK_PROFILING_LEVERAGE_ADVANCE = original_leverage
        ValiConfig.RISK_PROFILING_TIME_CRITERIA = original_time


if __name__ == "__main__":
    unittest.main()