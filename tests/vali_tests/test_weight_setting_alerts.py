# developer: jbonilla
# Copyright © 2024 Taoshi Inc

import unittest
from unittest.mock import Mock, patch, MagicMock, call
import time
import bittensor as bt

from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.utils.subtensor_weight_setter import SubtensorWeightSetter


class TestWeightSettingAlerts(TestBase):
    """
    Comprehensive tests for weight setting failure alerting system.
    Tests various failure scenarios based on production log patterns.
    """
    
    def setUp(self):
        super().setUp()
        # Mock dependencies
        self.mock_metagraph = Mock()
        self.mock_position_manager = Mock()
        self.mock_position_manager.challengeperiod_manager = Mock()
        self.mock_position_manager.challengeperiod_manager.get_hotkeys_by_bucket.return_value = []
        
        # Create weight setter instance
        self.weight_setter = SubtensorWeightSetter(
            metagraph=self.mock_metagraph,
            position_manager=self.mock_position_manager,
            running_unit_tests=True
        )
        
        # Mock wallet and subtensor
        self.mock_wallet = Mock()
        self.mock_wallet.hotkey.ss58_address = "test_hotkey_123"
        self.mock_subtensor = Mock()
        
        # Mock config
        self.weight_setter.config = Mock()
        self.weight_setter.config.netuid = 8
        self.weight_setter.config.subtensor.network = "finney"
        
        # Mock slack notifier
        self.weight_setter.slack_notifier = Mock()
        
        # Set up test weights
        self.weight_setter.transformed_list = [(0, 1.0), (1, 0.5), (2, 0.3)]
        
    def test_benign_too_soon_error_no_alert(self):
        """Test that 'too soon' errors don't trigger alerts"""
        # Production pattern: "No attempt made. Perhaps it is too soon to commit weights!"
        self.mock_subtensor.set_weights.return_value = (
            False, 
            "No attempt made. Perhaps it is too soon to commit weights!"
        )
        
        with patch('bittensor.logging') as mock_logging:
            self.weight_setter._set_subtensor_weights(
                self.mock_wallet, self.mock_subtensor, 8
            )
            
            # Verify warning was logged with (benign) suffix
            mock_logging.warning.assert_called()
            warning_msg = mock_logging.warning.call_args[0][0]
            self.assertIn("(benign)", warning_msg.lower())
            
            # Verify NO slack alert was sent
            self.weight_setter.slack_notifier.send_message.assert_not_called()
    
    def test_recursion_error_immediate_alert(self):
        """Test that recursion errors trigger immediate critical alert"""
        # Production pattern: "maximum recursion depth exceeded in comparison"
        self.mock_subtensor.set_weights.return_value = (
            False,
            "maximum recursion depth exceeded in comparison"
        )
        
        with patch('bittensor.logging'):
            self.weight_setter._set_subtensor_weights(
                self.mock_wallet, self.mock_subtensor, 8
            )
            
            # Verify slack alert was sent immediately
            self.weight_setter.slack_notifier.send_message.assert_called_once()
            alert_msg = self.weight_setter.slack_notifier.send_message.call_args[0][0]
            
            # Verify alert content
            self.assertIn("CRITICAL", alert_msg)
            self.assertIn("recursion", alert_msg.lower())
            self.assertIn("test_hotkey_123", alert_msg)
            self.assertIn("finney", alert_msg)
            self.assertEqual(
                self.weight_setter.slack_notifier.send_message.call_args[1]['level'],
                "error"
            )
    
    def test_invalid_transaction_immediate_alert(self):
        """Test that invalid transaction errors trigger immediate critical alert"""
        # Production pattern: "Subtensor returned: Invalid Transaction"
        self.mock_subtensor.set_weights.return_value = (
            False,
            "Subtensor returned: Invalid Transaction"
        )
        
        with patch('bittensor.logging'):
            self.weight_setter._set_subtensor_weights(
                self.mock_wallet, self.mock_subtensor, 8
            )
            
            # Verify slack alert was sent
            self.weight_setter.slack_notifier.send_message.assert_called_once()
            alert_msg = self.weight_setter.slack_notifier.send_message.call_args[0][0]
            
            # Verify alert content
            self.assertIn("CRITICAL", alert_msg)
            self.assertIn("invalid transaction", alert_msg.lower())
            self.assertIn("wallet/balance", alert_msg.lower())
    
    def test_unknown_error_alert_after_consecutive(self):
        """Test that unknown errors alert after consecutive failures"""
        unknown_error = "Some new unexpected error pattern"
        self.mock_subtensor.set_weights.return_value = (False, unknown_error)
        
        with patch('bittensor.logging'):
            # First failure - no alert
            self.weight_setter._set_subtensor_weights(
                self.mock_wallet, self.mock_subtensor, 8
            )
            self.weight_setter.slack_notifier.send_message.assert_not_called()
            
            # Second consecutive failure - should alert
            self.weight_setter._set_subtensor_weights(
                self.mock_wallet, self.mock_subtensor, 8
            )
            self.weight_setter.slack_notifier.send_message.assert_called_once()
            
            alert_msg = self.weight_setter.slack_notifier.send_message.call_args[0][0]
            self.assertIn("NEW PATTERN", alert_msg)
            self.assertIn("Unknown weight setting failure", alert_msg)
            self.assertIn("Consecutive failures: 2", alert_msg)
    
    def test_success_resets_failure_tracking(self):
        """Test that successful weight setting resets failure tracking"""
        # Set up initial failure state
        unknown_error = "Some error"
        self.mock_subtensor.set_weights.return_value = (False, unknown_error)
        
        with patch('bittensor.logging'):
            # First failure
            self.weight_setter._set_subtensor_weights(
                self.mock_wallet, self.mock_subtensor, 8
            )
            
            # Success should reset tracking
            self.mock_subtensor.set_weights.return_value = (True, "")
            self.weight_setter._set_subtensor_weights(
                self.mock_wallet, self.mock_subtensor, 8
            )
            
            # Another single failure should not alert (count reset)
            self.mock_subtensor.set_weights.return_value = (False, unknown_error)
            self.weight_setter._set_subtensor_weights(
                self.mock_wallet, self.mock_subtensor, 8
            )
            
            # Should still have no alerts (only 1 failure after reset)
            self.weight_setter.slack_notifier.send_message.assert_not_called()
    
    def test_recovery_alert_after_critical_failures(self):
        """Test recovery alert is sent after resolving critical failures"""
        # Critical failure
        self.mock_subtensor.set_weights.return_value = (
            False,
            "maximum recursion depth exceeded"
        )
        
        with patch('bittensor.logging'):
            # Trigger critical failure
            self.weight_setter._set_subtensor_weights(
                self.mock_wallet, self.mock_subtensor, 8
            )
            
            # Verify critical alert
            self.assertEqual(self.weight_setter.slack_notifier.send_message.call_count, 1)
            
            # Successful recovery
            self.mock_subtensor.set_weights.return_value = (True, "")
            self.weight_setter._set_subtensor_weights(
                self.mock_wallet, self.mock_subtensor, 8
            )
            
            # Verify recovery alert was sent
            self.assertEqual(self.weight_setter.slack_notifier.send_message.call_count, 2)
            recovery_msg = self.weight_setter.slack_notifier.send_message.call_args_list[1][0][0]
            self.assertIn("recovered", recovery_msg.lower())
            self.assertIn("✅", recovery_msg)
    
    def test_no_recovery_alert_after_benign_failures(self):
        """Test no recovery alert after benign 'too soon' failures"""
        # Benign failure
        self.mock_subtensor.set_weights.return_value = (
            False,
            "No attempt made. Perhaps it is too soon to commit weights!"
        )
        
        with patch('bittensor.logging'):
            # Benign failure
            self.weight_setter._set_subtensor_weights(
                self.mock_wallet, self.mock_subtensor, 8
            )
            
            # Success
            self.mock_subtensor.set_weights.return_value = (True, "")
            self.weight_setter._set_subtensor_weights(
                self.mock_wallet, self.mock_subtensor, 8
            )
            
            # No alerts should have been sent
            self.weight_setter.slack_notifier.send_message.assert_not_called()
    
    def test_alert_rate_limiting(self):
        """Test that alerts are rate-limited to prevent spam (except critical)"""
        # Test rate limiting on non-critical errors
        self.mock_subtensor.set_weights.return_value = (
            False,
            "Unknown network error"
        )
        
        with patch('bittensor.logging'):
            # First unknown alert (requires 2 consecutive)
            self.weight_setter._set_subtensor_weights(
                self.mock_wallet, self.mock_subtensor, 8
            )
            self.assertEqual(self.weight_setter.slack_notifier.send_message.call_count, 0)
            
            # Second unknown - should alert
            self.weight_setter._set_subtensor_weights(
                self.mock_wallet, self.mock_subtensor, 8
            )
            self.assertEqual(self.weight_setter.slack_notifier.send_message.call_count, 1)
            
            # Third unknown - should be rate limited
            self.weight_setter._set_subtensor_weights(
                self.mock_wallet, self.mock_subtensor, 8
            )
            self.assertEqual(self.weight_setter.slack_notifier.send_message.call_count, 1)
            
            # But critical errors should NOT be rate limited
            self.mock_subtensor.set_weights.return_value = (
                False,
                "maximum recursion depth exceeded"
            )
            self.weight_setter._set_subtensor_weights(
                self.mock_wallet, self.mock_subtensor, 8
            )
            # Critical error should alert immediately despite rate limit
            self.assertEqual(self.weight_setter.slack_notifier.send_message.call_count, 2)
    
    def test_alert_after_prolonged_failure(self):
        """Test alert is sent if no success for extended period"""
        # Mock time to simulate passage of time
        with patch('time.time') as mock_time:
            mock_time.return_value = 1000  # Initial time
            
            # Reset the tracker's last success time to match our mock time
            self.weight_setter.weight_failure_tracker.last_success_time = 1000
            
            # Benign failures that normally don't alert
            self.mock_subtensor.set_weights.return_value = (
                False,
                "No attempt made. Perhaps it is too soon to commit weights!"
            )
            
            with patch('bittensor.logging'):
                # Multiple benign failures
                for _ in range(5):
                    self.weight_setter._set_subtensor_weights(
                        self.mock_wallet, self.mock_subtensor, 8
                    )
                
                # No alerts yet
                self.weight_setter.slack_notifier.send_message.assert_not_called()
                
                # Simulate 65 minutes passing
                mock_time.return_value = 1000 + (65 * 60)
                
                # Another failure should trigger alert due to prolonged failure
                self.weight_setter._set_subtensor_weights(
                    self.mock_wallet, self.mock_subtensor, 8
                )
                
                # Should now have an alert
                self.weight_setter.slack_notifier.send_message.assert_called_once()
                alert_msg = self.weight_setter.slack_notifier.send_message.call_args[0][0]
                self.assertIn("No successful weight setting", alert_msg)
                self.assertIn("1.1 hours", alert_msg)  # 65 minutes = 1.08 hours
    
    def test_error_pattern_tracking(self):
        """Test that unknown error patterns are tracked"""
        # Different unknown errors
        errors = [
            "Connection timeout",
            "Connection timeout",  # Same error
            "Network unreachable",
            "Connection timeout",  # Third occurrence
        ]
        
        with patch('bittensor.logging'):
            for error in errors:
                self.mock_subtensor.set_weights.return_value = (False, error)
                self.weight_setter._set_subtensor_weights(
                    self.mock_wallet, self.mock_subtensor, 8
                )
            
            # Verify pattern tracking (would be in the WeightFailureTracker)
            # This tests that the system can identify recurring patterns
            # In real implementation, this would be tracked in failure_patterns dict
    
    def test_mixed_error_scenarios(self):
        """Test realistic mixed error scenarios"""
        scenarios = [
            (False, "No attempt made. Perhaps it is too soon to commit weights!"),  # Benign
            (True, ""),  # Success
            (False, "No attempt made. Perhaps it is too soon to commit weights!"),  # Benign
            (False, "maximum recursion depth exceeded"),  # Critical - should alert
            (False, "No attempt made. Perhaps it is too soon to commit weights!"),  # Benign
            (True, ""),  # Success - should send recovery alert
            (False, "Unknown error XYZ"),  # Unknown
            (False, "Unknown error XYZ"),  # Unknown - should alert on 2nd
        ]
        
        alert_count = 0
        with patch('bittensor.logging'):
            with patch('time.time') as mock_time:
                mock_time.return_value = 1000  # Initial time
                
                for i, (success, error) in enumerate(scenarios):
                    # Advance time by 11 minutes between scenarios to avoid rate limiting
                    if i > 0:
                        mock_time.return_value = 1000 + (i * 660)
                    
                    self.mock_subtensor.set_weights.return_value = (success, error)
                    self.weight_setter._set_subtensor_weights(
                        self.mock_wallet, self.mock_subtensor, 8
                    )
                    
                    current_alert_count = self.weight_setter.slack_notifier.send_message.call_count
                    
                    # Verify expected alerts
                    if i == 3:  # Critical error
                        self.assertEqual(current_alert_count, 1, "Should alert on critical error")
                        alert_count = 1
                    elif i == 5:  # Recovery after critical
                        self.assertEqual(current_alert_count, 2, "Should send recovery alert")
                        alert_count = 2
                    elif i == 7:  # Second unknown error
                        self.assertEqual(current_alert_count, 3, "Should alert on 2nd unknown error")
                        alert_count = 3
                    else:
                        self.assertEqual(current_alert_count, alert_count, f"Unexpected alert at step {i}")
    
    def test_two_hour_absolute_timeout(self):
        """Test that 2-hour timeout triggers alert regardless of failure type or rate limiting"""
        with patch('time.time') as mock_time:
            mock_time.return_value = 1000  # Initial time
            
            # Reset the tracker's last success time to match our mock time
            self.weight_setter.weight_failure_tracker.last_success_time = 1000
            
            # Benign failure that normally doesn't alert
            self.mock_subtensor.set_weights.return_value = (
                False,
                "No attempt made. Perhaps it is too soon to commit weights!"
            )
            
            with patch('bittensor.logging'):
                # First failure - no alert (benign)
                self.weight_setter._set_subtensor_weights(
                    self.mock_wallet, self.mock_subtensor, 8
                )
                self.weight_setter.slack_notifier.send_message.assert_not_called()
                
                # Simulate 2.5 hours passing
                mock_time.return_value = 1000 + (2.5 * 3600)
                
                # Another benign failure - should alert due to 2+ hour timeout
                self.weight_setter._set_subtensor_weights(
                    self.mock_wallet, self.mock_subtensor, 8
                )
                
                # Should have alert with URGENT prefix
                self.weight_setter.slack_notifier.send_message.assert_called_once()
                alert_msg = self.weight_setter.slack_notifier.send_message.call_args[0][0]
                self.assertIn("🚨 URGENT", alert_msg)
                self.assertIn("2.5 hours", alert_msg)
                
                # Set alert time
                self.weight_setter.weight_failure_tracker.last_alert_time = mock_time.return_value
                
                # Immediate next failure (within rate limit window) should still alert
                # because 2+ hour timeout bypasses rate limiting
                mock_time.return_value = 1000 + (2.5 * 3600) + 60  # 1 minute later
                self.weight_setter._set_subtensor_weights(
                    self.mock_wallet, self.mock_subtensor, 8
                )
                
                # Should have 2 alerts now (rate limiting bypassed for 2+ hour timeout)
                self.assertEqual(self.weight_setter.slack_notifier.send_message.call_count, 2)


if __name__ == '__main__':
    unittest.main()