#!/usr/bin/env python3
# Quick test script to verify weight setting alerts work

from unittest.mock import Mock, patch
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vali_objects.utils.subtensor_weight_setter import SubtensorWeightSetter, WeightFailureTracker

def test_failure_classification():
    """Test the failure classification logic"""
    tracker = WeightFailureTracker()
    
    # Test benign errors
    benign_errors = [
        "No attempt made. Perhaps it is too soon to commit weights!",
        "Failed: too soon to commit weights",
        "Error: TOO SOON TO COMMIT"
    ]
    
    for error in benign_errors:
        result = tracker.classify_failure(error)
        print(f"Error: '{error}' -> Classification: {result}")
        assert result == "benign", f"Expected benign for '{error}', got {result}"
    
    # Test critical errors
    critical_errors = [
        "maximum recursion depth exceeded in comparison",
        "Subtensor returned: Invalid Transaction",
        "Error: Invalid transaction detected"
    ]
    
    for error in critical_errors:
        result = tracker.classify_failure(error)
        print(f"Error: '{error}' -> Classification: {result}")
        assert result == "critical", f"Expected critical for '{error}', got {result}"
    
    # Test unknown errors
    unknown_errors = [
        "Connection timeout",
        "Network unreachable",
        "Some other error"
    ]
    
    for error in unknown_errors:
        result = tracker.classify_failure(error)
        print(f"Error: '{error}' -> Classification: {result}")
        assert result == "unknown", f"Expected unknown for '{error}', got {result}"
    
    print("\n✅ Failure classification tests passed!")


def test_alert_logic():
    """Test the alert decision logic"""
    import time
    tracker = WeightFailureTracker()
    
    # Benign should never alert (within timeout window)
    assert not tracker.should_alert("benign", 1)
    assert not tracker.should_alert("benign", 100)
    print("✅ Benign errors never alert (within timeout)")
    
    # Critical should always alert
    assert tracker.should_alert("critical", 1)
    assert tracker.should_alert("critical", 2)
    print("✅ Critical errors always alert")
    
    # Unknown should alert after 2 consecutive
    assert not tracker.should_alert("unknown", 1)
    assert tracker.should_alert("unknown", 2)
    assert tracker.should_alert("unknown", 3)
    print("✅ Unknown errors alert after 2 consecutive")
    
    # Test 2-hour timeout override
    original_time = time.time()
    tracker.last_success_time = original_time - (2.5 * 3600)  # 2.5 hours ago
    assert tracker.should_alert("benign", 1)  # Even benign errors alert after 2 hours
    print("✅ 2-hour timeout triggers alert regardless of error type")
    
    # Test 1-hour timeout for non-benign errors
    tracker.last_success_time = original_time - (1.5 * 3600)  # 1.5 hours ago
    assert tracker.should_alert("unknown", 1)  # Unknown errors alert after 1 hour
    print("✅ 1-hour timeout triggers alert for non-benign errors")
    
    print("\n✅ Alert logic tests passed!")


def test_weight_setter_integration():
    """Test the integration with SubtensorWeightSetter"""
    # Create mocks
    mock_metagraph = Mock()
    mock_position_manager = Mock()
    mock_position_manager.challengeperiod_manager = Mock()
    mock_position_manager.challengeperiod_manager.get_hotkeys_by_bucket.return_value = []
    mock_slack = Mock()
    
    # Create weight setter
    weight_setter = SubtensorWeightSetter(
        metagraph=mock_metagraph,
        position_manager=mock_position_manager,
        slack_notifier=mock_slack
    )
    
    # Set up config and wallet
    weight_setter.config = Mock()
    weight_setter.config.netuid = 8
    weight_setter.config.subtensor.network = "finney"
    weight_setter.wallet = Mock()
    weight_setter.wallet.hotkey.ss58_address = "test_hotkey"
    
    # Set up test weights
    weight_setter.transformed_list = [(0, 1.0), (1, 0.5)]
    
    # Mock subtensor
    mock_subtensor = Mock()
    mock_wallet = Mock()
    
    # Test 1: Benign error - no alert
    mock_subtensor.set_weights.return_value = (False, "No attempt made. Perhaps it is too soon to commit weights!")
    
    with patch('bittensor.logging'):
        weight_setter._set_subtensor_weights(mock_wallet, mock_subtensor, 8)
    
    mock_slack.send_message.assert_not_called()
    print("✅ Benign error did not trigger alert")
    
    # Test 2: Critical error - immediate alert
    mock_subtensor.set_weights.return_value = (False, "maximum recursion depth exceeded")
    
    with patch('bittensor.logging'):
        weight_setter._set_subtensor_weights(mock_wallet, mock_subtensor, 8)
    
    mock_slack.send_message.assert_called_once()
    alert_msg = mock_slack.send_message.call_args[0][0]
    assert "CRITICAL" in alert_msg
    assert "recursion" in alert_msg.lower()
    print("✅ Critical error triggered immediate alert")
    
    # Test 3: Success after critical - recovery alert
    mock_slack.reset_mock()
    mock_subtensor.set_weights.return_value = (True, "")
    
    with patch('bittensor.logging'):
        weight_setter._set_subtensor_weights(mock_wallet, mock_subtensor, 8)
    
    mock_slack.send_message.assert_called_once()
    recovery_msg = mock_slack.send_message.call_args[0][0]
    assert "recovered" in recovery_msg.lower()
    print("✅ Recovery alert sent after critical failure")
    
    print("\n✅ Weight setter integration tests passed!")


if __name__ == "__main__":
    print("Testing Weight Setting Alert System\n")
    print("=" * 50)
    
    test_failure_classification()
    print("\n" + "=" * 50 + "\n")
    
    test_alert_logic()
    print("\n" + "=" * 50 + "\n")
    
    test_weight_setter_integration()
    
    print("\n" + "=" * 50)
    print("✅ All tests passed successfully!")
    print("=" * 50)