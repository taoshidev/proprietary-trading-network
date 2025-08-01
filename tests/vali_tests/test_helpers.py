"""
Helper functions for tests to handle ValiConfig values
"""

from vali_objects.vali_config import ValiConfig


def get_challenge_period_minimum_ms():
    """Get the challenge period minimum in milliseconds"""
    # CHALLENGE_PERIOD_MINIMUM_DAYS is an InterpolatedValueFromDate
    # For testing, we'll use the current value
    min_days = ValiConfig.CHALLENGE_PERIOD_MINIMUM_DAYS
    if hasattr(min_days, 'get_value'):
        min_days = min_days.get_value()
    elif hasattr(min_days, '__call__'):
        min_days = min_days()
    elif isinstance(min_days, (int, float)):
        pass  # Already a number
    else:
        # Default to 60 days if we can't determine
        min_days = 60
    
    return int(min_days * ValiConfig.DAILY_MS)