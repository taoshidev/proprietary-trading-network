"""
Debt-Based Scoring

This module computes miner weights based on debt ledger information.
The algorithm pays miners based on their previous month's performance (PnL scaled by penalties),
proportionally distributing emissions to cover remaining debt over the days left in the current month.

Key Concepts:
- "Needed payout" = What miners earned in previous month (PnL * penalties)
- "Actual payout" = What they've been paid so far in current month (ALPHA emissions)
- "Remaining payout" = needed_payout - actual_payout
- Weights = Rate of emissions needed to cover remaining payout over remaining days

Important Notes:
- Debt-based scoring only activates starting November 2025
- Before November 2025, all miners get zero weights
- Checkpoints are 12-hour intervals (2 per day)
"""

import bittensor as bt
from datetime import datetime, timezone
from typing import List, Tuple
from calendar import monthrange

from time_util.time_util import TimeUtil
from vali_objects.vali_dataclasses.debt_ledger import DebtLedger


class DebtBasedScoring:
    """
    Debt-based scoring system that pays miners proportionally to their previous month's
    performance, distributing remaining payments over the days left in the current month.
    """

    # Activation date: November 2025
    ACTIVATION_YEAR = 2025
    ACTIVATION_MONTH = 11

    @staticmethod
    def compute_results(
        ledger_dict: dict[str, DebtLedger],
        current_time_ms: int = None,
        verbose: bool = False
    ) -> List[Tuple[str, float]]:
        """
        Compute miner weights based on debt ledger information.

        The algorithm works as follows:
        1. Check if we're in activation period (>= November 2025)
        2. For each miner, calculate their "needed payout" from previous month's performance
        3. Calculate "actual payout" given so far in current month
        4. Calculate "remaining payout" to be distributed
        5. Compute target emission rate to cover remaining payout over remaining days
        6. Normalize weights so they sum to 1.0

        Args:
            ledger_dict: Dict of {hotkey: DebtLedger} containing debt ledger data
            current_time_ms: Current timestamp in milliseconds (defaults to now)
            verbose: Enable detailed logging

        Returns:
            List of (hotkey, weight) tuples sorted by weight (descending)
        """
        if current_time_ms is None:
            current_time_ms = TimeUtil.now_in_millis()

        # Handle edge cases
        if not ledger_dict:
            bt.logging.info("No debt ledgers provided, returning empty weights")
            return []

        if len(ledger_dict) == 1:
            hotkey = list(ledger_dict.keys())[0]
            bt.logging.info(f"Only one miner: {hotkey}, returning weight 1.0")
            return [(hotkey, 1.0)]

        # Step 1: Get current month and year
        current_dt = TimeUtil.millis_to_datetime(current_time_ms)
        current_year = current_dt.year
        current_month = current_dt.month

        if verbose:
            bt.logging.info(
                f"Computing debt-based weights for {current_dt.strftime('%B %Y')} "
                f"({len(ledger_dict)} miners)"
            )

        # Step 2: Check if previous month is before November 2025
        # Calculate previous month
        if current_month == 1:
            prev_month = 12
            prev_year = current_year - 1
        else:
            prev_month = current_month - 1
            prev_year = current_year

        if verbose:
            bt.logging.info(f"Previous month: {prev_year}-{prev_month:02d}")

        # Check activation date
        if (prev_year < DebtBasedScoring.ACTIVATION_YEAR or
            (prev_year == DebtBasedScoring.ACTIVATION_YEAR and
             prev_month < DebtBasedScoring.ACTIVATION_MONTH)):
            bt.logging.info(
                f"Previous month ({prev_year}-{prev_month:02d}) is before activation "
                f"({DebtBasedScoring.ACTIVATION_YEAR}-{DebtBasedScoring.ACTIVATION_MONTH:02d}). "
                f"Returning zero weights for all miners."
            )
            return [(hotkey, 0.0) for hotkey in ledger_dict.keys()]

        # Step 3: Calculate month boundaries
        # Previous month: full month
        prev_month_start_dt = datetime(prev_year, prev_month, 1, 0, 0, 0, tzinfo=timezone.utc)
        prev_month_days = monthrange(prev_year, prev_month)[1]  # Number of days in previous month
        prev_month_end_dt = datetime(prev_year, prev_month, prev_month_days, 23, 59, 59, 999999, tzinfo=timezone.utc)

        prev_month_start_ms = int(prev_month_start_dt.timestamp() * 1000)
        prev_month_end_ms = int(prev_month_end_dt.timestamp() * 1000)

        # Current month: from start of month to now
        current_month_start_dt = datetime(current_year, current_month, 1, 0, 0, 0, tzinfo=timezone.utc)
        current_month_start_ms = int(current_month_start_dt.timestamp() * 1000)

        if verbose:
            bt.logging.info(
                f"Previous month window: {prev_month_start_dt.strftime('%Y-%m-%d')} to "
                f"{prev_month_end_dt.strftime('%Y-%m-%d')}"
            )
            bt.logging.info(
                f"Current month elapsed: {current_month_start_dt.strftime('%Y-%m-%d')} to "
                f"{current_dt.strftime('%Y-%m-%d %H:%M:%S')}"
            )

        # Calculate days remaining in current month
        current_month_days = monthrange(current_year, current_month)[1]
        current_day = current_dt.day
        days_remaining = current_month_days - current_day + 1  # +1 to include today

        if verbose:
            bt.logging.info(
                f"Days in current month: {current_month_days}, "
                f"current day: {current_day}, "
                f"days remaining: {days_remaining}"
            )

        # Step 4-6: Process each miner
        miner_target_emissions = {}

        for hotkey, debt_ledger in ledger_dict.items():
            if not debt_ledger.checkpoints:
                if verbose:
                    bt.logging.debug(f"Skipping {hotkey}: no checkpoints")
                miner_target_emissions[hotkey] = 0.0
                continue

            # Extract checkpoints for previous month
            prev_month_checkpoints = [
                cp for cp in debt_ledger.checkpoints
                if prev_month_start_ms <= cp.timestamp_ms <= prev_month_end_ms
            ]

            # Extract checkpoints for current month (up to now)
            current_month_checkpoints = [
                cp for cp in debt_ledger.checkpoints
                if current_month_start_ms <= cp.timestamp_ms <= current_time_ms
            ]

            if verbose:
                bt.logging.debug(
                    f"{hotkey[:16]}...{hotkey[-8:]}: "
                    f"{len(prev_month_checkpoints)} prev month checkpoints, "
                    f"{len(current_month_checkpoints)} current month checkpoints"
                )

            # Step 4: Calculate needed payout from previous month
            # "needed payout" = sum of (net_pnl * total_penalty) for all checkpoints in previous month
            needed_payout = 0.0
            if prev_month_checkpoints:
                # Use the LAST checkpoint's cumulative values (not sum of chunks)
                last_prev_cp = prev_month_checkpoints[-1]
                needed_payout = last_prev_cp.net_pnl * last_prev_cp.total_penalty

            # Step 5: Calculate actual payout given so far in current month
            # "actual payout" = sum of chunk_emissions_alpha for current month
            actual_payout = sum(cp.chunk_emissions_alpha for cp in current_month_checkpoints)

            # Step 6: Calculate remaining payout
            remaining_payout = needed_payout - actual_payout

            # Clamp to zero if negative (over-paid or negative performance)
            if remaining_payout < 0:
                remaining_payout = 0.0

            # Calculate target emission rate per day
            if days_remaining > 0:
                target_emission_per_day = remaining_payout / days_remaining
            else:
                target_emission_per_day = 0.0

            miner_target_emissions[hotkey] = target_emission_per_day

            if verbose:
                bt.logging.debug(
                    f"{hotkey[:16]}...{hotkey[-8:]}: "
                    f"needed_payout={needed_payout:.6f}, "
                    f"actual_payout={actual_payout:.6f}, "
                    f"remaining={remaining_payout:.6f}, "
                    f"target_per_day={target_emission_per_day:.6f}"
                )

        # Step 7: Normalize weights
        # Weights must sum to 1.0 (this is the "rate" of emissions each miner gets)
        normalized_weights = DebtBasedScoring._normalize_scores(miner_target_emissions)

        # Convert to sorted list
        result = sorted(normalized_weights.items(), key=lambda x: x[1], reverse=True)

        if verbose:
            bt.logging.info(f"Debt-based weights computed for {len(result)} miners")
            if result:
                top_5 = result[:5]
                bt.logging.info("Top 5 miners:")
                for hotkey, weight in top_5:
                    bt.logging.info(f"  {hotkey[:16]}...{hotkey[-8:]}: {weight:.6f}")

        return result

    @staticmethod
    def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
        """
        Normalize scores so they sum to 1.0 (weights must sum to 1.0).

        Args:
            scores: Dict of {hotkey: score}

        Returns:
            Dict of {hotkey: normalized_weight}
        """
        if not scores:
            bt.logging.info("No scores to normalize, returning empty dict")
            return {}

        sum_scores = sum(scores.values())

        if sum_scores == 0:
            bt.logging.info("Sum of scores is 0, returning equal weights")
            # Return equal weights for all miners
            equal_weight = 1.0 / len(scores)
            return {hotkey: equal_weight for hotkey in scores.keys()}

        # Normalize
        normalized_scores = {
            hotkey: (score / sum_scores) for hotkey, score in scores.items()
        }

        return normalized_scores
