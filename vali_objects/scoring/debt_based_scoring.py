"""
Debt-Based Scoring

This module computes miner weights based on debt ledger information.
The algorithm pays miners based on their previous month's performance (PnL scaled by penalties),
proportionally distributing emissions to cover remaining debt by day 25 of the current month.

Key Concepts:
- "Needed payout" = What miners earned in previous month (PnL in USD * penalties)
- "Actual payout" = What they've been paid so far in current month (emissions in USD)
- "Remaining payout" = needed_payout_usd - actual_payout_usd (in USD)
- "Projected emissions" = Estimated total ALPHA available, converted to USD for comparison
- Weights = Proportional to remaining_payout_usd, with warning if insufficient emissions

Algorithm Flow:
1. Calculate needed_payout_usd from previous month's performance (only MAINCOMP/PROBATION checkpoints)
2. Calculate actual_payout_usd from current month's emissions (only MAINCOMP/PROBATION checkpoints)
3. Calculate remaining_payout_usd for each miner (in USD)
4. Query real-time TAO emission rate from subtensor
5. Convert to ALPHA, then convert ALPHA to USD using current conversion rates
6. Apply aggressive payout strategy (early month = 4-day horizon, late month = actual remaining)
7. Project total USD value available over aggressive timeline
8. Set weights proportional to remaining_payout_usd
9. Warn if sum(remaining_payouts_usd) > projected_usd_emissions
10. Enforce minimum weights based on challenge period status:
    - CHALLENGE/PLAGIARISM: 1x dust
    - PROBATION: 2x dust
    - MAINCOMP: 3x dust
    - UNKNOWN: 0x dust (no weight)
11. Normalize weights with burn address logic:
    - If sum < 1.0: assign (1.0 - sum) to burn address (uid 229 mainnet / uid 5 testnet)
    - If sum >= 1.0: normalize to 1.0, burn address gets 0

Aggressive Payout Strategy:
- Day 1-20: Target completion in 4 days (aggressive, creates urgency)
- Day 21-24: Target completion in actual remaining days (tapers off)
- Day 25: Final deadline
- This front-loads emissions early in the month while respecting the hard deadline

Important Notes:
- Debt-based scoring activates December 2025 (nominal payouts begin Dec 1)
- Before December 2025, miners only receive minimum dust weights
- Excess weight (when sum < 1.0) goes to burn address (uid 229 mainnet, uid 5 testnet)
- Hard deadline: day 25 of each month
- Checkpoints are 12-hour intervals (2 per day)
- Uses real-time subtensor queries for emission rate estimation
"""

import bittensor as bt
from datetime import datetime, timezone
from typing import List, Tuple, Optional
from calendar import monthrange

from time_util.time_util import TimeUtil
from vali_objects.vali_dataclasses.debt_ledger import DebtLedger
from vali_objects.utils.miner_bucket_enum import MinerBucket
from vali_objects.vali_config import ValiConfig
from vali_objects.scoring.scoring import Scoring
from collections import defaultdict


class DebtBasedScoring:
    """
    Debt-based scoring system that pays miners proportionally to their previous month's
    performance, targeting payout completion by day 25 of each month.

    Uses real-time subtensor queries to estimate emission rates and project available ALPHA.
    """

    # Activation date: December 2025 (nominal payouts begin Dec 1)
    ACTIVATION_YEAR = 2025
    ACTIVATION_MONTH = 12

    # Target payout completion by day 25
    PAYOUT_TARGET_DAY = 25

    # Aggressive payout buffer: aim to complete this many days from now (minimum)
    # This makes early-month payouts more aggressive (day 1 targets 4-day completion)
    # while tapering to actual remaining days as we approach the deadline
    AGGRESSIVE_PAYOUT_BUFFER_DAYS = 4

    # Bittensor network parameters (approximate, for fallback)
    BLOCKS_PER_DAY_FALLBACK = 7200  # ~12 seconds per block
    RAO_PER_TOKEN = 1e9

    # Burn address UIDs (receives excess weight when sum < 1.0)
    BURN_UID_MAINNET = 229
    BURN_UID_TESTNET = 220

    @staticmethod
    def get_burn_uid(is_testnet: bool = False) -> int:
        """
        Get the correct burn UID based on network (testnet vs mainnet).

        Args:
            is_testnet: True for testnet (netuid 116), False for mainnet (netuid 8)

        Returns:
            229 for mainnet, 5 for testnet
        """
        return DebtBasedScoring.BURN_UID_TESTNET if is_testnet else DebtBasedScoring.BURN_UID_MAINNET

    @staticmethod
    def calculate_dynamic_dust(
        metagraph: 'bt.metagraph',
        target_daily_usd: float = 0.01,
        verbose: bool = False
    ) -> float:
        """
        Calculate dynamic dust weight that yields target daily USD earnings.

        The calculation ensures that a miner receiving only dust weight will earn
        approximately target_daily_usd per day in ALPHA emissions.

        Formula:
            dust_weight = (ALPHA equivalent of target_daily_usd) / (total ALPHA per day)

        This provides market-responsive minimum rewards that automatically adjust as:
        - TAO/USD price changes
        - ALPHA/TAO conversion rate changes
        - Total subnet emission rate changes

        Args:
            metagraph: Shared IPC metagraph with emission data and substrate reserves
            target_daily_usd: Target daily USD earnings for dust weight (default: $0.01)
            verbose: Enable detailed logging

        Returns:
            Dynamic dust weight (unitless proportion)
            Falls back to ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT on any error

        Fallback Triggers:
            - Missing metagraph attributes (emission, reserves, tao_to_usd_rate)
            - Zero or negative values in any calculation step
            - Invalid conversion rates (reserves = 0, tao_to_usd_rate <= 0)
            - Dust weight outside reasonable range (0, 0.001]
            - Any exception during calculation
        """
        try:
            # Fallback detection: Check if metagraph has emission data
            if not hasattr(metagraph, 'emission') or metagraph.emission is None:
                bt.logging.warning(
                    "Metagraph missing 'emission' attribute. "
                    f"Falling back to static dust: {ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT}"
                )
                return ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT

            # Step 1: Calculate total ALPHA emissions per day
            try:
                total_tao_per_tempo = sum(metagraph.emission)  # TAO per tempo (360 blocks)
            except (TypeError, AttributeError) as e:
                bt.logging.warning(
                    f"Failed to sum metagraph.emission: {e}. "
                    f"Falling back to static dust: {ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT}"
                )
                return ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT

            # Fallback detection: Check for zero/negative emissions
            if total_tao_per_tempo <= 0:
                bt.logging.warning(
                    f"Total TAO per tempo is non-positive: {total_tao_per_tempo}. "
                    f"Falling back to static dust: {ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT}"
                )
                return ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT

            total_tao_per_block = total_tao_per_tempo / 360
            total_tao_per_day = total_tao_per_block * DebtBasedScoring.BLOCKS_PER_DAY_FALLBACK

            if verbose:
                bt.logging.info(f"Total subnet emissions: {total_tao_per_day:.6f} TAO/day")

            # Step 2: Get conversion rates from metagraph with comprehensive fallback detection
            tao_reserve_obj = getattr(metagraph, 'tao_reserve_rao', None)
            alpha_reserve_obj = getattr(metagraph, 'alpha_reserve_rao', None)

            # Fallback detection: Check for missing reserve attributes
            if tao_reserve_obj is None or alpha_reserve_obj is None:
                bt.logging.warning(
                    f"Substrate reserve attributes not found in metagraph "
                    f"(tao_reserve_rao={tao_reserve_obj is not None}, "
                    f"alpha_reserve_rao={alpha_reserve_obj is not None}). "
                    f"Falling back to static dust: {ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT}"
                )
                return ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT

            # Extract values with fallback detection
            try:
                tao_reserve_rao = tao_reserve_obj.value if hasattr(tao_reserve_obj, 'value') else float(tao_reserve_obj)
                alpha_reserve_rao = alpha_reserve_obj.value if hasattr(alpha_reserve_obj, 'value') else float(alpha_reserve_obj)
            except (AttributeError, TypeError, ValueError) as e:
                bt.logging.warning(
                    f"Failed to extract reserve values: {e}. "
                    f"Falling back to static dust: {ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT}"
                )
                return ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT

            # Fallback detection: Check for zero/negative reserves
            if tao_reserve_rao <= 0 or alpha_reserve_rao <= 0:
                bt.logging.warning(
                    f"Substrate reserve data not available or invalid for dynamic dust calculation "
                    f"(TAO={tao_reserve_rao} RAO, ALPHA={alpha_reserve_rao} RAO). "
                    f"Falling back to static dust: {ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT}"
                )
                return ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT

            # Calculate ALPHA-to-TAO rate
            alpha_to_tao_rate = tao_reserve_rao / alpha_reserve_rao

            # Fallback detection: Sanity check on conversion rate
            if alpha_to_tao_rate <= 0 or alpha_to_tao_rate > 1.0:
                bt.logging.warning(
                    f"ALPHA-to-TAO rate outside expected range (0, 1.0]: {alpha_to_tao_rate}. "
                    f"Falling back to static dust: {ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT}"
                )
                return ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT

            # Convert TAO/day to ALPHA/day
            total_alpha_per_day = total_tao_per_day / alpha_to_tao_rate

            if verbose:
                bt.logging.info(
                    f"Total subnet emissions: {total_alpha_per_day:.2f} ALPHA/day "
                    f"(conversion rate: {alpha_to_tao_rate:.6f} TAO/ALPHA)"
                )

            # Step 3: Get TAO/USD price with fallback detection
            tao_to_usd_rate_raw = getattr(metagraph, 'tao_to_usd_rate', None)

            # Fallback detection: Check for missing TAO/USD price
            if tao_to_usd_rate_raw is None:
                bt.logging.warning(
                    "TAO/USD price not available in metagraph for dynamic dust calculation. "
                    f"Falling back to static dust: {ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT}"
                )
                return ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT

            # Fallback detection: Validate TAO/USD price type and value
            try:
                tao_to_usd_rate = float(tao_to_usd_rate_raw)
            except (TypeError, ValueError) as e:
                bt.logging.warning(
                    f"TAO/USD price has invalid type: {type(tao_to_usd_rate_raw).__name__}, error: {e}. "
                    f"Falling back to static dust: {ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT}"
                )
                return ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT

            if tao_to_usd_rate <= 0:
                bt.logging.warning(
                    f"TAO/USD price is non-positive: {tao_to_usd_rate}. "
                    f"Falling back to static dust: {ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT}"
                )
                return ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT

            # Fallback detection: Sanity check on TAO price (should be between $1 and $10,000)
            if tao_to_usd_rate < 1.0 or tao_to_usd_rate > 10000.0:
                bt.logging.warning(
                    f"TAO/USD price outside reasonable range [$1, $10,000]: ${tao_to_usd_rate}. "
                    f"Falling back to static dust: {ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT}"
                )
                return ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT

            # Step 4: Calculate ALPHA equivalent of target USD amount
            target_in_tao = target_daily_usd / tao_to_usd_rate
            target_in_alpha = target_in_tao / alpha_to_tao_rate

            if verbose:
                bt.logging.info(
                    f"${target_daily_usd:.2f} USD = {target_in_tao:.6f} TAO = "
                    f"{target_in_alpha:.6f} ALPHA"
                )

            # Step 5: Calculate dust weight as proportion of daily emissions
            # Fallback detection: Check for zero/negative total emissions
            if total_alpha_per_day <= 0:
                bt.logging.warning(
                    f"Total ALPHA per day is non-positive: {total_alpha_per_day}. "
                    f"Falling back to static dust: {ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT}"
                )
                return ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT

            dust_weight = target_in_alpha / total_alpha_per_day

            if verbose:
                bt.logging.info(
                    f"Dynamic dust weight: {dust_weight:.8f} "
                    f"(yields ${target_daily_usd:.2f}/day at current emission rates)"
                )

            # Fallback detection: Sanity check on dust weight range
            # Should be small but not zero (typical range: 1e-8 to 1e-3)
            if dust_weight <= 0:
                bt.logging.warning(
                    f"Dynamic dust weight is non-positive: {dust_weight}. "
                    f"Falling back to static dust: {ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT}"
                )
                return ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT

            if dust_weight > 0.001:
                bt.logging.warning(
                    f"Dynamic dust weight ({dust_weight:.8f}) exceeds reasonable maximum (0.001). "
                    f"This suggests anomalous market conditions. "
                    f"Falling back to static dust: {ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT}"
                )
                return ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT

            # Success! Return dynamic dust weight
            return dust_weight

        except Exception as e:
            # Fallback detection: Catch-all for any unexpected errors
            bt.logging.error(
                f"Unexpected error calculating dynamic dust: {e}. "
                f"Falling back to static dust: {ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT}",
                exc_info=True
            )
            return ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT

    @staticmethod
    def log_projections(metagraph, days_until_target, verbose, total_remaining_payout_usd):
        # Query current emission rate and project availability
        # Get projected ALPHA emissions
        projected_alpha_available = DebtBasedScoring._estimate_alpha_emissions_until_target(
            metagraph=metagraph,
            days_until_target=days_until_target,
            verbose=verbose
        )

        # Convert projected ALPHA to USD for comparison
        projected_usd_available = DebtBasedScoring._convert_alpha_to_usd(
            alpha_amount=projected_alpha_available,
            metagraph=metagraph,
            verbose=verbose
        )

        if verbose:
            bt.logging.info(
                f"Projected emissions: {projected_alpha_available:.2f} ALPHA "
                f"≈ ${projected_usd_available:.2f} USD"
            )

        # Check if projected emissions (in USD) are sufficient
        if projected_usd_available < total_remaining_payout_usd:
            shortage_pct = ((
                                        total_remaining_payout_usd - projected_usd_available) / total_remaining_payout_usd) * 100
            bt.logging.warning(
                f"⚠️  INSUFFICIENT EMISSIONS: Projected USD value in next {days_until_target} days "
                f"(${projected_usd_available:.2f}) is less than total remaining payout needed "
                f"(${total_remaining_payout_usd:.2f}). Shortage: {shortage_pct:.1f}%. "
                f"Using aggressive {days_until_target}-day payout strategy (target day {DebtBasedScoring.PAYOUT_TARGET_DAY}). "
                f"Miners will receive proportional payouts."
            )
        else:
            surplus_pct = ((projected_usd_available - total_remaining_payout_usd) / total_remaining_payout_usd) * 100
            bt.logging.info(
                f"✓ Projected USD value in next {days_until_target} days (${projected_usd_available:.2f}) exceeds "
                f"total remaining payout needed (${total_remaining_payout_usd:.2f}). "
                f"Surplus: {surplus_pct:.1f}%. "
                f"Using aggressive {days_until_target}-day payout strategy (actual deadline: day {DebtBasedScoring.PAYOUT_TARGET_DAY})."
            )

    @staticmethod
    def compute_results(
        ledger_dict: dict[str, DebtLedger],
        metagraph: 'bt.metagraph',
        challengeperiod_manager: 'ChallengePeriodManager',
        current_time_ms: int = None,
        verbose: bool = False,
        is_testnet: bool = False
    ) -> List[Tuple[str, float]]:
        """
        Compute miner weights based on debt ledger information with real-time emission projections.

        The algorithm works as follows:
        1. Check if we're in activation period (>= December 2025)
        2. For each miner, calculate their "needed payout" from previous month's performance
        3. Calculate "actual payout" given so far in current month
        4. Calculate "remaining payout" to be distributed
        5. Query real-time TAO emission rate from metagraph
        6. Convert to ALPHA using reserve data from shared metagraph (TAO/ALPHA ratio)
        7. Project total ALPHA available from now until day 25
        8. Set weights proportional to remaining_payout
        9. Warn if sum(remaining_payouts) > projected_emissions
        10. Enforce minimum weights with dynamic dust (performance-scaled by 30-day PnL)
        11. Normalize weights with burn address logic (sum < 1.0 → burn gets excess)

        Args:
            ledger_dict: Dict of {hotkey: DebtLedger} containing debt ledger data
            metagraph: Shared IPC metagraph with emission data and substrate reserves
            challengeperiod_manager: Manager for querying current challenge period status (required)
            current_time_ms: Current timestamp in milliseconds (defaults to now)
            verbose: Enable detailed logging
            is_testnet: True for testnet (netuid 116), False for mainnet (netuid 8)

        Returns:
            List of (hotkey, weight) tuples sorted by weight (descending)
            Includes burn address (uid 229 mainnet / uid 5 testnet) if sum of weights < 1.0

        Note:
            Dynamic dust weights are always enabled. Miners receive dust weights scaled by
            30-day penalty-adjusted PnL within their bucket:
            floor = original dust multiplier, ceiling = floor + 1 DUST
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
                f"Applying only minimum dust weights, excess goes to burn address."
            )
            # Before activation: apply minimum dust weights only, burn the rest
            return DebtBasedScoring._apply_pre_activation_weights(
                ledger_dict=ledger_dict,
                metagraph=metagraph,
                challengeperiod_manager=challengeperiod_manager,
                current_time_ms=current_time_ms,
                is_testnet=is_testnet,
                verbose=verbose
            )

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

        # Calculate days until target payout day (day 25)
        current_day = current_dt.day

        if current_day > DebtBasedScoring.PAYOUT_TARGET_DAY:
            # Past target day, treat as 0 days remaining (will warn about insufficient time)
            actual_days_until_target = 0
        else:
            actual_days_until_target = DebtBasedScoring.PAYOUT_TARGET_DAY - current_day + 1  # +1 to include today

        # Apply aggressive payout strategy:
        # Early in month: Use shorter time horizon (e.g., 4 days) to be more aggressive
        # Late in month: Use actual remaining days as we approach deadline
        # This creates urgency early while respecting the hard deadline
        days_until_target = min(actual_days_until_target, DebtBasedScoring.AGGRESSIVE_PAYOUT_BUFFER_DAYS)

        # Ensure at least 1 day if we haven't reached deadline yet
        if actual_days_until_target > 0 and days_until_target == 0:
            days_until_target = 1

        if verbose:
            bt.logging.info(
                f"Current day: {current_day}, "
                f"target day: {DebtBasedScoring.PAYOUT_TARGET_DAY}, "
                f"actual days until target: {actual_days_until_target}, "
                f"aggressive days until target: {days_until_target}"
            )

        # Step 4-6: Process each miner to calculate remaining payouts (in USD)
        miner_remaining_payouts_usd = {}

        for hotkey, debt_ledger in ledger_dict.items():
            if not debt_ledger.checkpoints:
                if verbose:
                    bt.logging.debug(f"Skipping {hotkey}: no checkpoints")
                miner_remaining_payouts_usd[hotkey] = 0.0
                continue

            # Extract checkpoints for previous month
            # Only include checkpoints where status is MAINCOMP or PROBATION (earning periods)
            prev_month_checkpoints = [
                cp for cp in debt_ledger.checkpoints
                if prev_month_start_ms <= cp.timestamp_ms <= prev_month_end_ms
                and cp.challenge_period_status in (MinerBucket.MAINCOMP.value, MinerBucket.PROBATION.value)
            ]

            # Extract checkpoints for current month (up to now)
            # Only include checkpoints where status is MAINCOMP or PROBATION (earning periods)
            current_month_checkpoints = [
                cp for cp in debt_ledger.checkpoints
                if current_month_start_ms <= cp.timestamp_ms <= current_time_ms
                and cp.challenge_period_status in (MinerBucket.MAINCOMP.value, MinerBucket.PROBATION.value)
            ]

            if verbose:
                bt.logging.debug(
                    f"{hotkey[:16]}...{hotkey[-8:]}: "
                    f"{len(prev_month_checkpoints)} prev month checkpoints, "
                    f"{len(current_month_checkpoints)} current month checkpoints"
                )

            # Step 4: Calculate needed payout from previous month (in USD)
            # "needed payout" = sum of (net_pnl * total_penalty) across all prev month checkpoints
            # NOTE: net_pnl is in USD, pnl_gain/pnl_loss are per-checkpoint values (NOT cumulative)
            needed_payout_usd = 0.0
            if prev_month_checkpoints:
                # Sum penalty-adjusted PnL across all checkpoints in the month
                # Each checkpoint has its own PnL (for that 12-hour period) and its own penalty
                needed_payout_usd = sum(cp.net_pnl * cp.total_penalty for cp in prev_month_checkpoints)

            # Step 5: Calculate actual payout given so far in current month (in USD)
            # "actual payout" = sum of chunk_emissions_usd for current month
            actual_payout_usd = sum(cp.chunk_emissions_usd for cp in current_month_checkpoints)

            # Step 6: Calculate remaining payout (in USD)
            remaining_payout_usd = needed_payout_usd - actual_payout_usd

            # Clamp to zero if negative (over-paid or negative performance)
            if remaining_payout_usd < 0:
                remaining_payout_usd = 0.0

            miner_remaining_payouts_usd[hotkey] = remaining_payout_usd

            if verbose:
                bt.logging.debug(
                    f"{hotkey[:16]}...{hotkey[-8:]}: "
                    f"needed_payout_usd=${needed_payout_usd:.2f}, "
                    f"actual_payout_usd=${actual_payout_usd:.2f}, "
                    f"remaining_usd=${remaining_payout_usd:.2f}"
                )

        # Step 7-9: Query real-time emissions and project availability (in USD)
        total_remaining_payout_usd = sum(miner_remaining_payouts_usd.values())

        if total_remaining_payout_usd > 0 and days_until_target > 0:
            DebtBasedScoring.log_projections(metagraph, days_until_target, verbose, total_remaining_payout_usd)
        else:
            bt.logging.info(
                f"No remaining payouts needed {total_remaining_payout_usd} or no days until target "
                f"{days_until_target}, skipping projection log"
            )

        # Step 10: Enforce minimum weights based on challenge period status
        # All miners get minimum "dust" weights based on their current status
        # Dust is calculated dynamically to yield $0.01/day in emissions
        # Weights are dynamically scaled by 30-day performance within each bucket
        # NOTE: Weights are unitless proportions, but derived from USD payouts
        miner_weights_with_minimums = DebtBasedScoring._apply_minimum_weights(
            ledger_dict=ledger_dict,
            miner_remaining_payouts_usd=miner_remaining_payouts_usd,
            challengeperiod_manager=challengeperiod_manager,
            metagraph=metagraph,
            current_time_ms=current_time_ms,
            verbose=verbose
        )

        # Step 11: Normalize weights with special burn address logic
        # If sum < 1.0: assign remaining weight to burn address (uid 229 / uid 5)
        # If sum >= 1.0: normalize to 1.0, burn address gets 0
        result = DebtBasedScoring._normalize_with_burn_address(
            weights=miner_weights_with_minimums,
            metagraph=metagraph,
            is_testnet=is_testnet,
            verbose=verbose
        )

        if verbose:
            bt.logging.info(f"Debt-based weights computed for {len(result)} miners")
            if result:
                top_5 = result[:5]
                bt.logging.info("Top 5 miners:")
                for hotkey, weight in top_5:
                    bt.logging.info(f"  {hotkey[:16]}...{hotkey[-8:]}: {weight:.6f}")

        return result

    @staticmethod
    def _estimate_alpha_emissions_until_target(
        metagraph: 'bt.metagraph',
        days_until_target: int,
        verbose: bool = False
    ) -> float:
        """
        Estimate total ALPHA emissions available from now until target day.

        Uses real-time metagraph data to get current TAO emission rate,
        then converts to ALPHA using reserve data from shared metagraph.

        Args:
            metagraph: Shared IPC metagraph with emission data and substrate reserves
            days_until_target: Number of days until target payout day
            verbose: Enable detailed logging

        Returns:
            Estimated total ALPHA emissions available (float)
        """
        try:
            # Get total TAO emission per block for the subnet (sum across all miners)
            # metagraph.emission is already in TAO (not RAO), but per tempo (360 blocks)
            # Need to convert: per-tempo → per-block (÷360)
            total_tao_per_tempo = sum(metagraph.emission)
            total_tao_per_block = total_tao_per_tempo / 360

            if verbose:
                bt.logging.info(f"Current subnet emission rate: {total_tao_per_block:.6f} TAO/block")

            # Estimate blocks until target day
            # Use approximate 12 seconds per block (7200 blocks/day)
            blocks_until_target = days_until_target * DebtBasedScoring.BLOCKS_PER_DAY_FALLBACK

            # Calculate total TAO emissions until target
            total_tao_until_target = total_tao_per_block * blocks_until_target

            if verbose:
                bt.logging.info(
                    f"Estimated blocks until day {DebtBasedScoring.PAYOUT_TARGET_DAY}: {blocks_until_target}, "
                    f"total TAO: {total_tao_until_target:.2f}"
                )

            # Get substrate reserves from shared metagraph (refreshed by MetagraphUpdater)
            # Use .value accessor for manager.Value() objects (thread-safe IPC)
            tao_reserve_obj = getattr(metagraph, 'tao_reserve_rao', None)
            alpha_reserve_obj = getattr(metagraph, 'alpha_reserve_rao', None)

            tao_reserve_rao = tao_reserve_obj.value if tao_reserve_obj else 0.0
            alpha_reserve_rao = alpha_reserve_obj.value if alpha_reserve_obj else 0.0

            if tao_reserve_rao == 0 or alpha_reserve_rao == 0:
                bt.logging.warning(
                    "Substrate reserve data not available in shared metagraph "
                    f"(TAO={tao_reserve_rao} RAO, ALPHA={alpha_reserve_rao} RAO). "
                    "Cannot calculate ALPHA conversion rate."
                )
                return 0.0

            # Calculate ALPHA-to-TAO conversion rate from reserve data
            # alpha_to_tao_rate = tao_reserve / alpha_reserve (both in RAO, ratio is unitless)
            # (How much TAO per ALPHA)
            alpha_to_tao_rate = tao_reserve_rao / alpha_reserve_rao

            if verbose:
                bt.logging.info(
                    f"Substrate reserves: TAO={tao_reserve_rao / 1e9:.2f} TAO ({tao_reserve_rao:.0f} RAO), "
                    f"ALPHA={alpha_reserve_rao / 1e9:.2f} ALPHA ({alpha_reserve_rao:.0f} RAO), "
                    f"rate={alpha_to_tao_rate:.6f} TAO/ALPHA"
                )

            # Convert TAO to ALPHA
            # If ALPHA costs X TAO per ALPHA, then Y TAO buys Y/X ALPHA
            if alpha_to_tao_rate > 0:
                total_alpha_until_target = total_tao_until_target / alpha_to_tao_rate
            else:
                bt.logging.warning("ALPHA-to-TAO rate is zero, cannot convert")
                return 0.0

            if verbose:
                bt.logging.info(f"Projected ALPHA available until target: {total_alpha_until_target:.2f}")

            return total_alpha_until_target

        except Exception as e:
            bt.logging.error(f"Error estimating ALPHA emissions: {e}")
            raise

    @staticmethod
    def _convert_alpha_to_usd(
        alpha_amount: float,
        metagraph: 'bt.metagraph',
        verbose: bool = False
    ) -> float:
        """
        Convert ALPHA amount to USD value using current market rates.

        Uses reserve data from shared metagraph to calculate conversion rate:
        ALPHA → TAO (via reserves) → USD (via TAO price oracle)

        Args:
            alpha_amount: Amount of ALPHA tokens to convert
            metagraph: Shared IPC metagraph with substrate reserves
            verbose: Enable detailed logging

        Returns:
            USD value of the ALPHA amount (float)
        """
        if alpha_amount == 0:
            return 0.0

        # Get substrate reserves from shared metagraph
        tao_reserve_obj = getattr(metagraph, 'tao_reserve_rao', None)
        alpha_reserve_obj = getattr(metagraph, 'alpha_reserve_rao', None)

        tao_reserve_rao = tao_reserve_obj.value if tao_reserve_obj else 0.0
        alpha_reserve_rao = alpha_reserve_obj.value if alpha_reserve_obj else 0.0

        if tao_reserve_rao == 0 or alpha_reserve_rao == 0:
            bt.logging.warning(
                "Substrate reserve data not available for ALPHA→USD conversion. "
                f"(TAO={tao_reserve_rao} RAO, ALPHA={alpha_reserve_rao} RAO)"
            )
            return 0.0

        # Calculate ALPHA→TAO conversion rate
        # alpha_to_tao_rate = how much TAO per ALPHA
        alpha_to_tao_rate = tao_reserve_rao / alpha_reserve_rao

        # Convert ALPHA to TAO
        tao_amount = alpha_amount * alpha_to_tao_rate

        # Get TAO→USD price from metagraph
        # This is set by MetagraphUpdater via live_price_fetcher.get_close_at_date(TradePair.TAOUSD)
        tao_to_usd_rate_raw = getattr(metagraph, 'tao_to_usd_rate', None)

        # Validate that we have a valid TAO/USD rate
        if tao_to_usd_rate_raw is None:
            raise ValueError(
                "TAO/USD price not available in metagraph. "
                "MetagraphUpdater must set metagraph.tao_to_usd_rate via live_price_fetcher."
            )

        if not isinstance(tao_to_usd_rate_raw, (int, float)) or tao_to_usd_rate_raw <= 0:
            raise ValueError(
                f"Invalid TAO/USD price in metagraph: {tao_to_usd_rate_raw}. "
                f"Expected positive number, got {type(tao_to_usd_rate_raw).__name__}."
            )

        tao_to_usd_rate = float(tao_to_usd_rate_raw)

        # Convert TAO to USD
        usd_amount = tao_amount * tao_to_usd_rate

        if verbose:
            bt.logging.debug(
                f"ALPHA→USD conversion: {alpha_amount:.2f} ALPHA "
                f"→ {tao_amount:.6f} TAO "
                f"→ ${usd_amount:.2f} USD "
                f"(rates: {alpha_to_tao_rate:.6f} TAO/ALPHA, ${tao_to_usd_rate:.2f}/TAO)"
            )

        return usd_amount



    @staticmethod
    def _calculate_penalty_adjusted_pnl(
        ledger: DebtLedger,
        start_time_ms: int,
        end_time_ms: int,
        earning_statuses: set[int] = None
    ) -> float:
        """
        Calculate penalty-adjusted PnL for a time period (in USD).

        This is the SINGLE SOURCE OF TRUTH for PnL calculations,
        used by both main scoring and dynamic dust weight calculations.

        NOTE: net_pnl in checkpoints is in USD (performance value),
        so the return value is also in USD.

        Args:
            ledger: Miner's debt ledger
            start_time_ms: Period start (inclusive)
            end_time_ms: Period end (inclusive)
            earning_statuses: Set of statuses to include (default: MAINCOMP, PROBATION)

        Returns:
            Penalty-adjusted PnL for the period in USD (sum of net_pnl * total_penalty)
        """
        # Default to earning statuses
        if earning_statuses is None:
            earning_statuses = {
                MinerBucket.MAINCOMP.value,
                MinerBucket.PROBATION.value
            }

        if not ledger.checkpoints:
            return 0.0

        # Filter checkpoints within time range and matching statuses
        relevant_checkpoints = [
            cp for cp in ledger.checkpoints
            if start_time_ms <= cp.timestamp_ms <= end_time_ms
            and cp.challenge_period_status in earning_statuses
        ]

        if not relevant_checkpoints:
            return 0.0

        # Sum penalty-adjusted PnL across all checkpoints in the time range
        # NOTE: pnl_gain/pnl_loss are per-checkpoint values (NOT cumulative), so we must sum
        # Each checkpoint has its own PnL (for that 12-hour period) and its own penalty
        penalty_adjusted_pnl = sum(cp.net_pnl * cp.total_penalty for cp in relevant_checkpoints)

        return penalty_adjusted_pnl

    @staticmethod
    def _calculate_dynamic_dust_weights(
        ledger_dict: dict[str, DebtLedger],
        challengeperiod_manager: 'ChallengePeriodManager',
        current_time_ms: int,
        base_dust: float,
        verbose: bool = False
    ) -> dict[str, float]:
        """
        Calculate performance-scaled dust weights for all miners.

        Process:
        1. Group miners by bucket
        2. For each bucket, calculate 30-day penalty-adjusted PnL (in USD) for all miners
        3. Normalize PnL within bucket to [0, 1] range
        4. Scale to [floor, ceiling] where ceiling = floor + base_dust

        This incentivizes recent performance while maintaining bucket hierarchy.

        NOTE: PnL values are in USD as calculated by _calculate_penalty_adjusted_pnl.

        Args:
            ledger_dict: All miner ledgers
            challengeperiod_manager: Bucket status manager
            current_time_ms: Current timestamp
            base_dust: Base dust value (ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT)
            verbose: Enable detailed logging

        Returns:
            Dict mapping hotkey -> dynamic_dust_weight (unitless proportion)
        """
        # Original dust floor multipliers (respecting existing system)
        BUCKET_DUST_FLOORS = {
            MinerBucket.CHALLENGE.value: 1,      # 1x dust floor
            MinerBucket.PROBATION.value: 2,      # 2x dust floor
            MinerBucket.MAINCOMP.value: 3,       # 3x dust floor
            MinerBucket.UNKNOWN.value: 0,        # 0x dust (no weight for unknown status)
            MinerBucket.PLAGIARISM.value: 1,     # 1x dust floor
        }

        dynamic_weights = {}
        thirty_days_ms = 30 * 24 * 60 * 60 * 1000
        lookback_start = current_time_ms - thirty_days_ms

        # Group miners by current bucket
        bucket_groups = defaultdict(list)
        for hotkey, ledger in ledger_dict.items():
            bucket = challengeperiod_manager.get_miner_bucket(hotkey).value
            bucket_groups[bucket].append((hotkey, ledger))

        if verbose:
            bt.logging.info(
                f"Dynamic dust: Processing {len(ledger_dict)} miners across "
                f"{len(bucket_groups)} buckets (30-day lookback)"
            )

        # Process each bucket independently
        for bucket, miners in bucket_groups.items():
            floor_multiplier = BUCKET_DUST_FLOORS.get(bucket, 1)
            floor = floor_multiplier * base_dust
            ceiling = floor + base_dust  # +1 DUST range above floor

            if verbose:
                bucket_name = MinerBucket(bucket).name if bucket in [b.value for b in MinerBucket] else "UNKNOWN"
                bt.logging.debug(
                    f"Dynamic dust bucket {bucket_name}: {len(miners)} miners, "
                    f"floor={floor:.8f}, ceiling={ceiling:.8f}"
                )

            # Calculate 30-day PnL for all miners in bucket
            # Use ALL statuses for lookback (not just earning statuses)
            # This rewards miners who performed well even in CHALLENGE period
            pnl_scores = {}
            all_statuses = {b.value for b in MinerBucket}

            for hotkey, ledger in miners:
                pnl = DebtBasedScoring._calculate_penalty_adjusted_pnl(
                    ledger,
                    start_time_ms=lookback_start,
                    end_time_ms=current_time_ms,
                    earning_statuses=all_statuses  # Consider all recent performance
                )
                # Floor at 0 (negative PnL doesn't reduce dust below floor)
                pnl_scores[hotkey] = max(0.0, pnl)

            # Normalize within bucket [0, 1]
            if pnl_scores:
                max_pnl = max(pnl_scores.values())

                if max_pnl > 0:
                    # Scale each miner's PnL to [0, 1] then map to [floor, ceiling]
                    for hotkey, pnl in pnl_scores.items():
                        normalized = pnl / max_pnl
                        # Scale to [floor, ceiling]
                        dynamic_weights[hotkey] = floor + (normalized * (ceiling - floor))

                        if verbose:
                            bt.logging.debug(
                                f"  {hotkey[:16]}...{hotkey[-8:]}: "
                                f"pnl_usd=${pnl:.2f}, norm={normalized:.4f}, "
                                f"weight={dynamic_weights[hotkey]:.8f}"
                            )
                else:
                    # All miners have 0 PnL -> all get floor
                    for hotkey in pnl_scores.keys():
                        dynamic_weights[hotkey] = floor

                    if verbose:
                        bt.logging.debug(f"  All miners have 0 PnL, assigning floor={floor:.8f}")

        if verbose:
            bt.logging.info(f"Dynamic dust weights calculated for {len(dynamic_weights)} miners")

        return dynamic_weights

    @staticmethod
    def _apply_minimum_weights(
        ledger_dict: dict[str, DebtLedger],
        miner_remaining_payouts_usd: dict[str, float],
        challengeperiod_manager: 'ChallengePeriodManager',
        metagraph: 'bt.metagraph',
        current_time_ms: int = None,
        verbose: bool = False
    ) -> dict[str, float]:
        """
        Enforce minimum weights based on challenge period status with dynamic dust scaling.

        All miners receive minimum "dust" weights based on their current status:
        - CHALLENGE/PLAGIARISM: 1x dust
        - PROBATION: 2x dust
        - MAINCOMP: 3x dust
        - UNKNOWN: 0x dust (no weight)

        Dust value is calculated dynamically to yield $0.01/day in emissions,
        automatically adjusting for TAO price, ALPHA conversion rate, and emission rate changes.
        Falls back to ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT if dynamic calculation fails.

        Dynamic dust scaling is always enabled: miners are scaled within bucket based on 30-day
        penalty-adjusted PnL (in USD), with range [floor, floor+1 DUST].

        Args:
            ledger_dict: Dict of {hotkey: DebtLedger}
            miner_remaining_payouts_usd: Dict of {hotkey: remaining_payout_usd} in USD
            challengeperiod_manager: Manager for querying current challenge period status (required)
            metagraph: Shared IPC metagraph (required for dynamic dust calculation)
            current_time_ms: Current timestamp (required for dynamic dust calculation)
            verbose: Enable detailed logging

        Returns:
            Dict of {hotkey: weight} with minimums applied (weights are unitless proportions)
        """
        # Calculate dynamic dust weight ($0.01/day target)
        # Falls back to static dust if metagraph data is unavailable
        DUST = DebtBasedScoring.calculate_dynamic_dust(
            metagraph=metagraph,
            target_daily_usd=0.01,
            verbose=verbose
        )

        # Calculate dynamic dust weights (always enabled)
        if current_time_ms is None:
            bt.logging.warning(
                "current_time_ms not provided. Falling back to static dust weights."
            )
            dynamic_dust_weights = None
        else:
            try:
                dynamic_dust_weights = DebtBasedScoring._calculate_dynamic_dust_weights(
                    ledger_dict=ledger_dict,
                    challengeperiod_manager=challengeperiod_manager,
                    current_time_ms=current_time_ms,
                    base_dust=DUST,
                    verbose=verbose
                )
                if verbose:
                    bt.logging.info("Using dynamic dust weights (30-day performance scaling)")
            except Exception as e:
                bt.logging.error(f"Error calculating dynamic dust weights: {e}. Falling back to static.")
                dynamic_dust_weights = None

        # Static minimum weights (fallback)
        status_to_minimum_weight = {
            MinerBucket.CHALLENGE.value: 1 * DUST,
            MinerBucket.PLAGIARISM.value: 1 * DUST,
            MinerBucket.UNKNOWN.value: 0 * DUST,  # 0x dust (no weight for unknown status)
            MinerBucket.PROBATION.value: 2 * DUST,
            MinerBucket.MAINCOMP.value: 3 * DUST,
        }

        # Batch read all statuses in one IPC call to minimize overhead
        miner_statuses = {
            hotkey: challengeperiod_manager.get_miner_bucket(hotkey).value
            for hotkey in ledger_dict.keys()
        }

        miner_weights_with_minimums = {}

        for hotkey, debt_ledger in ledger_dict.items():
            # Get debt-based weight (from remaining payout in USD)
            # Note: This is converted to unitless weight proportion later during normalization
            debt_weight = miner_remaining_payouts_usd.get(hotkey, 0.0)

            # Get current status from batch-loaded statuses
            current_status = miner_statuses.get(hotkey, MinerBucket.UNKNOWN.value)

            # Get minimum weight (dynamic or static)
            if dynamic_dust_weights is not None and hotkey in dynamic_dust_weights:
                minimum_weight = dynamic_dust_weights[hotkey]
            else:
                # Fallback to static dust weight
                minimum_weight = status_to_minimum_weight.get(current_status, 1 * DUST)

            # Apply max(debt_weight, minimum_weight)
            final_weight = max(debt_weight, minimum_weight)

            miner_weights_with_minimums[hotkey] = final_weight

            if verbose:
                bt.logging.debug(
                    f"{hotkey[:16]}...{hotkey[-8:]}: "
                    f"status={current_status}, "
                    f"debt_weight={debt_weight:.8f}, "
                    f"minimum={minimum_weight:.8f}, "
                    f"final={final_weight:.8f}"
                )

        return miner_weights_with_minimums

    @staticmethod
    def _get_burn_address_hotkey(
        metagraph: 'bt.metagraph',
        is_testnet: bool = False
    ) -> str:
        """
        Get the hotkey for the burn address.

        Args:
            metagraph: Bittensor metagraph for accessing hotkeys
            is_testnet: True for testnet (uid 5), False for mainnet (uid 229)

        Returns:
            Hotkey string for burn address (uid 229 mainnet / uid 5 testnet)
        """
        burn_uid = DebtBasedScoring.get_burn_uid(is_testnet)

        # Get hotkey for burn UID
        if burn_uid < len(metagraph.hotkeys):
            return metagraph.hotkeys[burn_uid]
        else:
            bt.logging.warning(
                f"Burn UID {burn_uid} not found in metagraph "
                f"(only {len(metagraph.hotkeys)} UIDs). Using placeholder."
            )
            return f"burn_uid_{burn_uid}"

    @staticmethod
    def _normalize_with_burn_address(
        weights: dict[str, float],
        metagraph: 'bt.metagraph',
        is_testnet: bool = False,
        verbose: bool = False
    ) -> List[Tuple[str, float]]:
        """
        Normalize weights with special burn address logic.

        If sum of weights < 1.0:
            - Assign remaining weight (1.0 - sum) to burn address (uid 229 mainnet / uid 5 testnet)
        If sum of weights >= 1.0:
            - Normalize all weights to sum to 1.0
            - Burn address gets 0 (not included)

        Args:
            weights: Dict of {hotkey: weight}
            metagraph: Bittensor metagraph for accessing hotkeys
            is_testnet: True for testnet (uid 5), False for mainnet (uid 229)
            verbose: Enable detailed logging

        Returns:
            List of (hotkey, weight) tuples sorted by weight (descending)
        """
        if not weights:
            bt.logging.info("No weights to normalize, returning empty list")
            return []

        sum_weights = sum(weights.values())

        if verbose:
            bt.logging.info(f"Sum of weights before normalization: {sum_weights:.6f}")

        burn_uid = DebtBasedScoring.get_burn_uid(is_testnet)

        if sum_weights < 1.0:
            # Excess weight goes to burn address
            burn_weight = 1.0 - sum_weights

            # Get burn address hotkey
            burn_hotkey = DebtBasedScoring._get_burn_address_hotkey(metagraph, is_testnet)

            bt.logging.info(
                f"Sum of weights ({sum_weights:.6f}) < 1.0. "
                f"Assigning {burn_weight:.6f} to burn address (uid {burn_uid})"
            )

            # Create result with original weights + burn address
            result = [(hotkey, weight) for hotkey, weight in weights.items()]
            result.append((burn_hotkey, burn_weight))

        else:
            # Sum >= 1.0: normalize to exactly 1.0
            bt.logging.info(
                f"Sum of weights ({sum_weights:.6f}) >= 1.0. "
                f"Normalizing to 1.0, burn address gets 0."
            )

            # Use standard normalization
            normalized_weights = Scoring.normalize_scores(weights)
            result = [(hotkey, weight) for hotkey, weight in normalized_weights.items()]

        # Sort by weight descending
        result = sorted(result, key=lambda x: x[1], reverse=True)

        return result

    @staticmethod
    def _apply_pre_activation_weights(
        ledger_dict: dict[str, DebtLedger],
        metagraph: 'bt.metagraph',
        challengeperiod_manager: 'ChallengePeriodManager',
        current_time_ms: int = None,
        is_testnet: bool = False,
        verbose: bool = False
    ) -> List[Tuple[str, float]]:
        """
        Apply weights for pre-activation period (before Dec 2025).

        During pre-activation, miners only receive minimum dust weights based on
        their challenge period status. Excess weight goes to burn address.
        Dynamic dust scaling is always enabled.

        Args:
            ledger_dict: Dict of {hotkey: DebtLedger}
            metagraph: Bittensor metagraph for accessing hotkeys
            challengeperiod_manager: Manager for querying current challenge period status (required)
            current_time_ms: Current timestamp (required for dynamic dust calculation)
            is_testnet: True for testnet (uid 5), False for mainnet (uid 229)
            verbose: Enable detailed logging

        Returns:
            List of (hotkey, weight) tuples with dust weights + burn address
        """
        # Apply minimum dust weights only (no debt-based earnings)
        miner_dust_weights = DebtBasedScoring._apply_minimum_weights(
            ledger_dict=ledger_dict,
            miner_remaining_payouts_usd={hotkey: 0.0 for hotkey in ledger_dict.keys()},  # No debt earnings
            challengeperiod_manager=challengeperiod_manager,
            metagraph=metagraph,
            current_time_ms=current_time_ms,
            verbose=verbose
        )

        # Apply burn address normalization
        result = DebtBasedScoring._normalize_with_burn_address(
            weights=miner_dust_weights,
            metagraph=metagraph,
            is_testnet=is_testnet,
            verbose=verbose
        )

        if verbose:
            bt.logging.info(
                f"Pre-activation weights: {len(ledger_dict)} miners with dust weights, "
                f"excess to burn address"
            )

        return result
