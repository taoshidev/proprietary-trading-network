# developer: trdougherty
from typing import List
import math

from scipy.stats import percentileofscore

from time_util.time_util import TimeUtil
from vali_config import ValiConfig
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.subtensor_weight_setter import SubtensorWeightSetter
from vali_objects.utils.position_utils import PositionUtils
from vali_objects.utils.position_penalties import PositionPenalties
from vali_objects.utils.position_filtering import PositionFiltering
from vali_objects.utils.ledger_utils import LedgerUtils
from vali_objects.scoring.scoring import Scoring


def rank_dictionary(d, ascending=False):
    """
    Rank the values in a dictionary. Higher values get lower ranks by default.
    
    Args:
    d (dict): The dictionary to rank.
    ascending (bool): If True, ranks in ascending order. Default is False.
    
    Returns:
    dict: A dictionary with the same keys and ranked values.
    """
    # Sort the dictionary by value
    sorted_items = sorted(d.items(), key=lambda item: item[1], reverse=not ascending)

    # Assign ranks
    ranks = {item[0]: rank + 1 for rank, item in enumerate(sorted_items)}

    return ranks


def apply_penalties(scores: dict[str, float], penalties: dict[str, float]) -> dict[str, float]:
    """
    Apply penalties to scores.

    Args:
    scores (dict): The scores to penalize.
    penalties (dict): The penalties to apply.

    Returns:
    dict: A dictionary with the same keys and penalized values.
    """
    penalized_scores = {k: scores[k] * penalties.get(k, 1.0) for k in scores.keys()}

    return penalized_scores


def percentile_rank_dictionary(d, ascending=False) -> dict:
    """
    Rank the values in a dictionary as a percentile. Higher values get lower ranks by default.
    
    Args:
    d (dict): The dictionary to rank.
    ascending (bool): If True, ranks in ascending order. Default is False.
    
    Returns:
    dict: A dictionary with the same keys and ranked values.
    """
    # Sort the dictionary by value
    miner_names = list(d.keys())
    scores = list(d.values())

    percentiles = percentileofscore(scores, scores) / 100
    miner_percentiles = dict(zip(miner_names, percentiles))

    return miner_percentiles


def generate_miner_statistics_data(time_now: int = None, checkpoints: bool = True, selected_miner_hotkeys: List[str] = None):
    if time_now is None:
        time_now = TimeUtil.now_in_millis()

    subtensor_weight_setter = SubtensorWeightSetter(
        config=None,
        wallet=None,
        metagraph=None,
        running_unit_tests=False
    )

    # Collect information from the disk and populate variables in memory
    subtensor_weight_setter._refresh_eliminations_in_memory()
    subtensor_weight_setter._refresh_challengeperiod_in_memory()

    # Get the dictionaries
    challengeperiod_testing_dictionary = subtensor_weight_setter.challengeperiod_testing
    challengeperiod_success_dictionary = subtensor_weight_setter.challengeperiod_success

    # Sort dictionaries by value
    sorted_challengeperiod_testing = dict(sorted(challengeperiod_testing_dictionary.items(), key=lambda item: item[1]))
    sorted_challengeperiod_success = dict(sorted(challengeperiod_success_dictionary.items(), key=lambda item: item[1]))

    challengeperiod_testing_hotkeys = list(challengeperiod_testing_dictionary.keys())
    challengeperiod_success_hotkeys = list(challengeperiod_success_dictionary.keys())

    # get plagiarism scores
    plagiarism = subtensor_weight_setter.get_plagiarism_scores_from_disk()

    try:
        all_miner_hotkeys: list = ValiBkpUtils.get_directories_in_dir(
            ValiBkpUtils.get_miner_dir()
        )
    except FileNotFoundError:
        raise Exception(
            f"directory for miners doesn't exist "
            f"[{ValiBkpUtils.get_miner_dir()}]. Skip run for now."
        )

    # full ledger of all miner hotkeys
    all_miner_hotkeys = challengeperiod_success_hotkeys + challengeperiod_testing_hotkeys
    if selected_miner_hotkeys is None:
        selected_miner_hotkeys = all_miner_hotkeys

    filtered_ledger = subtensor_weight_setter.filtered_ledger(hotkeys=all_miner_hotkeys)
    filtered_positions = subtensor_weight_setter.filtered_positions(hotkeys=all_miner_hotkeys)

    # # Sync the ledger and positions
    # filtered_ledger, filtered_positions = subtensor_weight_setter.sync_ledger_positions(
    #     filtered_ledger,
    #     filtered_positions
    # )

    # Lookback window positions
    lookback_positions = PositionFiltering.filter(
        filtered_positions,
        evaluation_time_ms=time_now
    )

    lookback_positions_recent = PositionFiltering.filter_recent(
        filtered_positions,
        evaluation_time_ms=time_now
    )

    # Penalties
    miner_penalties = Scoring.miner_penalties(
        lookback_positions,
        filtered_ledger
    )

    # Scoring metrics
    omega_dict = {}
    sharpe_dict = {}
    short_return_dict = {}
    return_dict = {}
    short_risk_adjusted_return_dict = {}
    risk_adjusted_return_dict = {}

    # Positional ratios
    positional_return_time_consistency_ratios = {}
    positional_realized_returns_ratios = {}

    # Positional penalties
    positional_return_time_consistency_penalties = {}
    positional_realized_returns_penalties = {}

    # Ledger Ratios
    ledger_daily_consistency_ratios = {}
    ledger_biweekly_consistency_ratios = {}

    # Ledger Penalties
    daily_consistency_penalty = {}
    biweekly_consistency_penalty = {}
    drawdown_penalties = {}
    max_drawdown_threshold_penalties = {}

    # Ledger Drawdowns
    recent_drawdowns = {}
    approximate_drawdowns = {}
    effective_drawdowns = {}

    # Perf Ledger Calculations
    n_checkpoints = {}
    checkpoint_durations = {}

    # Positional Statistics
    n_positions = {}
    positional_return = {}
    positional_duration = {}

    for hotkey, hotkey_ledger in filtered_ledger.items():
        # Collect miner positions
        miner_positions = filtered_positions.get(hotkey, [])

        # Lookback window positions
        miner_lookback_positions = lookback_positions.get(hotkey, [])
        miner_lookback_positions_recent = lookback_positions_recent.get(hotkey, [])

        scoring_input = {
            "ledger": hotkey_ledger,
            "positions": miner_lookback_positions,
        }

        # Positional Scoring
        omega_dict[hotkey] = Scoring.omega(**scoring_input)
        sharpe_dict[hotkey] = Scoring.sharpe(**scoring_input)

        short_return_dict[hotkey] = math.exp(Scoring.base_return(miner_lookback_positions_recent))
        return_dict[hotkey] = math.exp(Scoring.base_return(miner_lookback_positions))

        short_risk_adjusted_return_dict[hotkey] = Scoring.risk_adjusted_return(
            miner_lookback_positions_recent,
            hotkey_ledger
        )

        risk_adjusted_return_dict[hotkey] = Scoring.risk_adjusted_return(
            miner_lookback_positions,
            hotkey_ledger
        )

        # Ledger consistency penalties
        recent_drawdown = LedgerUtils.recent_drawdown(hotkey_ledger.cps)
        recent_drawdowns[hotkey] = recent_drawdown

        approximate_drawdown = LedgerUtils.approximate_drawdown(hotkey_ledger.cps)
        approximate_drawdowns[hotkey] = approximate_drawdown

        effective_drawdowns[hotkey] = LedgerUtils.effective_drawdown(recent_drawdown, approximate_drawdown)
        drawdown_penalties[hotkey] = LedgerUtils.risk_normalization(hotkey_ledger.cps)

        ledger_daily_consistency_ratios[hotkey] = LedgerUtils.daily_consistency_ratio(hotkey_ledger.cps)
        daily_consistency_penalty[hotkey] = LedgerUtils.daily_consistency_penalty(hotkey_ledger.cps)

        ledger_biweekly_consistency_ratios[hotkey] = LedgerUtils.biweekly_consistency_ratio(hotkey_ledger.cps)
        biweekly_consistency_penalty[hotkey] = LedgerUtils.biweekly_consistency_penalty(hotkey_ledger.cps)

        max_drawdown_threshold_penalties[hotkey] = LedgerUtils.max_drawdown_threshold_penalty(hotkey_ledger.cps)

        # Positional consistency ratios
        positional_realized_returns_ratios[hotkey] = PositionPenalties.returns_ratio(miner_lookback_positions)
        positional_realized_returns_penalties[hotkey] = PositionPenalties.returns_ratio_penalty(miner_lookback_positions)

        positional_return_time_consistency_ratios[hotkey] = PositionPenalties.time_consistency_ratio(miner_lookback_positions)
        positional_return_time_consistency_penalty = PositionPenalties.time_consistency_penalty(miner_lookback_positions)
        positional_return_time_consistency_penalties[hotkey] = positional_return_time_consistency_penalty

        # Now for the ledger statistics
        n_checkpoints[hotkey] = len([x for x in hotkey_ledger.cps if x.open_ms > 0])
        checkpoint_durations[hotkey] = sum([x.open_ms for x in hotkey_ledger.cps])

        # Now for the full positions statistics
        n_positions[hotkey] = len(miner_positions)
        positional_return[hotkey] = math.exp(Scoring.base_return(miner_positions))
        positional_duration[hotkey] = PositionUtils.total_duration(miner_positions)

    # Cumulative ledger, for printing
    cumulative_return_ledger = LedgerUtils.cumulative(filtered_ledger)

    # This is when we only want to look at the successful miners
    successful_ledger = subtensor_weight_setter.filtered_ledger(hotkeys=challengeperiod_success_hotkeys)
    successful_positions = subtensor_weight_setter.filtered_positions(hotkeys=challengeperiod_success_hotkeys)

    # successful_ledger, successful_positions = subtensor_weight_setter.sync_ledger_positions(
    #     successful_ledger,
    #     successful_positions
    # )

    checkpoint_results = Scoring.compute_results_checkpoint(
        successful_ledger,
        successful_positions,
        evaluation_time_ms=time_now,
        verbose=False
    )

    challengeperiod_scores = [
        (x, ValiConfig.CHALLENGE_PERIOD_WEIGHT) for x in challengeperiod_testing_hotkeys
    ]

    scoring_results = checkpoint_results + challengeperiod_scores
    weights = dict(scoring_results)

    # Rankings
    weights_rank = rank_dictionary(weights)
    weights_percentile = percentile_rank_dictionary(weights)

    # Rankings on Traditional Metrics
    omega_rank = rank_dictionary(omega_dict)
    omega_percentile = percentile_rank_dictionary(omega_dict)

    sharpe_rank = rank_dictionary(sharpe_dict)
    sharpe_percentile = percentile_rank_dictionary(sharpe_dict)

    short_return_rank = rank_dictionary(short_return_dict)
    short_return_percentile = percentile_rank_dictionary(short_return_dict)

    return_rank = rank_dictionary(return_dict)
    return_percentile = percentile_rank_dictionary(return_dict)

    short_risk_adjusted_return_rank = rank_dictionary(short_risk_adjusted_return_dict)
    short_risk_adjusted_return_percentile = percentile_rank_dictionary(short_risk_adjusted_return_dict)

    risk_adjusted_return_rank = rank_dictionary(risk_adjusted_return_dict)
    risk_adjusted_return_percentile = percentile_rank_dictionary(risk_adjusted_return_dict)

    # Rankings on Penalized Metrics
    omega_penalized_dict = apply_penalties(omega_dict, miner_penalties)
    omega_penalized_rank = rank_dictionary(omega_penalized_dict)
    omega_penalized_percentile = percentile_rank_dictionary(omega_penalized_dict)

    sharpe_penalized_dict = apply_penalties(sharpe_dict, miner_penalties)
    sharpe_penalized_rank = rank_dictionary(sharpe_penalized_dict)
    sharpe_penalized_percentile = percentile_rank_dictionary(sharpe_penalized_dict)

    short_risk_adjusted_return_penalized_dict = apply_penalties(short_risk_adjusted_return_dict, miner_penalties)
    short_risk_adjusted_return_penalized_rank = rank_dictionary(short_risk_adjusted_return_penalized_dict)
    short_risk_adjusted_return_penalized_percentile = percentile_rank_dictionary(short_risk_adjusted_return_penalized_dict)

    risk_adjusted_return_penalized_dict = apply_penalties(risk_adjusted_return_dict, miner_penalties)
    risk_adjusted_return_penalized_rank = rank_dictionary(risk_adjusted_return_penalized_dict)
    risk_adjusted_return_penalized_percentile = percentile_rank_dictionary(risk_adjusted_return_penalized_dict)

    # Here is the full list of data in frontend format
    combined_data = []
    for miner_id in selected_miner_hotkeys:
        # challenge period specific data
        challengeperiod_specific = {}

        if miner_id in sorted_challengeperiod_testing:
            challengeperiod_testing_time = sorted_challengeperiod_testing[miner_id]
            chellengeperiod_end_time = challengeperiod_testing_time + ValiConfig.CHALLENGE_PERIOD_MS
            remaining_time = chellengeperiod_end_time - time_now
            challengeperiod_specific = {
                "status": "testing",
                "start_time_ms": challengeperiod_testing_time,
                "remaining_time_ms": remaining_time,
            }

            challengeperiod_positions = filtered_positions.get(miner_id, [])
            challengeperiod_positions_length = len(challengeperiod_positions)
            challengeperiod_positions_target = ValiConfig.CHALLENGE_PERIOD_MIN_POSITIONS
            challengeperiod_positions_passing = bool(challengeperiod_positions_length >= challengeperiod_positions_target)

            challengeperiod_return_ratio = positional_realized_returns_ratios.get(miner_id, 1.0)
            challengeperiod_return_ratio_target = ValiConfig.CHALLENGE_PERIOD_MAX_POSITIONAL_RETURNS_RATIO
            challengeperiod_return_ratio_passing = bool(challengeperiod_return_ratio < challengeperiod_return_ratio_target)

            challengeperiod_return = return_dict.get(miner_id, 0)
            challengeperiod_return_percentage = (math.exp(challengeperiod_return) - 1)*100
            challengeperiod_return_target = ValiConfig.CHALLENGE_PERIOD_RETURN_PERCENTAGE
            challengeperiod_return_passing = bool(challengeperiod_return_percentage >= challengeperiod_return_target)

            challengeperiod_unrealized_ratio = ledger_daily_consistency_ratios.get(miner_id, 1.0)
            challengeperiod_unrealized_ratio_target = ValiConfig.CHALLENGE_PERIOD_MAX_UNREALIZED_RETURNS_RATIO
            challengeperiod_unrealized_ratio_passing = bool(challengeperiod_unrealized_ratio <= challengeperiod_unrealized_ratio_target)

            challengeperiod_specific = {**challengeperiod_specific, **{
                "positions": {
                    "value": challengeperiod_positions_length,
                    "target": challengeperiod_positions_target,
                    "passing": challengeperiod_positions_passing,
                },
                "return_ratio": {
                    "value": challengeperiod_return_ratio,
                    "target": challengeperiod_return_ratio_target,
                    "passing": challengeperiod_return_ratio_passing,
                },
                "return": {
                    "value": challengeperiod_return_percentage,
                    "target": challengeperiod_return_target,
                    "passing": challengeperiod_return_passing,
                },
                "unrealized_ratio": {
                    "value": challengeperiod_unrealized_ratio,
                    "target": challengeperiod_unrealized_ratio_target,
                    "passing": challengeperiod_unrealized_ratio_passing,
                }
            }}

        elif miner_id in sorted_challengeperiod_success:
            challengeperiod_success_time = sorted_challengeperiod_success[miner_id]
            challengeperiod_specific = {
                "status": "success",
                "start_time": challengeperiod_success_time,
            }

        # checkpoint specific data
        miner_cumulative_return_ledger = cumulative_return_ledger.get(miner_id)
        miner_standard_ledger = filtered_ledger.get(miner_id)

        if miner_standard_ledger is None:
            continue

        miner_data = {
            "hotkey": miner_id,
            "weight": {
                "value": weights.get(miner_id),
                "rank": weights_rank.get(miner_id),
                "percentile": weights_percentile.get(miner_id),
            },
            "challengeperiod": challengeperiod_specific,
            "penalties": {
                "time_consistency": positional_return_time_consistency_penalties.get(miner_id),
                "returns_ratio": positional_realized_returns_penalties.get(miner_id),
                "drawdown_threshold": max_drawdown_threshold_penalties.get(miner_id),
                "drawdown": drawdown_penalties.get(miner_id),
                "daily": daily_consistency_penalty.get(miner_id),
                "biweekly": biweekly_consistency_penalty.get(miner_id),
                "total": miner_penalties.get(miner_id, 0.0),
            },
            "ratios": {
                "time_consistency": positional_return_time_consistency_ratios.get(miner_id),
                "returns_ratio": positional_realized_returns_ratios.get(miner_id),
                "daily": ledger_daily_consistency_ratios.get(miner_id),
                "biweekly": ledger_biweekly_consistency_ratios.get(miner_id),
            },
            "drawdowns": {
                "recent": recent_drawdowns.get(miner_id),
                "approximate": approximate_drawdowns.get(miner_id),
                "effective": effective_drawdowns.get(miner_id),
            },
            "scores": {
                "omega": {
                    "value": omega_dict.get(miner_id),
                    "rank": omega_rank.get(miner_id),
                    "percentile": omega_percentile.get(miner_id),
                },
                "sharpe": {
                    "value": sharpe_dict.get(miner_id),
                    "rank": sharpe_rank.get(miner_id),
                    "percentile": sharpe_percentile.get(miner_id),
                },
                "short_return": {
                    "value": short_return_dict.get(miner_id),
                    "rank": short_return_rank.get(miner_id),
                    "percentile": short_return_percentile.get(miner_id),
                },
                "return": {
                    "value": return_dict.get(miner_id),
                    "rank": return_rank.get(miner_id),
                    "percentile": return_percentile.get(miner_id),
                },
                "short_risk_adjusted_return": {
                    "value": short_risk_adjusted_return_dict.get(miner_id),
                    "rank": short_risk_adjusted_return_rank.get(miner_id),
                    "percentile": short_risk_adjusted_return_percentile.get(miner_id),
                },
                "risk_adjusted_return": {
                    "value": risk_adjusted_return_dict.get(miner_id),
                    "rank": risk_adjusted_return_rank.get(miner_id),
                    "percentile": risk_adjusted_return_percentile.get(miner_id),
                }
            },
            "penalized_scores": {
                "omega": {
                    "value": omega_penalized_dict.get(miner_id),
                    "rank": omega_penalized_rank.get(miner_id),
                    "percentile": omega_penalized_percentile.get(miner_id),
                },
                "sharpe": {
                    "value": sharpe_penalized_dict.get(miner_id),
                    "rank": sharpe_penalized_rank.get(miner_id),
                    "percentile": sharpe_penalized_percentile.get(miner_id),
                },
                "short_risk_adjusted_return": {
                    "value": short_risk_adjusted_return_penalized_dict.get(miner_id),
                    "rank": short_risk_adjusted_return_penalized_rank.get(miner_id),
                    "percentile": short_risk_adjusted_return_penalized_percentile.get(miner_id),
                },
                "risk_adjusted_return": {
                    "value": risk_adjusted_return_penalized_dict.get(miner_id),
                    "rank": risk_adjusted_return_penalized_rank.get(miner_id),
                    "percentile": risk_adjusted_return_penalized_percentile.get(miner_id),
                }
            },
            "engagement": {
                "n_checkpoints": n_checkpoints.get(miner_id),
                "n_positions": n_positions.get(miner_id),
                "position_duration": positional_duration.get(miner_id),
                "checkpoint_durations": checkpoint_durations.get(miner_id)
            },
            "plagiarism": plagiarism.get(miner_id)
        }

        miner_checkpoints = {
            "checkpoints": miner_cumulative_return_ledger.get('cps', [])
        }

        if checkpoints:
            miner_data = {**miner_data, **miner_checkpoints}

        combined_data.append(miner_data)

    # Now pipe the vali_config data into the final dictionary
    ldconfig_data = dict(ValiConfig.__dict__)
    ldconfig_printable = {
        key: value for key, value in ldconfig_data.items()
        if isinstance(value, (int, float, str))
           and key not in ['BASE_DIR', 'base_directory']
    }

    # Filter out invalid entries
    valid_data = [item for item in combined_data if item is not None and
                  item.get('weight') is not None and
                  item['weight'].get('rank') is not None]

    # If there's no valid data, return an empty dict or handle accordingly
    if not valid_data:
        return {
            'version': ValiConfig.VERSION,
            'created_timestamp_ms': time_now,
            'created_date': TimeUtil.millis_to_formatted_date_str(time_now),
            'data': [],  # empty list as there's no valid data
            'constants': ldconfig_printable,
        }

    final_dict = {
        'version': ValiConfig.VERSION,
        'created_timestamp_ms': time_now,
        'created_date': TimeUtil.millis_to_formatted_date_str(time_now),
        'data': sorted(valid_data, key=lambda x: x['weight']['rank']),
        'constants': ldconfig_printable,
    }

    return final_dict


def generate_request_minerstatistics(time_now: int, checkpoints: bool = True):
    final_dict = generate_miner_statistics_data(time_now, checkpoints)

    output_file_path = ValiBkpUtils.get_vali_outputs_dir() + "minerstatistics.json"
    ValiBkpUtils.write_file(
        output_file_path,
        final_dict,
    )
