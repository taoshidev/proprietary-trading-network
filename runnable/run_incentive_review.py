from matplotlib import pyplot as plt
import argparse
import pandas as pd
import math

from vali_objects.scoring.scoring import Scoring, ScoringUnit
from vali_objects.utils.logger_utils import LoggerUtils
from vali_objects.utils.subtensor_weight_setter import SubtensorWeightSetter
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.utils.position_manager import PositionManager, PositionUtils
from vali_objects.utils.challengeperiod_manager import ChallengePeriodManager
from time_util.time_util import TimeUtil
from vali_config import ValiConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scoring comparison and output.")
    parser.add_argument("--output", type=str, help="Path to save the output CSV file")
    parser.add_argument("--plot", action="store_true", help="Plot the results")
    args = parser.parse_args()

    logger = LoggerUtils.init_logger("run incentive review")

    current_time = TimeUtil.now_in_millis()
    subtensor_weight_setter = SubtensorWeightSetter(None, None, None)
    challengeperiod_manager = ChallengePeriodManager(None, None)

    hotkeys = ValiBkpUtils.get_directories_in_dir(ValiBkpUtils.get_miner_dir())

    eliminations_json = ValiUtils.get_vali_json_file(
        ValiBkpUtils.get_eliminations_dir()
    )["eliminations"]

    subtensor_weight_setter._refresh_eliminations_in_memory()
    subtensor_weight_setter._refresh_challengeperiod_in_memory()

    challengeperiod_miners = subtensor_weight_setter.challengeperiod_testing
    challengeperiod_passing = subtensor_weight_setter.challengeperiod_success

    passing_hotkeys = list(challengeperiod_passing.keys())
    testing_hotkeys = list(challengeperiod_miners.keys())

    challenge_ledger = subtensor_weight_setter.filtered_ledger(hotkeys=passing_hotkeys + testing_hotkeys)
    challengeperiod_success, challengeperiod_eliminations = challengeperiod_manager.inspect(
        ledger = challenge_ledger,
        inspection_hotkeys = subtensor_weight_setter.challengeperiod_testing,
        current_time = current_time
    )

    filtered_ledger = subtensor_weight_setter.filtered_ledger(hotkeys=passing_hotkeys + challengeperiod_success)
    return_decay_coefficient_short = ValiConfig.HISTORICAL_DECAY_COEFFICIENT_RETURNS_SHORT
    return_decay_coefficient_long = ValiConfig.HISTORICAL_DECAY_COEFFICIENT_RETURNS_LONG
    risk_adjusted_decay_coefficient = ValiConfig.HISTORICAL_DECAY_COEFFICIENT_RISKMETRIC
    return_decay_short_lookback_time_ms = ValiConfig.RETURN_DECAY_SHORT_LOOKBACK_TIME_MS

    # Compute miner penalties
    miner_penalties = Scoring.miner_penalties(filtered_ledger)

    ## Miners with full penalty
    fullpenalty_miner_scores: list[tuple[str, float]] = [ ( miner, 0 ) for miner, penalty in miner_penalties.items() if penalty == 0 ]
    fullpenalty_miners = set([ x[0] for x in fullpenalty_miner_scores ])

    ## Individual miner penalties
    consistency_penalties = {}
    drawdown_penalties = {}

    for miner, ledger in filtered_ledger.items():
        minercps = ledger.cps
        consistency_penalties[miner] = PositionUtils.compute_consistency_penalty_cps(minercps)
        drawdown_penalties[miner] = PositionUtils.compute_drawdown_penalty_cps(minercps)

    # Augmented returns ledgers
    returns_ledger_short = PositionManager.limit_perf_ledger(
        filtered_ledger,
        evaluation_time_ms=current_time,
        lookback_time_ms=return_decay_short_lookback_time_ms,
    )

    scoring_config = {
        'return_cps_short': {
            'function': Scoring.return_cps,
            'weight': ValiConfig.SCORING_RETURN_CPS_SHORT_WEIGHT,
            'ledger': returns_ledger_short,
        },
        'return_cps_long': {
            'function': Scoring.return_cps,
            'weight': ValiConfig.SCORING_RETURN_CPS_LONG_WEIGHT,
            'ledger': filtered_ledger,
        },
        'omega_cps': {
            'function': Scoring.omega_cps,
            'weight': ValiConfig.SCORING_OMEGA_CPS_WEIGHT,
            'ledger': filtered_ledger,
        },
    }

    scoring_config_for_printing = {
        'return_cps_short': {
            'function': Scoring.return_cps,
            'weight': ValiConfig.SCORING_RETURN_CPS_SHORT_WEIGHT
        },
        'return_cps_long': {
            'function': Scoring.return_cps,
            'weight': ValiConfig.SCORING_RETURN_CPS_LONG_WEIGHT
        },  
        'omega_cps': {
            'function': Scoring.omega_cps,
            'weight': ValiConfig.SCORING_OMEGA_CPS_WEIGHT
        },
    }

    # Print the scoring configuration without the 'ledger' elements
    print(f"Scoring configuration: {scoring_config_for_printing}")
    combined_scores = {}

    # Store rankings and original scores
    rankings = {}
    original_scores = {}
    consistencies = {}

    for metric_name, config in scoring_config.items():
        miner_scores = []
        for miner, minerledger in config['ledger'].items():
            scoringunit = ScoringUnit.from_perf_ledger(minerledger)
            score = config['function'](scoringunit=scoringunit)
            score_riskadjusted = score * miner_penalties.get(miner, 0)
            miner_scores.append((miner, score_riskadjusted))
        
        # Save original scores for printout
        original_scores[metric_name] = {miner: score for miner, score in miner_scores}

        # Now filter out miners with full penalties
        filtered_miner_scores = [ x for x in miner_scores if x[0] not in fullpenalty_miners ]

        # Apply weight and calculate weighted scores
        weighted_scores = Scoring.miner_scores_percentiles(filtered_miner_scores)
        rankings[metric_name] = sorted(weighted_scores, key=lambda x: x[1], reverse=True)

        # Combine scores
        for miner, score in weighted_scores:
            if miner not in combined_scores:
                combined_scores[miner] = 1
            combined_scores[miner] *= config['weight'] * score + (1 - config['weight'])

    combined_weighed = Scoring.weigh_miner_scores(list(combined_scores.items())) + fullpenalty_miner_scores
    combined_scores = dict(combined_weighed)

    ## Normalize the scores
    normalized_scores = Scoring.normalize_scores(combined_scores)
    checkpoint_results = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)

    # Prepare data for DataFrame
    combined_data = {}

    for metric, ranks in rankings.items():
        for rank, (miner, weighted_score) in enumerate(ranks, start=1):
            if miner not in combined_data:
                combined_data[miner] = {}
            combined_data[miner][f'{metric} Original Score'] = original_scores[metric][miner]
            combined_data[miner][f'{metric} Weighted Score'] = weighted_score
            combined_data[miner][f'{metric} Rank'] = rank

    for rank, (miner, score) in enumerate(checkpoint_results, start=1):
        if miner not in combined_data:
            combined_data[miner] = {}
        combined_data[miner]['Final Normalized Score'] = score
        combined_data[miner]['Final Rank'] = rank
        combined_data[miner]['Penalty'] = miner_penalties.get(miner, 0)
        combined_data[miner]['Drawdown Penalty'] = drawdown_penalties.get(miner, 0)
        combined_data[miner]['Consistency Penalty'] = consistency_penalties.get(miner, 0)

    df = pd.DataFrame.from_dict(combined_data, orient='index')

    # printing_columns = [
    #     'return_cps_short Weighted Score',
    #     'return_cps_long Weighted Score',
    #     'Final Normalized Score',
    #     'Final Rank',
    #     'Penalty',
    #     'Drawdown Penalty',
    #     'Consistency Penalty',
    # ]

    # df_subset = df[printing_columns].round(3)
    # df_subset = df_subset.sort_values(by='Final Rank', ascending=True)
    # # print(df_subset)

    # Print rankings and original scores for each metric
    for metric, ranks in rankings.items():
        print(f"Ranking for {metric}:")
        for rank, (miner, weighted_score) in enumerate(ranks, start=1):
            original_score = original_scores[metric][miner]
            print(f"{rank}. {miner} - Original Score: {original_score:.4e}, Weighted Score: {weighted_score:.2f}")
        print()

    print("\nMiner Penalties:\n")
    for miner, penalty in miner_penalties.items():
        print(f"{miner}: {penalty:.2f}")

    # Print final rankings
    print("\nFinal Rankings:")
    for rank, (miner, score) in enumerate(checkpoint_results, start=1):
        print(f"{rank}. {miner} - Normalized Score: {score:.2f} - Long Return: {(math.exp(original_scores['return_cps_long'][miner]) - 1)*100:.2f} - Short Return: {(math.exp(original_scores['return_cps_short'][miner]) - 1)*100:.2f}")

    # Save to CSV if the --output flag is set
    if args.output:
        df.to_csv(args.output, index=True)
        print(f"Output saved to {args.output}")

    challengeperiod_results = [(x, ValiConfig.SET_WEIGHT_MINER_CHALLENGE_PERIOD_WEIGHT) for x in challengeperiod_miners]
    sorted_data = checkpoint_results + challengeperiod_results

    if args.plot:
        y_values = [x[1] for x in sorted_data]
        top_miners = [x[0] for x in sorted_data]

        # Add names for each value
        for x in range(len(y_values)):
            plt.text(x, y_values[x], f"({top_miners[x]}, {y_values[x]})", ha="left")

        plt.plot([x for x in range(len(y_values))], y_values, marker="o", linestyle="-")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.title("Top Miners Incentive")
        plt.grid(True)
        plt.show()
