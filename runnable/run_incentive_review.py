from matplotlib import pyplot as plt
import argparse
import pandas as pd

from vali_objects.scoring.scoring import Scoring, ScoringUnit
from vali_objects.utils.logger_utils import LoggerUtils
from vali_objects.utils.subtensor_weight_setter import SubtensorWeightSetter
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.utils.position_manager import PositionManager
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
    filtered_ledger = subtensor_weight_setter.filtered_ledger(hotkeys=passing_hotkeys)

    return_decay_coefficient = ValiConfig.HISTORICAL_DECAY_COEFFICIENT_RETURNS
    risk_adjusted_decay_coefficient = ValiConfig.HISTORICAL_DECAY_COEFFICIENT_RISKMETRIC

    returns_ledger = PositionManager.augment_perf_ledger(
        filtered_ledger,
        evaluation_time_ms=current_time,
        time_decay_coefficient=return_decay_coefficient,
    )

    risk_adjusted_ledger = PositionManager.augment_perf_ledger(
        filtered_ledger,
        evaluation_time_ms=current_time,
        time_decay_coefficient=risk_adjusted_decay_coefficient,
    )

    scoring_config = {
        'return_cps': {
            'function': Scoring.return_cps,
            'weight': ValiConfig.SCORING_RETURN_CPS_WEIGHT,
            'ledger': returns_ledger,
        },
        'omega_cps': {
            'function': Scoring.omega_cps,
            'weight': ValiConfig.SCORING_OMEGA_CPS_WEIGHT,
            'ledger': risk_adjusted_ledger,
        },
        'inverted_sortino_cps': {
            'function': Scoring.inverted_sortino_cps,
            'weight': ValiConfig.SCORING_SORTINO_CPS_WEIGHT,
            'ledger': risk_adjusted_ledger,
        },
    }

    scoring_config_for_printing = {
        'return_cps': {
            'function': Scoring.return_cps,
            'weight': ValiConfig.SCORING_RETURN_CPS_WEIGHT
        },
        'omega_cps': {
            'function': Scoring.omega_cps,
            'weight': ValiConfig.SCORING_OMEGA_CPS_WEIGHT
        },
        'inverted_sortino_cps': {
            'function': Scoring.inverted_sortino_cps,
            'weight': ValiConfig.SCORING_SORTINO_CPS_WEIGHT
        }
    }

    # Print the scoring configuration without the 'ledger' elements
    print(f"Scoring configuration: {scoring_config_for_printing}")
    combined_scores = {}

    # Store rankings and original scores
    rankings = {}
    original_scores = {}

    for metric_name, config in scoring_config.items():
        miner_scores = []
        for miner, minerledger in config['ledger'].items():
            scoringunit = ScoringUnit.from_perf_ledger(minerledger)
            score = config['function'](scoringunit=scoringunit)
            miner_scores.append((miner, score))
        
        # Save original scores for printout
        original_scores[metric_name] = {miner: score for miner, score in miner_scores}

        # Apply weight and calculate weighted scores
        weighted_scores = Scoring.weigh_miner_scores(miner_scores)
        rankings[metric_name] = sorted(weighted_scores, key=lambda x: x[1], reverse=True)

        # Combine scores
        for miner, score in weighted_scores:
            if miner not in combined_scores:
                combined_scores[miner] = 1
            combined_scores[miner] *= config['weight'] * score + (1 - config['weight'])

    # Calculate the final weighted score and normalize
    combined_weighed = Scoring.weigh_miner_scores(list(combined_scores.items()))
    combined_scores = dict(combined_weighed)

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

    df = pd.DataFrame.from_dict(combined_data, orient='index')

    # Print rankings and original scores for each metric
    for metric, ranks in rankings.items():
        print(f"Ranking for {metric}:")
        for rank, (miner, weighted_score) in enumerate(ranks, start=1):
            original_score = original_scores[metric][miner]
            print(f"{rank}. {miner} - Original Score: {original_score:.4e}, Weighted Score: {weighted_score:.2f}")
        print()

    # Print final rankings
    print("\nFinal Rankings:")
    for rank, (miner, score) in enumerate(checkpoint_results, start=1):
        print(f"{rank}. {miner} - Normalized Score: {score:.2f}")

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
