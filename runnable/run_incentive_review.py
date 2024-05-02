from matplotlib import pyplot as plt

from vali_objects.scoring.scoring import Scoring
from vali_objects.utils.logger_utils import LoggerUtils
from vali_objects.utils.subtensor_weight_setter import SubtensorWeightSetter
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
from time_util.time_util import TimeUtil
from vali_config import ValiConfig

if __name__ == "__main__":
    logger = LoggerUtils.init_logger("run incentive review")

    current_time = TimeUtil.now_in_millis()
    subtensor_weight_setter = SubtensorWeightSetter(None, None, None)

    hotkeys = ValiBkpUtils.get_directories_in_dir(ValiBkpUtils.get_miner_dir())

    eliminations_json = ValiUtils.get_vali_json_file(
        ValiBkpUtils.get_eliminations_dir()
    )["eliminations"]

    logger.info(f"Testing hotkeys: {hotkeys}")

    challengeperiod_resultdict = subtensor_weight_setter.challenge_period_screening(
        hotkeys = hotkeys,
        eliminations = eliminations_json,
        current_time = current_time
    )

    challengeperiod_miners = challengeperiod_resultdict["challengeperiod_miners"]
    challengeperiod_elimination_hotkeys = challengeperiod_resultdict["challengeperiod_eliminations"]

    logger.info(f"Challenge period miners [{challengeperiod_miners}]")
    logger.info(f"Challenge period eliminations [{challengeperiod_elimination_hotkeys}]")

    # augmented ledger should have the gain, loss, n_updates, and time_duration
    augmented_ledger = subtensor_weight_setter.augmented_ledger(
        hotkeys = hotkeys,
        omitted_miners = challengeperiod_miners + challengeperiod_elimination_hotkeys,
        eliminations = eliminations_json
    )

    checkpoint_results = Scoring.compute_results_checkpoint(augmented_ledger)
    challengeperiod_results = [ (x, ValiConfig.SET_WEIGHT_MINER_CHALLENGE_PERIOD_WEIGHT) for x in challengeperiod_miners ]

    sorted_data = checkpoint_results + challengeperiod_results

    logger.info(f"Challenge period results [{challengeperiod_results}]")

    y_values = [x[1] for x in sorted_data]
    top_miners = [x[0] for x in sorted_data]

    logger.info(f"top miners [{top_miners}]")
    logger.info(f"top miners scores [{y_values}]")

    # Add names for each value
    for x in range(len(y_values)):
        plt.text(x, y_values[x], f"({top_miners[x]}, {y_values[x]})", ha="left")

    plt.plot([x for x in range(len(y_values))], y_values, marker="o", linestyle="-")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Top Miners Incentive")
    plt.grid(True)
    plt.show()
