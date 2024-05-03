from matplotlib import pyplot as plt
import time

from vali_objects.scoring.scoring import Scoring
from vali_objects.utils.logger_utils import LoggerUtils
from vali_objects.utils.subtensor_weight_setter import SubtensorWeightSetter
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
from time_util.time_util import TimeUtil
from vali_objects.utils.challengeperiod_manager import ChallengePeriodManager

if __name__ == "__main__":
    logger = LoggerUtils.init_logger("run challenge review")

    current_time = TimeUtil.now_in_millis()
    subtensor_weight_setter = SubtensorWeightSetter(None, None, None)
    challengeperiod_manager = ChallengePeriodManager(None, None, None)

    ## Collect the ledger
    ledger = subtensor_weight_setter.perf_manager.load_perf_ledgers_from_disk()

    ## Check the current testing miners for different scenarios
    subtensor_weight_setter._refresh_eliminations_in_memory()
    subtensor_weight_setter._refresh_challengeperiod_in_memory()

    inspection_hotkeys_dict = subtensor_weight_setter.challengeperiod_testing
    print(f"Testing hotkeys: {inspection_hotkeys_dict}")

    ## filter the ledger for the miners that passed the challenge period
    success_hotkeys = list(inspection_hotkeys_dict.keys())
    filtered_ledger = subtensor_weight_setter.filtered_ledger(hotkeys=success_hotkeys)

    print(f"Filtered ledger: {filtered_ledger}")

    # Get all possible positions, even beyond the lookback range
    success, eliminations = challengeperiod_manager.inspect(
        ledger=filtered_ledger,
        inspection_hotkeys=inspection_hotkeys_dict,
        current_time=current_time,
        log=True
    )

    prior_challengeperiod_miners = set(inspection_hotkeys_dict.keys())
    success_miners = set(success)
    eliminations = set(eliminations)

    post_challengeperiod_miners = prior_challengeperiod_miners - eliminations - success_miners

    logger.info(f"{len(prior_challengeperiod_miners)} prior_challengeperiod_miners [{prior_challengeperiod_miners}]")
    logger.info(f"{len(success_miners)} success_miners [{success_miners}]")
    logger.info(f"{len(eliminations)} challengeperiod_eliminations [{eliminations}]")
    logger.info(f"{len(post_challengeperiod_miners)} challenge period remaining [{post_challengeperiod_miners}]")