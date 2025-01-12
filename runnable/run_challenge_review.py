from vali_objects.utils.elimination_manager import EliminationManager
from vali_objects.utils.logger_utils import LoggerUtils
from vali_objects.utils.plagiarism_detector import PlagiarismDetector
from vali_objects.utils.position_manager import PositionManager
from vali_objects.utils.subtensor_weight_setter import SubtensorWeightSetter
from time_util.time_util import TimeUtil
from vali_objects.utils.challengeperiod_manager import ChallengePeriodManager
from vali_objects.vali_dataclasses.perf_ledger import PerfLedgerManager

if __name__ == "__main__":
    logger = LoggerUtils.init_logger("run challenge review")

    current_time = TimeUtil.now_in_millis()

    perf_ledger_manager = PerfLedgerManager(None)
    elimination_manager = EliminationManager(None, None, None)
    position_manager = PositionManager(None, None, elimination_manager=elimination_manager,
                                       challengeperiod_manager=None,
                                       perf_ledger_manager=perf_ledger_manager)
    challengeperiod_manager = ChallengePeriodManager(None, None, position_manager=position_manager)

    elimination_manager.position_manager = position_manager
    position_manager.challengeperiod_manager = challengeperiod_manager
    elimination_manager.challengeperiod_manager = challengeperiod_manager
    challengeperiod_manager.position_manager = position_manager
    perf_ledger_manager.position_manager = position_manager
    subtensor_weight_setter = SubtensorWeightSetter(
        config=None,
        wallet=None,
        metagraph=None,
        running_unit_tests=False,
        position_manager=position_manager,
    )
    plagiarism_detector = PlagiarismDetector(None, None, position_manager=position_manager)

    ## Collect the ledger
    ledger = subtensor_weight_setter.perf_ledger_manager.get_perf_ledgers_from_memory()

    ## Check the current testing miners for different scenarios
    challengeperiod_manager._refresh_challengeperiod_in_memory()

    inspection_hotkeys_dict = challengeperiod_manager.challengeperiod_testing

    ## filter the ledger for the miners that passed the challenge period
    success_hotkeys = list(inspection_hotkeys_dict.keys())
    filtered_ledger = subtensor_weight_setter.filtered_ledger(hotkeys=success_hotkeys)

    # Get all possible positions, even beyond the lookback range
    success, eliminations = challengeperiod_manager.inspect(
        ledger=filtered_ledger,
        inspection_hotkeys=inspection_hotkeys_dict,
        current_time=current_time,
    )

    prior_challengeperiod_miners = set(inspection_hotkeys_dict.keys())
    success_miners = set(success)
    eliminations = set(eliminations)

    post_challengeperiod_miners = prior_challengeperiod_miners - eliminations - success_miners

    logger.info(f"{len(prior_challengeperiod_miners)} prior_challengeperiod_miners [{prior_challengeperiod_miners}]")
    logger.info(f"{len(success_miners)} success_miners [{success_miners}]")
    logger.info(f"{len(eliminations)} challengeperiod_eliminations [{eliminations}]")
    logger.info(f"{len(post_challengeperiod_miners)} challenge period remaining [{post_challengeperiod_miners}]")