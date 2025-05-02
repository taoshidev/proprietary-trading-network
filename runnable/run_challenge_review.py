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
        metagraph=None,
        running_unit_tests=False,
        position_manager=position_manager,
    )
    plagiarism_detector = PlagiarismDetector(None, None, position_manager=position_manager)

    ## Collect the ledger
    ledger = subtensor_weight_setter.perf_ledger_manager.get_perf_ledgers()


    inspection_hotkeys_dict = challengeperiod_manager.challengeperiod_testing

    ## filter the ledger for the miners that passed the challenge period
    testing_hotkeys = list(inspection_hotkeys_dict.keys())
    filtered_ledger = perf_ledger_manager.filtered_ledger_for_scoring(hotkeys=testing_hotkeys)
    filtered_positions, _ = position_manager.filtered_positions_for_scoring(hotkeys=testing_hotkeys)
    # Get all possible positions, even beyond the lookback range
    success, eliminations = challengeperiod_manager.inspect(
        positions=filtered_positions,
        success_hotkeys=list(challengeperiod_manager.challengeperiod_success.keys()),
        ledger=filtered_ledger,
        inspection_hotkeys=inspection_hotkeys_dict,
        current_time=current_time,
    )

    prior_challengeperiod_miners = set(inspection_hotkeys_dict.keys())
    success_miners = set(success)
    eliminated_miners = set(eliminations.keys())

    post_challengeperiod_miners = prior_challengeperiod_miners - eliminated_miners - success_miners

    logger.info(f"{len(prior_challengeperiod_miners)} prior_challengeperiod_miners [{prior_challengeperiod_miners}]")
    logger.info(f"{len(success_miners)} success_miners [{success_miners}]")
    logger.info(f"{len(eliminated_miners)} challengeperiod_eliminations [{eliminations}]")
    logger.info(f"{len(post_challengeperiod_miners)} challenge period remaining [{post_challengeperiod_miners}]")