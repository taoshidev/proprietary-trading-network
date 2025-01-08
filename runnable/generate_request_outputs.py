import time
import traceback

from runnable.generate_request_core import RequestCoreManager
from runnable.generate_request_minerstatistics import MinerStatisticsManager

from vali_objects.exceptions.corrupt_data_exception import ValiBkpCorruptDataException
from vali_objects.utils.challengeperiod_manager import ChallengePeriodManager
from vali_objects.utils.elimination_manager import EliminationManager
from vali_objects.utils.logger_utils import LoggerUtils
from time_util.time_util import TimeUtil
from json import JSONDecodeError

from vali_objects.utils.plagiarism_detector import PlagiarismDetector
from vali_objects.utils.position_manager import PositionManager
from vali_objects.utils.subtensor_weight_setter import SubtensorWeightSetter
from vali_objects.vali_dataclasses.perf_ledger import PerfLedgerManager

if __name__ == "__main__":

    logger = LoggerUtils.init_logger("generate_request_outputs")
    last_write_time = time.time()
    n_updates = 0
    try:
        while True:
            # NOTE: Reads from disk into memory every loop. Will not be needed once this logic is moved into validator
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

            rcm = RequestCoreManager(position_manager, subtensor_weight_setter, plagiarism_detector)
            msm = MinerStatisticsManager(position_manager, subtensor_weight_setter, plagiarism_detector)

            current_time = time.time()
            write_output = False
            # Check if it's time to write the legacy output
            if current_time - last_write_time >= 15:
                write_output = True

            current_time_ms = TimeUtil.now_in_millis()
            if write_output:
                # Generate the request outputs
                rcm.generate_request_core(time_now=current_time_ms)
                msm.generate_request_minerstatistics(time_now=current_time_ms, checkpoints=True)

            if write_output:
                last_validator_checkpoint_time = current_time

            # Log completion duration
            if write_output:
                n_updates += 1
                if n_updates % 10 == 0:
                    logger.info("Completed writing outputs in " + str(time.time() - current_time) + " seconds. n_updates: " + str(n_updates))
    except (JSONDecodeError, ValiBkpCorruptDataException):
        logger.error("error occurred trying to decode position json. Probably being written to simultaneously.")
        logger.error(traceback.format_exc())
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        # traceback
        logger.error(traceback.format_exc())
        time.sleep(10)
    # Sleep for a short time to prevent tight looping, adjust as necessary
    time.sleep(1)
