import time
import traceback

from setproctitle import setproctitle

from runnable.generate_request_core import RequestCoreManager
from runnable.generate_request_minerstatistics import MinerStatisticsManager

from vali_objects.exceptions.corrupt_data_exception import ValiBkpCorruptDataException
from vali_objects.utils.challengeperiod_manager import ChallengePeriodManager
from vali_objects.utils.elimination_manager import EliminationManager
from time_util.time_util import TimeUtil
from json import JSONDecodeError

from vali_objects.utils.plagiarism_detector import PlagiarismDetector
from vali_objects.utils.position_manager import PositionManager
from vali_objects.utils.subtensor_weight_setter import SubtensorWeightSetter
from vali_objects.vali_dataclasses.perf_ledger import PerfLedgerManager
import bittensor as bt

class RequestOutputGenerator:
    def __init__(self, running_deprecated=False, rcm=None, msm=None):
        self.running_deprecated = running_deprecated
        self.last_write_time_s = 0
        self.n_updates = 0
        if self.running_deprecated:
            self.repull_data_from_disk()
        else:
            self.rcm = rcm
            self.msm = msm


    def run_forever(self):
        setproctitle(f"vali_{self.__class__.__name__}")
        bt.logging.info(f'Running RequestOutputGenerator. running_deprecated: {self.running_deprecated}')
        while True:
            self.run_forever_wrap()
            # Sleep for a short time to prevent tight looping.
            time.sleep(1)

    def log_warning_message(self):
        bt.logging.warning("The generate script is no longer managed by pm2. Please update your repo and relaunch the"
                           " run.sh script with (same arguments). This will prevent this pm2 process from being "
                           "spawned and allow significant efficiency improvements by running this code from the"
                           "main validator loop.")

    def repull_data_from_disk(self):
        perf_ledger_manager = PerfLedgerManager(None)
        elimination_manager = EliminationManager(None, None, None)
        self.position_manager = PositionManager(None, None, elimination_manager=elimination_manager,
                                           challengeperiod_manager=None,
                                           perf_ledger_manager=perf_ledger_manager)
        challengeperiod_manager = ChallengePeriodManager(None, None, position_manager=self.position_manager)
        elimination_manager.position_manager = self.position_manager
        self.position_manager.challengeperiod_manager = challengeperiod_manager
        elimination_manager.challengeperiod_manager = challengeperiod_manager
        challengeperiod_manager.position_manager = self.position_manager
        perf_ledger_manager.position_manager = self.position_manager
        self.subtensor_weight_setter = SubtensorWeightSetter(
            config=None,
            wallet=None,
            metagraph=None,
            running_unit_tests=False,
            position_manager=self.position_manager,
        )
        self.plagiarism_detector = PlagiarismDetector(None, None, position_manager=self.position_manager)
        self.rcm = RequestCoreManager(self.position_manager, self.subtensor_weight_setter, self.plagiarism_detector)
        self.msm = MinerStatisticsManager(self.position_manager, self.subtensor_weight_setter, self.plagiarism_detector)

    def run_forever_wrap(self):
        try:
            # NOTE: Reads from disk into memory every loop. Will not be needed once this logic is moved into validator
            # Check if it's time to write the legacy output
            write_output = time.time() - self.last_write_time_s >= 15
            if write_output:
                current_time_ms = TimeUtil.now_in_millis()
                if self.running_deprecated:
                    self.log_warning_message()
                    self.repull_data_from_disk()

                # Generate the request outputs
                self.rcm.generate_request_core(time_now=current_time_ms)
                self.msm.generate_request_minerstatistics(time_now=current_time_ms, checkpoints=True)
                self.n_updates += 1
                now_s = time.time()
                if self.n_updates % 5 == 0:
                    t_i = current_time_ms / 1000.0
                    bt.logging.success("RequestOutputGenerator Completed a round of writing outputs in " + str(now_s - t_i) +
                                " seconds. n_updates total: " + str(self.n_updates))
                self.last_write_time_s = now_s

        except (JSONDecodeError, ValiBkpCorruptDataException):
            bt.logging.error("error occurred trying to decode position json. Probably being written to simultaneously.")
            bt.logging.error(traceback.format_exc())
        except Exception as e:
            bt.logging.error(f"An error occurred: {str(e)}")
            # traceback
            bt.logging.error(traceback.format_exc())
            time.sleep(10)


if __name__ == "__main__":
    bt.logging.enable_info()

    rog = RequestOutputGenerator(running_deprecated=True)
    rog.run_forever()
