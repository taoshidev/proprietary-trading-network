import multiprocessing
import time
import traceback

from setproctitle import setproctitle

from runnable.generate_request_core import RequestCoreManager
from runnable.generate_request_minerstatistics import MinerStatisticsManager

from vali_objects.utils.challengeperiod_manager import ChallengePeriodManager
from vali_objects.utils.elimination_manager import EliminationManager
from time_util.time_util import TimeUtil
from multiprocessing import Process
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
        self.msm_refresh_interval_ms = 10 * 1000
        self.rcm_refresh_interval_ms = 10 * 1000

        if self.running_deprecated:
            self.repull_data_from_disk()
        else:
            self.ctx = multiprocessing.get_context("spawn")
            self.rcm = rcm
            self.msm = msm


    def start_generation(self):
        bt.logging.info(f'Running RequestOutputGenerator. running_deprecated: {self.running_deprecated}')
        if self.running_deprecated:
            while True:
                current_time_ms = TimeUtil.now_in_millis()
                self.repull_data_from_disk()
                self.rcm.generate_request_core(time_now=current_time_ms)
                self.msm.generate_request_minerstatistics(time_now=current_time_ms, checkpoints=True)
        else:
            rcm_process = self.ctx.Process(target=self.run_rcm_loop, daemon=True)
            msm_process = self.ctx.Process(target=self.run_msm_loop, daemon=True)

            # Start both processes
            rcm_process.start()
            msm_process.start()
            while True:   # Stay alive
                time.sleep(100)

    def log_warning_message(self):
        bt.logging.warning("The generate script is no longer managed by pm2. Please update your repo and relaunch the"
                           " run.sh script with (same arguments). This will prevent this pm2 process from being "
                           "spawned and allow significant efficiency improvements by running this code from the"
                           "main validator loop.")

    def repull_data_from_disk(self):
        perf_ledger_manager = PerfLedgerManager(None)
        elimination_manager = EliminationManager(None, None, None)
        self.position_manager = PositionManager(None, None,
                                                elimination_manager=elimination_manager,
                                                challengeperiod_manager=None,
                                                perf_ledger_manager=perf_ledger_manager)
        challengeperiod_manager = ChallengePeriodManager(None, None,
                                                         position_manager=self.position_manager)
        elimination_manager.position_manager = self.position_manager
        self.position_manager.challengeperiod_manager = challengeperiod_manager
        elimination_manager.challengeperiod_manager = challengeperiod_manager
        challengeperiod_manager.position_manager = self.position_manager
        perf_ledger_manager.position_manager = self.position_manager
        self.subtensor_weight_setter = SubtensorWeightSetter(
            config=None,
            metagraph=None,
            running_unit_tests=False,
            position_manager=self.position_manager,
        )
        self.plagiarism_detector = PlagiarismDetector(None, None, position_manager=self.position_manager)
        self.rcm = RequestCoreManager(self.position_manager, self.subtensor_weight_setter, self.plagiarism_detector)
        self.msm = MinerStatisticsManager(self.position_manager, self.subtensor_weight_setter, self.plagiarism_detector)


    def run_rcm_loop(self):
        setproctitle(f"vali_RequestCoreManager")
        bt.logging.enable_info()
        bt.logging.info("Running RequestCoreManager process.")
        last_update_time_ms = 0
        n_updates = 0
        while True:
            try:
                current_time_ms = TimeUtil.now_in_millis()
                if current_time_ms - last_update_time_ms < self.rcm_refresh_interval_ms:
                    time.sleep(1)
                    continue
                self.rcm.generate_request_core(time_now=current_time_ms)
                n_updates += 1
                tf = TimeUtil.now_in_millis()
                if n_updates % 5 == 0:
                    bt.logging.success(f"RequestCoreManager completed a cycle in {tf - current_time_ms} ms.")
                last_update_time_ms = tf
            except Exception as e:
                bt.logging.error(f"RCM Error: {str(e)}")
                bt.logging.error(traceback.format_exc())
                time.sleep(10)

    def run_msm_loop(self):
        setproctitle(f"vali_MinerStatisticsManager")
        bt.logging.enable_info()
        bt.logging.info("Running MinerStatisticsManager process.")
        last_update_time_ms = 0
        n_updates = 0
        while True:
            try:
                current_time_ms = TimeUtil.now_in_millis()
                if current_time_ms - last_update_time_ms < self.msm_refresh_interval_ms:
                    time.sleep(1)
                    continue
                self.msm.generate_request_minerstatistics(time_now=current_time_ms, checkpoints=True)
                n_updates += 1
                tf = TimeUtil.now_in_millis()
                if n_updates % 5 == 0:
                    bt.logging.success(f"MinerStatisticsManager completed a cycle in {tf - current_time_ms} ms.")
                last_update_time_ms = tf
            except Exception as e:
                bt.logging.error(f"MSM Error: {str(e)}")
                bt.logging.error(traceback.format_exc())
                time.sleep(10)

if __name__ == "__main__":
    bt.logging.enable_info()
    rog = RequestOutputGenerator(running_deprecated=True)
    rog.start_generation()
