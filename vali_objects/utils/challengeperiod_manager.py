# developer: trdougherty
# Copyright © 2024 Taoshi Inc
import numpy as np
import time
import bittensor as bt
from vali_config import ValiConfig
from vali_objects.position import Position
from shared_objects.cache_controller import CacheController
from vali_objects.utils.position_manager import PositionManager
from vali_objects.vali_dataclasses.order import Order
from vali_objects.scoring.scoring import Scoring, ScoringUnit
from time_util.time_util import TimeUtil
from vali_objects.vali_dataclasses.perf_ledger import PerfLedgerManager, PerfCheckpoint, PerfLedger

class ChallengePeriodManager(CacheController):
    def __init__(self, config, metagraph, running_unit_tests=False):
        super().__init__(config, metagraph, running_unit_tests=running_unit_tests)
        self.perf_manager = PerfLedgerManager(metagraph=metagraph, running_unit_tests=running_unit_tests)

    def refresh(self, current_time: int = None):
        if not self.refresh_allowed(ValiConfig.CHALLENGE_PERIOD_REFRESH_TIME_MS):
            time.sleep(1)
            return

        # The refresh should just read the current eliminations
        self.eliminations = self.get_filtered_eliminations_from_disk()

        # Collect challengeperiod and update with new eliminations criteria
        self._refresh_challengeperiod_in_memory_and_disk(eliminations=self.eliminations)

        # challenge period adds to testing if not in eliminated, already in the challenge period, or in the new eliminations list from disk
        self._add_challengeperiod_testing_in_memory_and_disk(
            new_hotkeys = self.metagraph.hotkeys,
            eliminations = self.eliminations,
            current_time = current_time
        )

        ledger = self.perf_manager.load_perf_ledgers_from_disk()
        challengeperiod_success, challengeperiod_eliminations = self.inspect(
            ledger = ledger,
            inspection_hotkeys = self.challengeperiod_testing,
            current_time = current_time
        )
        
        # Moves challenge period testing to challenge period success in memory
        self._promote_challengeperiod_in_memory(hotkeys = challengeperiod_success, current_time = current_time)
        self._demote_challengeperiod_in_memory(hotkeys = challengeperiod_eliminations)

        ## Now sync challenge period with the disk
        self._write_challengeperiod_from_memory_to_disk()
        self._write_eliminations_from_memory_to_disk()

    def inspect(
        self,
        ledger: dict[str, PerfLedger],
        inspection_hotkeys: dict[str, int] = None,
        current_time: int = None,
        eliminations: list[str] = None,
        log: bool = False,
    ):
        """
        Runs a screening process to elminate miners who didn't pass the challenge period. Does not modify the challenge period in memory.
        """
        if inspection_hotkeys is None:
            return [], [] # no hotkeys to inspect

        if current_time is None:
            current_time = TimeUtil.now_in_millis()

        if eliminations is None:
            eliminations = self.eliminations

        passing_miners = []
        failing_miners = []
        for hotkey, inspection_time in inspection_hotkeys.items():
            ## Check the criteria for passing the challenge period
            if hotkey not in ledger:
                passing_criteria = False
                drawdown_criteria = True
            else:
                if log:
                    print(f"Inspecting hotkey: {hotkey}")
                ledger_element = ledger[hotkey]
                passing_criteria = self.screen_ledger(
                    ledger_element=ledger_element, 
                    log=log
                )
                drawdown_criteria = self.screen_ledger_drawdown(
                    ledger_element=ledger_element,
                    log=log
                )

            time_criteria = current_time - inspection_time < ValiConfig.SET_WEIGHT_CHALLENGE_PERIOD_MS

            # If the miner's drawdown is below the threshold at any time, they are added to the failing list
            if not drawdown_criteria:
                failing_miners.append(hotkey)
                continue

            # if the miner meets the criteria for passing, they are added to the passing list
            if passing_criteria:
                passing_miners.append(hotkey)
                continue

            # if the miner does not meet the criteria for passing within the time frame, they are added to the failing list
            if not time_criteria:
                failing_miners.append(hotkey)
                continue

        return passing_miners, failing_miners
    
    def screen_ledger_drawdown(
        self,
        ledger_element: PerfLedger,
        log: bool = False
    ) -> bool:
        """
        Runs a screening process to elminate miners who didn't pass the challenge period.
        """
        if ledger_element is None:
            return False

        if len(ledger_element.cps) == 0:
            return False

        minimum_drawdown = ValiConfig.SET_WEIGHT_MINER_CHALLENGE_PERIOD_DRAWDOWN

        ## Create a scoring unit from the ledger element
        if not ledger_element.cps or len(ledger_element.cps) == 0:
            return True

        minimum_drawdown_seen = min([ x.mdd for x in ledger_element.cps ])
        if log:
            print(f"Within Drawdown Criteria: {minimum_drawdown_seen:.4f} > {minimum_drawdown}: {minimum_drawdown_seen > minimum_drawdown}")
            print()

        if minimum_drawdown_seen <= minimum_drawdown:
            return False

        return True
    
    def screen_ledger(
        self, 
        ledger_element: PerfLedger,
        log: bool = False
    ) -> bool:
        """
        Runs a screening process to elminate miners who didn't pass the challenge period.
        """
        if ledger_element is None:
            return False

        if len(ledger_element.cps) == 0:
            return False
                
        minimum_return = ValiConfig.SET_WEIGHT_MINER_CHALLENGE_PERIOD_RETURN_CPS_PERCENT
        minimum_omega = ValiConfig.SET_WEIGHT_MINER_CHALLENGE_PERIOD_OMEGA_CPS
        minimum_duration = ValiConfig.SET_WEIGHT_MINER_CHALLENGE_PERIOD_TOTAL_POSITION_DURATION
        minimum_volume_checkpoints = ValiConfig.SET_WEIGHT_MINER_CHALLENGE_PERIOD_VOLUME_CHECKPOINTS

        ## Create a scoring unit from the ledger element
        scoringunit = ScoringUnit.from_perf_ledger(ledger_element)

        ## Compute the criteria for passing the challenge period
        omega_cps = Scoring.omega_cps(scoringunit)
        return_cps = np.exp(Scoring.return_cps(scoringunit))
        position_duration = sum(scoringunit.open_ms)
        volume_cps = Scoring.checkpoint_volume_threshold_count(scoringunit)

        ## Criteria
        omega_criteria = omega_cps >= minimum_omega
        return_criteria = return_cps >= minimum_return
        duration_criteria = position_duration >= minimum_duration
        volume_crtieria = volume_cps >= minimum_volume_checkpoints

        if log:
            dayhours = (60 * 60 * 1000)
            viewable_return = 100 * (return_cps - 1)
            viewable_minimum_return = 100 * (minimum_return - 1)
            print(f"Omega: {omega_cps:.4f} >= {minimum_omega}: {omega_criteria}")
            print(f"Return: {viewable_return:.4f}% >= {viewable_minimum_return:.2f}%: {return_criteria}")
            print(f"Duration (Hours): {position_duration / dayhours:.2f} >= {minimum_duration / dayhours:.2f}: {duration_criteria}")
            print(f"Volume Checkpoints: {volume_cps} >= {minimum_volume_checkpoints}: {volume_crtieria}")

        return omega_criteria and return_criteria and duration_criteria and volume_crtieria

