# developer: trdougherty
# Copyright © 2024 Taoshi Inc
import numpy as np
import bittensor as bt
from vali_config import ValiConfig
from vali_objects.position import Position
from shared_objects.cache_controller import CacheController
from vali_objects.utils.position_manager import PositionManager
from vali_objects.vali_dataclasses.order import Order
from vali_objects.scoring.scoring import Scoring
from time_util.time_util import TimeUtil
from vali_objects.vali_dataclasses.perf_ledger import PerfLedgerManager, PerfCheckpoint, PerfLedger

class ChallengePeriodManager(CacheController):
    def __init__(self, config, metagraph, running_unit_tests=False):
        super().__init__(config, metagraph, running_unit_tests=running_unit_tests)
        self.perf_manager = PerfLedgerManager(metagraph=metagraph, running_unit_tests=running_unit_tests)

    def refresh(self, current_time: int = None):
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
        
        if current_time < 1715366700000: # Friday, May 10, 2024 6:45:00 PM
            current_timestamp = TimeUtil.now_in_millis()                                                                                        
            self.challengeperiod_success = { k: current_timestamp for k in ['5DDBXwjobdGeiM3svUPUogVB6T3YVNC7nmin9p7EQGBrK2hA', '5C8Wegdus2cAcwSNU47MdiLXwZdewFkSv93xUWQP3wn32QJV', '5DqmvEK7Viv2NpEEJGJVuYaQEGpeSW6HAVxrNvV18CLxKve5', '5Cfx8PtVZxXcdVLBW6suwyvU8QmnZCHom5fVPfexJhkQh16U', '5FpypsPpSFUBpByFXMkJ34sV88PRjAKSSBkHkmGXMqFHR19Q', '5EAS8w6A4Nwc4quVQUs6xEDdhNSCNFgJ2ZzkHtJm83KthJaN', '5DaHdgTLPrGCdiNMosKq9GEpDmA6pPaMvNopXtnG28AtYghm', '5Fjz3ENZwDkn2txvryhPofbn2T3DbyHferTxvsastmmggFFb', '5GRFAJ3iwukm1CQpDTBsxTh2xjm227KtbVC1za8ukk6WqeyN', '5CnuyazUFWumVNTxoqMPc3bWk3FBj6eUbRjHDm8jsrEqaDkS', '5EPhd4PXgdQtxSXBUfB6FodJ2Uxy7TeVf8ZVGoP8gfGyCuqW', '5Da5hqCMSVgeGWmzeEnNrime3JKfgTpQmh7dXsdMP58dgeBd', '5E7DEGmFUewdJTnSh829jGc3SpSd295hhvUgNcNiQST6bw4A', '5C5GANtAKokcPvJBGyLcFgY5fYuQaXC3MpVt75codZbLLZrZ', '5EPDfdoeYhygXYEJ9xo8DV6kLuQZnrZgvH87sqci7WDM2j4g', '5CrGaMAv5guzzoyef6XBUPiBGhsrnox7nxPayV8DPzZh1zQL', '5GhCxfBcA7Ur5iiAS343xwvrYHTUfBjBi4JimiL5LhujRT9t', '5Eh9p81ioCeoTArv7kSa1PWcaXw33UdRjVQfLQsFPpn474GC', '5Dxqzduahnqw8q3XSUfTcEZGU7xmAsfJubhHZwvXVLN9fSjR', '5EZoATFyB3FdCEqEBuWSSDpdqFc8pePm6n5fMVRTuKpLu6Dr', '5G3ys2356ovgUivX3endMP7f37LPEjRkzDAM3Km8CxQnErCw', '5H8niLrzmxZUhAzRM29GNcnDyJPWEwujw5nbENuWcDV889W4', '5FqSBwa7KXvv8piHdMyVbcXQwNWvT9WjHZGHAQwtoGVQD3vo', '5CSHrvBiEJAFAAj7YAr5y8jzmpFajRsV9PahphPGi7P8PZrA', '5EF393sRCV3Q6SFNTpKQed8m3QDGRgfDvke8sUoH3kbLqGZS', '5DcgKr6s8z75sE4c69iMSM8adfRVex7A8BZe2mouVwMVRis4', '5DX8tSyGrx1QuoR1wL99TWDusvmmWgQW5su3ik2Sc8y8Mqu3', '5HCJ6okRkmCsu7iLEWotBxgcZy11RhbxSzs8MXT4Dei9osUx', '5F1sxW5apTPEYfDJUoHTRG4kGaUmkb3YVi5hwt5A9Fu8Gi6a', '5Df8YED2EoxY65B2voeCHzY9rn1R76DXB8Cq9f62CsGVRoU5', '5His3c7GyUKpWgRpuWAZfHKxtszZLQuTSMaEWM4NbkS1wsNm', '5HgBDcx8Z9oEWrGQm7obH4aPE2M5YXWA6S6MP1HtFHguUqek', '5HY6RF4RVDnX8CQ5JrcnzjM2rixQXZpW2eh48aNLUYT1V9LW', '5D7ZcGnnzT3yzwkZd94oGYXdHbCkrkrn7XELaXdR5dDHrtJX', '5HDmzyhrEco9w6Jv8eE3hDMcXSE4AGg1MuezPR4u2covxKwZ', '5Ec93qtHkKprEaA5EWXrmPmWppMeMiwaY868bpxfkH5ocBxi', '5Exax1W9RiNbARDejrthf4SK1FQ2u9DPUhCq9jm58gUysTy4', '5HBCKWiy27ncsMzX3aF1hP4yPqPJy86knbAoedeS1XymfSpn', '5Ct1J2jNxb9zeHpsj547BR1nZk4ZD51Bb599tzEWnxyEr4WR', '5DnViSacXqrP8FnQMtpAFGyahUPvU2A6pbrX7wcexb3bmVjb'] }
            self.challengeperiod_testing = { k: v for k,v in self.challengeperiod_testing.items() if k not in self.challengeperiod_success.keys() }

        else:
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
            else:
                if log:
                    print(f"Inspecting hotkey: {hotkey}")
                passing_criteria = self.screen_ledger(
                    ledger_element=ledger[hotkey], 
                    log=log
                )

            time_criteria = current_time - inspection_time < ValiConfig.SET_WEIGHT_CHALLENGE_PERIOD_MS

            # if the miner meets the criteria for passing, they are added to the passing list
            if passing_criteria:
                passing_miners.append(hotkey)
                continue

            # if the miner does not meet the criteria for passing, they are added to the failing list
            if not time_criteria:
                failing_miners.append(hotkey)
                continue

        return passing_miners, failing_miners
    

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
        
        ledger_cps = ledger_element.cps
        
        minimum_return = ValiConfig.SET_WEIGHT_MINER_CHALLENGE_PERIOD_RETURN_CPS_PERCENT
        minimum_omega = ValiConfig.SET_WEIGHT_MINER_CHALLENGE_PERIOD_OMEGA_CPS
        minimum_sortino = ValiConfig.SET_WEIGHT_MINER_CHALLENGE_PERIOD_SORTINO_CPS
        minimum_duration = ValiConfig.SET_WEIGHT_MINER_CHALLENGE_PERIOD_TOTAL_POSITION_DURATION
        minimum_volume_checkpoints = ValiConfig.SET_WEIGHT_MINER_CHALLENGE_PERIOD_VOLUME_CHECKPOINTS

        gains = [x.gain for x in ledger_cps]
        loss = [x.loss for x in ledger_cps]
        n_updates = [x.n_updates for x in ledger_cps]
        open_ms = [x.open_ms for x in ledger_cps]

        ## Compute the criteria for passing the challenge period
        omega_cps = Scoring.omega_cps(
            gains, 
            loss, 
            n_updates,
            open_ms
        )

        sortino_cps = Scoring.inverted_sortino_cps(
            gains,
            loss,
            n_updates,
            open_ms
        )

        return_cps = np.exp(Scoring.return_cps(
            gains,
            loss,
            n_updates,
            open_ms
        ))

        volume_cps = Scoring.checkpoint_volume_threshold_count(
            gains,
            loss,
            n_updates,
            open_ms
        )

        position_duration = sum(open_ms)

        ## Criteria
        omega_criteria = omega_cps >= minimum_omega
        sortino_criteria = sortino_cps >= minimum_sortino
        return_criteria = return_cps >= minimum_return
        duration_criteria = position_duration >= minimum_duration
        volume_crtieria = volume_cps >= minimum_volume_checkpoints

        if log:
            dayhours = (60 * 60 * 1000)
            viewable_return = 100 * (return_cps - 1)
            viewable_minimum_return = 100 * (minimum_return - 1)
            print(f"Omega: {omega_cps:.4f} >= {minimum_omega}: {omega_criteria}")
            print(f"Sortino: {sortino_cps:.3e} >= {minimum_sortino}: {sortino_criteria}")
            print(f"Return: {viewable_return:.4f}% >= {viewable_minimum_return:.2f}%: {return_criteria}")
            print(f"Duration (Hours): {position_duration / dayhours:.2f} >= {minimum_duration / dayhours:.2f}: {duration_criteria}")
            print(f"Volume Checkpoints: {volume_cps} >= {minimum_volume_checkpoints}: {volume_crtieria}")
            print()

        return omega_criteria and sortino_criteria and return_criteria and duration_criteria and volume_crtieria

