# developer: jbonilla
# Copyright © 2024 Taoshi Inc

import time
import bittensor as bt

from time_util.time_util import TimeUtil
from vali_config import ValiConfig
from vali_objects.scoring.scoring import Scoring
from shared_objects.cache_controller import CacheController
from vali_objects.utils.position_manager import PositionManager

class SubtensorWeightSetter(CacheController):
    def __init__(self, config, wallet, metagraph, running_unit_tests=False):
        super().__init__(config, metagraph, running_unit_tests=running_unit_tests)
        self.position_manager = PositionManager(metagraph=metagraph, running_unit_tests=running_unit_tests)
        self.wallet = wallet

    def set_weights(self):
        if not self.refresh_allowed(ValiConfig.SET_WEIGHT_REFRESH_TIME_MS):
            time.sleep(1)
            return

        bt.logging.info("running set weights")
        self._refresh_eliminations_in_memory_and_disk()
        return_per_netuid = self._calculate_return_per_netuid()
        bt.logging.info(f"return per uid [{return_per_netuid}]")
        if len(return_per_netuid) == 0:
            bt.logging.info("no returns to set weights with. Do nothing for now.")
        else:
            bt.logging.info("calculating new subtensor weights...")
            filtered_results = Scoring.filter_results(return_per_netuid)
            scaled_transformed_list = Scoring.transform_and_scale_results(filtered_results)
            self._set_subtensor_weights(scaled_transformed_list)
        self.set_last_update_time()

    def _calculate_return_per_netuid(self) -> dict[str, float]:
        return_per_netuid = {}
        netuid_returns = []

        # Note, eliminated miners will not appear in the dict below
        hotkey_positions = self.position_manager.get_all_miner_positions_by_hotkey(
            self.metagraph.hotkeys,
            sort_positions=True,
            eliminations=self.eliminations,
            acceptable_position_end_ms=TimeUtil.timestamp_to_millis(
                TimeUtil.generate_start_timestamp(ValiConfig.SET_WEIGHT_LOOKBACK_RANGE_DAYS)
            ),
        )

        # have to have a minimum number of positions during the period
        # this removes anyone who got lucky on a couple trades
        for hotkey, positions in hotkey_positions.items():
            if len(positions) <= ValiConfig.SET_WEIGHT_MINIMUM_POSITIONS:
                continue
            per_position_return = self.position_manager.get_return_per_closed_position(positions)
            last_positional_return = per_position_return[-1]
            netuid_returns.append(last_positional_return)
            netuid = self.metagraph.hotkeys.index(hotkey)
            return_per_netuid[netuid] = last_positional_return

        return return_per_netuid
    
    def _set_subtensor_weights(self, filtered_results: list[tuple[str, float]]):
        filtered_netuids = [ x[0] for x in filtered_results ]
        scaled_transformed_list = [ x[1] for x in filtered_results ]

        result = self.subtensor.set_weights(
            netuid=self.config.netuid,
            wallet=self.wallet,
            uids=filtered_netuids,
            weights=scaled_transformed_list,
        )

        if result:
            bt.logging.success("Successfully set weights.")
        else:
            bt.logging.error("Failed to set weights.")