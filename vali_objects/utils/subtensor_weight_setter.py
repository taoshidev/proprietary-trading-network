# developer: jbonilla
# Copyright © 2024 Taoshi Inc

import time
import bittensor as bt

from time_util.time_util import TimeUtil
from vali_config import ValiConfig
from shared_objects.cache_controller import CacheController
from vali_objects.utils.position_manager import PositionManager
from vali_objects.position import Position
from vali_objects.scoring.scoring import Scoring

class SubtensorWeightSetter(CacheController):
    def __init__(self, config, wallet, metagraph, running_unit_tests=False):
        super().__init__(config, metagraph, running_unit_tests=running_unit_tests)
        self.position_manager = PositionManager(
            metagraph=metagraph, running_unit_tests=running_unit_tests
        )
        self.wallet = wallet
        self.subnet_version = 200

    def set_weights(self):
        if not self.refresh_allowed(ValiConfig.SET_WEIGHT_REFRESH_TIME_MS):
            time.sleep(1)
            return

        bt.logging.info("running set weights")
        self._refresh_eliminations_in_memory()
        returns_per_netuid = self._calculate_return_per_netuid()
        bt.logging.info(f"return per uid [{returns_per_netuid}]")
        if len(returns_per_netuid) == 0:
            bt.logging.info("no returns to set weights with. Do nothing for now.")
        else:
            bt.logging.info("calculating new subtensor weights...")
            filtered_results = [ (k,v) for k,v in returns_per_netuid.items() ]
            scaled_transformed_list = Scoring.transform_and_scale_results(
                filtered_results
            )
            self._set_subtensor_weights(scaled_transformed_list)
        self.set_last_update_time()

    def _calculate_return_per_netuid(self) -> dict[str, list[float]]:
        """
        Calculate all returns for the .
        """
        return_per_netuid = {}

        # Note, eliminated miners will not appear in the dict below
        hotkey_positions = self.position_manager.get_all_miner_positions_by_hotkey(
            self.metagraph.hotkeys,
            sort_positions=True,
            eliminations=self.eliminations,
            acceptable_position_end_ms=TimeUtil.timestamp_to_millis(
                TimeUtil.generate_start_timestamp(
                    ValiConfig.SET_WEIGHT_LOOKBACK_RANGE_DAYS
                )
            ),
        )

        # have to have a minimum number of positions during the period
        # this removes anyone who got lucky on a couple trades
        current_time = TimeUtil.now_in_millis()
        for hotkey, positions in hotkey_positions.items():
            filtered_positions = self._filter_positions(positions)
            filter_miner_logic = self._filter_miner(filtered_positions, current_time)
            if filter_miner_logic:
                continue

            # compute the autmented returns for internal calculation
            per_position_return = (
                self.position_manager.get_return_per_closed_position_augmented(
                    filtered_positions, evaluation_time_ms=current_time
                )
            )

            # last_positional_return = per_position_return[-1]
            netuid = self.metagraph.hotkeys.index(hotkey)
            return_per_netuid[netuid] = per_position_return

        return return_per_netuid
    
    def _filter_miner(self, positions:list[Position], current_time:int):
        """
        Filter out miners who don't have enough positions to be considered for setting weights
        """
        if len(positions) == 0:
            return True
        
        # find the time when the first position was opened
        min_position_time = positions[0].close_ms
        for i in range(1, len(positions)):
            min_position_time = min(min_position_time, positions[i].close_ms)

        grace_period = (current_time - min_position_time) < ValiConfig.SET_WEIGHT_MINER_GRACE_PERIOD_MS
        if grace_period:
            return False

        if len(positions) < ValiConfig.SET_WEIGHT_MINIMUM_POSITIONS:
            return True
        
        # check that the position
        return False
    
    def _filter_positions(self, positions: list[Position]):
        """
        Filter out positions that are not within the lookback range.
        """
        filtered_positions = []
        for position in positions:
            if not position.is_closed_position:
                continue

            if position.close_ms - position.open_ms < ValiConfig.SET_WEIGHT_MINIMUM_POSITION_DURATION_MS:
                continue

            filtered_positions.append(position)
        return filtered_positions


    def _set_subtensor_weights(self, filtered_results: list[tuple[str, float]]):
        filtered_netuids = [x[0] for x in filtered_results]
        scaled_transformed_list = [x[1] for x in filtered_results]

        success, err_msg = self.subtensor.set_weights(
            netuid=self.config.netuid,
            wallet=self.wallet,
            uids=filtered_netuids,
            weights=scaled_transformed_list,
            version_key=self.subnet_version,
        )

        if success:
            bt.logging.success("Successfully set weights.")
        else:
            bt.logging.error(f"Failed to set weights. Error message: {err_msg}")
