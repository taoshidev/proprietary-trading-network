# developer: jbonilla
# Copyright © 2023 Taoshi Inc

import time
import bittensor as bt
import numpy as np
from scipy.stats import yeojohnson

from time_util.time_util import TimeUtil
from vali_config import ValiConfig
from vali_objects.scaling.scaling import Scaling
from shared_objects.challenge_utils import ChallengeBase
from vali_objects.utils.position_utils import PositionUtils

class SubtensorWeightSetter(ChallengeBase):
    def __init__(self, config, wallet, metagraph):
        super().__init__(config, metagraph)
        self.wallet = wallet

    def set_weights(self):
        # TODO: Should this check for eliminated miners and set their weight to 0 / exclude them.
        if time.time() - self.get_last_update_time() < ValiConfig.SET_WEIGHT_REFRESH_TIME_S:
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
            filtered_results, filtered_netuids = self._filter_results(return_per_netuid)
            scaled_transformed_list = self._transform_and_scale_results(filtered_results)
            self._set_subtensor_weights(filtered_netuids, scaled_transformed_list)
        self.set_last_update_time()

    def _calculate_return_per_netuid(self):
        return_per_netuid = {}
        netuid_returns = []
        netuids = []

        hotkey_positions = PositionUtils.get_all_miner_positions_by_hotkey(
            self.metagraph.hotkeys,
            sort_positions=True,
            eliminations=self.eliminations,
            acceptable_position_end_ms=TimeUtil.timestamp_to_millis(
                TimeUtil.generate_start_timestamp(ValiConfig.SET_WEIGHT_LOOKBACK_RANGE_DAYS)
            ),
        )

        # have to have a minimum number of positions during the period
        # this removes anyone who got lucky on a couple trades
        # TODO: should this be multiplying each position return by the previous? Seems to be overwriting the previous value.
        for hotkey, positions in hotkey_positions.items():
            if len(positions) <= ValiConfig.SET_WEIGHT_MINIMUM_POSITIONS:
                continue
            per_position_return = PositionUtils.get_return_per_closed_position(positions)
            last_positional_return = 1
            if len(per_position_return) > 0:
                last_positional_return = per_position_return[len(per_position_return) - 1]
            netuid_returns.append(last_positional_return)
            netuid = self.metagraph.hotkeys.index(hotkey)
            return_per_netuid[netuid] = last_positional_return
            netuids.append(netuid)

        return return_per_netuid

    def _filter_results(self, return_per_netuid):
        mean = np.mean(list(return_per_netuid.values()))
        std_dev = np.std(list(return_per_netuid.values()))

        lower_bound = mean - 3 * std_dev
        bt.logging.debug(f"returns lower bound: [{lower_bound}]")

        if lower_bound < 0:
            lower_bound = 0

        filtered_results = [(k, v) for k, v in return_per_netuid.items() if lower_bound < v]
        filtered_netuids = np.array([x[0] for x in filtered_results])
        bt.logging.info(f"filtered results list [{filtered_results}]")
        bt.logging.info(f"filtered netuids list [{filtered_netuids}]")
        return filtered_results, filtered_netuids

    def _transform_and_scale_results(self, filtered_results):
        filtered_scores = np.array([x[1] for x in filtered_results])
        bt.logging.success("calculated filtered_scores {filtered_scores}".format(filtered_scores=filtered_scores))
        # Normalize the list using Z-score normalization
        transformed_results = yeojohnson(filtered_scores, lmbda=500)
        bt.logging.success("calculated transformed_results {transformed_results}".format(transformed_results=transformed_results))
        if len(transformed_results) == 0:
            return []
        scaled_transformed_list = Scaling.min_max_scalar_list(transformed_results)
        return scaled_transformed_list

    def _set_subtensor_weights(self, filtered_netuids, scaled_transformed_list):
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
