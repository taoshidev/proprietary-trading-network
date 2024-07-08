# developer: jbonilla
# Copyright © 2024 Taoshi Inc

from time_util.time_util import TimeUtil
from vali_config import ValiConfig
from shared_objects.cache_controller import CacheController
from vali_objects.utils.position_manager import PositionManager

from vali_objects.utils.plagiarism_utils import PlagiarismUtils

class PlagiarismDetector(CacheController):
    def __init__(self, config, metagraph, running_unit_tests=False):
        super().__init__(config, metagraph, running_unit_tests=running_unit_tests)
        self.position_manager = PositionManager(metagraph=metagraph, running_unit_tests=running_unit_tests)

    def detect(
            self, 
            hotkeys = None,
        ) -> bool:
        """
        Kick off the plagiarism detection process.
        """

        current_time = TimeUtil.now_in_millis()
        if hotkeys is None:
            hotkeys = self.metagraph.hotkeys

        hotkey_positions = self.position_manager.get_all_miner_positions_by_hotkey(
            hotkeys,
            sort_positions=True,
            eliminations=self.eliminations,
            acceptable_position_end_ms=TimeUtil.timestamp_to_millis(
                TimeUtil.generate_start_timestamp(
                    ValiConfig.SET_WEIGHT_LOOKBACK_RANGE_DAYS
                )
            ),
        )

        elimination_mapping: dict[str, bool] = PlagiarismUtils.generate_elimination_mapping(
            hotkey_positions
        )

    def eliminate(self, elimination_mapping: dict[str, bool]):
        """
        Eliminate miners based on the elimination mapping.
        """
        for hotkey, elimination in elimination_mapping.items():
            self.eliminations[hotkey] = elimination

        

        

