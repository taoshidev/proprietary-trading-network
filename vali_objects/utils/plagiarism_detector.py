# developer: jbonilla
# Copyright © 2024 Taoshi Inc

from time_util.time_util import TimeUtil
from vali_objects.vali_config import ValiConfig
from shared_objects.cache_controller import CacheController
from vali_objects.utils.position_manager import PositionManager
import time

import bittensor as bt

from vali_objects.utils.plagiarism_pipeline import PlagiarismPipeline

class PlagiarismDetector(CacheController):
    
    def __init__(self, config, metagraph, running_unit_tests=False):
        super().__init__(config, metagraph, running_unit_tests=running_unit_tests)
        self.position_manager = PositionManager(metagraph=metagraph, running_unit_tests=running_unit_tests)
        self.last_time_plagiarism_run = 0

    def detect_plagiarism(self, current_time=None):
            if self.plagiarism_detection_allowed(ValiConfig.PLAGIARISM_REFRESH_TIME_MS):
                self.detect(current_time=current_time)
            else:
                time.sleep(1)
    def detect(
            self, 
            hotkeys = None,
            current_time = None,
            hotkey_positions = None
        ) -> None:
        """
        Kick off the plagiarism detection process.
        """        
        if self.running_unit_tests:
            current_time = ValiConfig.PLAGIARISM_LOOKBACK_RANGE_MS
        elif current_time is None:
            current_time = TimeUtil.now_in_millis()  # noqa: F841

        if hotkeys is None:
            hotkeys = self.metagraph.hotkeys

        bt.logging.info("Starting Plagiarism Detection")

        if hotkey_positions is None:
            hotkey_positions = self.position_manager.get_all_miner_positions_by_hotkey(
                hotkeys,
                eliminations=self.eliminations,
            )

        plagiarism_data, raster_positions, positions = PlagiarismPipeline.run_reporting(positions=hotkey_positions, current_time=current_time)

        self.plagiarism_data = plagiarism_data
        self.plagiarism_raster = raster_positions
        self.plagiarism_positions = positions

        self.write_plagiarism_scores_to_disk()
        self.write_plagiarism_raster_to_disk()
        self.write_plagiarism_positions_to_disk()

        # elimination_mapping: dict[str, bool] = PlagiarismUtils.generate_elimination_mapping(  # noqa: F841
        #    hotkey_positions
        #)

        self.last_time_plagiarism_run = TimeUtil.now_in_millis()

        bt.logging.info("Plagiarism Detection Complete")

    def eliminate(self, elimination_mapping: dict[str, bool]):
        """
        Eliminate miners based on the elimination mapping.
        """
        for hotkey, elimination in elimination_mapping.items():
            self.eliminations[hotkey] = elimination

        

        

