# developer: jbonilla
# Copyright © 2024 Taoshi Inc

from time_util.time_util import TimeUtil
from vali_config import ValiConfig
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

    def detect_plagiarism(self, current_time = None):
        if current_time - self.last_time_plagiarism_run < ValiConfig.PLAGIARISM_REFRESH_TIME_MS:
            time.sleep(1)
            return
        else:
            self.detect(current_time=current_time)

    def detect(
            self, 
            hotkeys = None,
            current_time = None,
            hotkey_positions = None #TODO revert this after testing
        ) -> None:
        """
        Kick off the plagiarism detection process.
        """        
        if self.running_unit_tests:
            current_time = ValiConfig.PLAGIARISM_LOOKBACK_RANGE_MS
        elif current_time == None:
            current_time = TimeUtil.now_in_millis()  # noqa: F841

        if hotkeys is None:
            hotkeys = self.metagraph.hotkeys
        bt.logging.info("Starting plagiarism Detection")
        if hotkey_positions is None:
            hotkey_positions = self.position_manager.get_all_miner_positions_by_hotkey(
                hotkeys,
                eliminations=self.eliminations,
                #acceptable_position_end_ms=TimeUtil.timestamp_to_millis(
                #    TimeUtil.generate_start_timestamp(
                #        ValiConfig.SET_WEIGHT_LOOKBACK_RANGE_DAYS
            )

        plagiarists, raster_positions, positions = PlagiarismPipeline.run_reporting(positions=hotkey_positions, current_time=current_time) #returns list of plagiarist file objects, raster_vectors and positions

        self.plagiarists_data = plagiarists
        self.plagiarism_raster = raster_positions
        self.plagiarism_positions = positions

        self.write_all_plagiarists_to_disk()
        self.write_plagiarism_raster_to_disk()
        self.write_plagiarism_positions_to_disk()
        #elimination_mapping: dict[str, bool] = PlagiarismUtils.generate_elimination_mapping(  # noqa: F841
        #    hotkey_positions
        #)
        # TODO put this in a helper function
        self.last_time_plagiarism_run = TimeUtil.now_in_millis()
        bt.logging.info("Plagiarism detection complete")


    def eliminate(self, elimination_mapping: dict[str, bool]):
        """
        Eliminate miners based on the elimination mapping.
        """
        for hotkey, elimination in elimination_mapping.items():
            self.eliminations[hotkey] = elimination

        

        

