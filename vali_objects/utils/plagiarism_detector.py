# developer: jbonilla
# Copyright © 2024 Taoshi Inc

from time_util.time_util import TimeUtil
from vali_objects.utils.plagiarism_definitions import FollowPercentage, LagDetection, CopySimilarity, TwoCopySimilarity, \
    ThreeCopySimilarity
from vali_objects.vali_config import ValiConfig
from shared_objects.cache_controller import CacheController
from vali_objects.utils.position_manager import PositionManager
import time
import traceback

import bittensor as bt

from vali_objects.utils.plagiarism_pipeline import PlagiarismPipeline

class PlagiarismDetector(CacheController):
    def __init__(self, config, metagraph, running_unit_tests=False, shutdown_dict=None):
        super().__init__(config, metagraph, running_unit_tests=running_unit_tests)
        self.plagiarism_classes = [FollowPercentage,
                                   LagDetection,
                                   CopySimilarity,
                                   TwoCopySimilarity,
                                   ThreeCopySimilarity]
        self.position_manager = PositionManager(metagraph=metagraph, running_unit_tests=running_unit_tests)
        self.plagiarism_pipeline = PlagiarismPipeline(self.plagiarism_classes)
        self.shutdown_dict = shutdown_dict

    def run_update_loop(self):
        while not self.shutdown_dict:
            try:
                if self.refresh_allowed(ValiConfig.PLAGIARISM_REFRESH_TIME_MS):
                    self.detect()
                    self.set_last_update_time(skip_message=False)  # TODO: set True

            except Exception as e:
                # Handle exceptions or log errors
                bt.logging.error(f"Error during plagiarism update: {e}. Please alert a team member ASAP!")
                bt.logging.error(traceback.format_exc())
                time.sleep(30)
            time.sleep(1)

    def detect(self, hotkeys = None, hotkey_positions = None) -> None:
        """
        Kick off the plagiarism detection process.
        """
        if self.running_unit_tests:
            current_time = ValiConfig.PLAGIARISM_LOOKBACK_RANGE_MS
        else:
            current_time = TimeUtil.now_in_millis()
        if hotkeys is None:
            hotkeys = self.metagraph.hotkeys
        if hotkey_positions is None:
            hotkey_positions = self.position_manager.get_all_miner_positions_by_hotkey(
                hotkeys,
                eliminations=self.eliminations,
            )

        bt.logging.info("Starting Plagiarism Detection")
        plagiarism_data, raster_positions, positions = self.plagiarism_pipeline.run_reporting(positions=hotkey_positions, current_time=current_time)


        self.write_plagiarism_scores_to_disk(plagiarism_data)
        self.write_plagiarism_raster_to_disk(raster_positions)
        self.write_plagiarism_positions_to_disk(positions)

        # elimination_mapping: dict[str, bool] = PlagiarismUtils.generate_elimination_mapping(  # noqa: F841
        #    hotkey_positions
        #)

        bt.logging.info("Plagiarism Detection Complete")

    def eliminate(self, elimination_mapping: dict[str, bool]):
        """
        Eliminate miners based on the elimination mapping.
        """
        for hotkey, elimination in elimination_mapping.items():
            self.eliminations[hotkey] = elimination

        

        

