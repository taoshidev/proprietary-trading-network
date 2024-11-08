import numpy as np
from time_util.time_util import TimeUtil
from vali_objects.vali_config import ValiConfig

class ReportingUtils:
    @staticmethod
    def rasterize(cumulative_leverages, current_time=None, lookback_window=None, time_resolution=None):
        if current_time is None:
           current_time = TimeUtil.now_in_millis()

        if lookback_window is None:
            lookback_window = ValiConfig.PLAGIARISM_LOOKBACK_RANGE_MS

        if time_resolution is None:
            time_resolution = ValiConfig.PLAGIARISM_MATCHING_TIME_RESOLUTION_MS

        rasterized_positions = {}
        for id, cumulative_leverage in cumulative_leverages.items():
            rasterized_positions[id] = ReportingUtils.rasterize_cumulative_position(cumulative_leverage, current_time=current_time, lookback_window=lookback_window, time_resolution=time_resolution)
        return rasterized_positions

    @staticmethod
    def rasterize_cumulative_position(cumulative_leverage, current_time=None, lookback_window=None, time_resolution=None):

        if current_time is None:
           current_time = TimeUtil.now_in_millis()

        if lookback_window is None:
           lookback_window = ValiConfig.PLAGIARISM_LOOKBACK_RANGE_MS

        if time_resolution is None:
            time_resolution = ValiConfig.PLAGIARISM_MATCHING_TIME_RESOLUTION_MS

        end_time = current_time
        start_time = current_time - lookback_window
        times = np.arange(start_time, end_time, time_resolution)
        rasterized_positions = np.zeros(len(times))
        
        for state in cumulative_leverage:
          time_criteria = (times >= state["start"]) & (times <= state["end"])
          rasterized_positions[time_criteria] = state["leverage"]
        return rasterized_positions
