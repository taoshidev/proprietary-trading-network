import numpy as np
from time_util.time_util import TimeUtil
from vali_config import ValiConfig


class ReportingUtils:
    #This is very similar to Plagiarism_utils.build_state_matrix, possibly refactor
    def rasterize(cumulative_leverages, current_time=None, lookback_window=None, time_resolution=None):
        if current_time == None:
           current_time = TimeUtil.now_in_millis()

        if lookback_window == None:
            lookback_window = ValiConfig.PLAGIARISM_LOOKBACK_RANGE_MS

        if time_resolution == None:
            time_resolution = ValiConfig.PLAGIARISM_MATCHING_TIME_RESOLUTION_MS

        rasterized_positions = {}
        for key, cumulative_leverage in cumulative_leverages.items():
            rasterized_positions[key] = ReportingUtils.rasterize_cumulative_position(cumulative_leverage)
        return rasterized_positions

    def rasterize_cumulative_position(cumulative_leverage, current_time=None, lookback_window=None, time_resolution=None):
        #These variables will change with integration
        if current_time == None:
           current_time = TimeUtil.now_in_millis()

        if lookback_window == None:
           lookback_window = ValiConfig.PLAGIARISM_LOOKBACK_RANGE_MS

        if time_resolution == None:
            time_resolution = ValiConfig.PLAGIARISM_MATCHING_TIME_RESOLUTION_MS

        end_time = current_time
        start_time = current_time - lookback_window

        times = np.arange(start_time, end_time, time_resolution)
        rasterized_positions = np.zeros(len(times))
        
        for state in cumulative_leverage:
          time_criteria = (times >= state["start"]) & (times <= state["end"])
          rasterized_positions[time_criteria] = state["leverage"]
        return rasterized_positions