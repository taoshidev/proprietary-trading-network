# developer: trdougherty
# Copyright © 2024 Taoshi Inc
import numpy as np
from vali_config import ValiConfig

class HistoricalScoring:
    @staticmethod
    def historical_decay_return(
        return_value: float,
        time_fraction: float,
    ) -> float:
        """
        Args:
            return_value: float - the return of the miner
            time_fraction: float - the fraction of the lookback period since the position was closed.
        """
        time_fraction = np.clip(time_fraction, 0, 1)
        alpha = HistoricalScoring.permute_time_intensity(1 - time_fraction) # the closer to 1, the more pertinent the result to the final output
        return alpha * return_value + (1 - alpha) * 0.0 # 0.0 is just here for clarity, the default logged trade return - exp(0) = return of 1
    
    @staticmethod
    def permute_time_intensity(
        time_pertinence: float,
    ) -> float:
        """
        We don't always want linear behavior, perhaps much older returns should have significantly less weight.
        Args:
            fraction_of_time: float - the fraction of the lookback period since the position was closed. Higher means more time has passed.
        """
        time_intensity_coef = ValiConfig.HISTORICAL_DECAY_TIME_INTENSITY_COEFFICIENT
        time_intensity_coef = np.clip(time_intensity_coef, 0.001, 1) # 1 is a straight line, 0

        return 1 - ((1 - time_pertinence) ** time_intensity_coef)