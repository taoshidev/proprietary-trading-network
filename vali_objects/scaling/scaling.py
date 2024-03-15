# developer: Taoshidev
# Copyright © 2023 Taoshi Inc

from typing import List

import numpy as np
from numpy import ndarray
from sklearn.preprocessing import MinMaxScaler

from vali_config import ValiConfig


class Scaling:

    @staticmethod
    def count_decimal_places(number):
        number_str = str(number)

        if '.' in number_str:
            integer_part, fractional_part = number_str.split('.')
            return len(fractional_part)
        else:
            # If there's no decimal point, return 0
            return 0

    @staticmethod
    def scale_values_exp(v: np) -> (float, np):
        avg = np.mean(v)
        k = ValiConfig.SCALE_FACTOR_EXP
        return float(avg), np.array([np.tanh(k * (x - avg)) for x in v])

    @staticmethod
    def unscale_values_exp(avg: float, decimal_places: int, v: np) -> np:
        k = ValiConfig.SCALE_FACTOR_EXP
        return np.array([np.round(avg + (1 / k) * np.arctanh(x), decimals=decimal_places) for x in v])

    @staticmethod
    def scale_values(scores: np, vmin: ndarray | float = None, vmax: ndarray | float = None):
        if vmin is None or vmax is None:
            vmin = np.min(scores)
            vmax = np.max(scores)
        normalized_scores = (scores - vmin) / (vmax - vmin)
        return vmin, vmax, (normalized_scores / ValiConfig.SCALE_FACTOR) + ValiConfig.SCALE_SHIFT

    @staticmethod
    def unscale_values(vmin: float, vmax: float, decimal_places: int, normalized_scores: np):
        denormalized_scores = np.round((((normalized_scores - ValiConfig.SCALE_SHIFT) * ValiConfig.SCALE_FACTOR)
                                        * (vmax - vmin)) + vmin, decimals=decimal_places)
        return denormalized_scores

    @staticmethod
    def scale_data_structure(ds: List[List]) -> (List[float], List[float], List[float], np):
        scaled_data_structure = []
        vmins = []
        vmaxs = []
        dp_decimal_places = []

        for dp in ds:
            vmin, vmax, scaled_data_point = Scaling.scale_values(np.array(dp))
            vmins.append(vmin)
            vmaxs.append(vmax)
            dp_decimal_places.append(Scaling.count_decimal_places(dp[0]))
            scaled_data_structure.append(scaled_data_point)
        return vmins, vmaxs, dp_decimal_places, np.array(scaled_data_structure)

    @staticmethod
    def unscale_data_structure(avgs: List[float], dp_decimal_places: List[int], sds: np) -> np:
        usds = []
        for i, dp in enumerate(sds):
            usds.append(Scaling.unscale_values_exp(avgs[i], dp_decimal_places[i], dp))
        return usds

    @staticmethod
    def scale_ds_with_ts(ds: List[List]) -> (List[float], List[float], List[float], np):
        ds_ts = ds[0]
        vmins, vmaxs, dp_decimal_places, scaled_data_structure = Scaling.scale_data_structure(ds[1:])
        sds_list = scaled_data_structure.tolist()
        sds_list.insert(0, ds_ts)
        return vmins, vmaxs, dp_decimal_places, np.array(sds_list)

    @staticmethod
    def min_max_scalar_list(l):
        original_values_2d = [[val] for val in l]
        scaler = MinMaxScaler()

        scaled_values_2d = scaler.fit_transform(original_values_2d)
        scaled_values = [val[0] for val in scaled_values_2d]

        return scaled_values
