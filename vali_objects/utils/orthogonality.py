import os
import numpy as np
from typing import Callable
from vali_objects.vali_config import ValiConfig
from vali_objects.utils.functional_utils import FunctionalUtils

class Orthogonality:
    @staticmethod
    def similarity(v1: list[float], v2: list[float]) -> float:
        """
        Calculate the similarity between two vectors.
        :param v1: First vector.
        :param v2: Second vector.
        :return: Similarity between the two vectors.
        """
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    @staticmethod
    def convolutional_similarity(v1: list[float], v2: list[float], max_shift: int = 10) -> np.array:
        """
        Determine the rolling similarity between two vectors
        with potentially variable length.
        :param v1: First vector.
        :param v2: Second vector.
        :return: Similarity between the two vectors.
        """
        return np.array([Orthogonality.similarity(v1[i:], v2[:len(v1) - i]) for i in range(max_shift)])
    
    @staticmethod
    def duration_metric(v: list[float]) -> float:
        """
        Determine the duration metric of a vector.
        :param v: Vector.
        :return: Duration metric of the vector.
        """
        return len(v)

    @staticmethod
    def time_preference(v1: list[float], v2: list[float], max_shift: int = 10) -> float:
        """
        Determine how preferred the first vector is over the second vector based on time longevity.
        :param v1: First vector.
        :param v2: Second vector.
        :return: Time preference between the two vectors.
        """
        time_preference_shift = ValiConfig.TIME_PREFERENCE_SHIFT
        time_preference_spread = ValiConfig.TIME_PREFERENCE_SPREAD

        # Determine the metric to use for the time preference
        v1_metric = Orthogonality.duration_metric(v1)
        v2_metric = Orthogonality.duration_metric(v2)

        # Establish a time preference score for each vector
        v1_preference = FunctionalUtils.sigmoid(
            v1_metric, 
            shift=time_preference_shift, 
            spread=time_preference_spread
        )

        v2_preference = FunctionalUtils.sigmoid(
            v2_metric, 
            shift=time_preference_shift, 
            spread=time_preference_spread
        )

        former_preference = v1_preference / (v1_preference + v2_preference)
        latter_preference = v2_preference / (v1_preference + v2_preference)

        # Calculate the time preference between the two vectors
        time_preference = former_preference - latter_preference

        # Return the time preference between the two vectors
        return time_preference
    
    @staticmethod
    def size_preference(v1: list[float], v2: list[float], max_shift: int = 10) -> float:
        """
        Determine how preferred the first vector is over the second vector based on size.
        :param v1: First vector.
        :param v2: Second vector.
        :return: Size preference between the two vectors.
        """
        size_preference_shift = ValiConfig.SIZE_PREFERENCE_SHIFT
        size_preference_spread = ValiConfig.SIZE_PREFERENCE_SPREAD

        # Determine the metric to use for the size preference
        v1_metric = Orthogonality.duration_metric(v1)
        v2_metric = Orthogonality.duration_metric(v2)

        # Establish a size preference score for each vector
        v1_preference = FunctionalUtils.sigmoid(
            v1_metric, 
            shift=size_preference_shift, 
            spread=size_preference_spread
        )

        v2_preference = FunctionalUtils.sigmoid(
            v2_metric, 
            shift=size_preference_shift, 
            spread=size_preference_spread
        ) 

        former_preference = v1_preference / (v1_preference + v2_preference)
        latter_preference = v2_preference / (v1_preference + v2_preference)

        # Calculate the size preference between the two vectors
        size_preference = former_preference - latter_preference

        # Return the size preference between the two vectors
        return size_preference
    

    @staticmethod
    def pairwise_pref(returns: dict[str, list[float]], metric_fn: Callable) -> dict[tuple[str, str], float]:
        """
        Compute pairwise preferences for all unique pairs using the given metric function.
        Returns a dict mapping (hotkey1, hotkey2) to the preference value (only for hotkey1 < hotkey2).
        """
        keys = list(returns.keys())
        n = len(keys)
        prefs = {}
        for i in range(n):
            for j in range(i + 1, n):
                k1, k2 = keys[i], keys[j]
                prefs[(k1, k2)] = metric_fn(returns[k1], returns[k2])
        return prefs

    @staticmethod
    def time_pref(returns: dict[str, list[float]]) -> dict[tuple[str, str], float]:
        """
        Pairwise time preference for all miners.
        """
        return Orthogonality.pairwise_pref(returns, Orthogonality.time_preference)

    @staticmethod
    def size_pref(returns: dict[str, list[float]]) -> dict[tuple[str, str], float]:
        """
        Pairwise size preference for all miners.
        """
        return Orthogonality.pairwise_pref(returns, Orthogonality.size_preference)

    @staticmethod
    def sim_pref(returns: dict[str, list[float]]) -> dict[tuple[str, str], float]:
        """
        Pairwise similarity for all miners.
        """
        return Orthogonality.pairwise_pref(returns, Orthogonality.convolutional_similarity)

    @staticmethod
    def full_pref(returns: dict[str, list[float]]) -> dict[str, float]:
        """
        For a dict of miners' daily returns, return a dict mapping hotkey to the sum of all its pairwise compositional preferences (size, time, similarity) with all other miners.
        """
        size_prefs = Orthogonality.size_pref(returns)
        time_prefs = Orthogonality.time_pref(returns)
        sim_prefs = Orthogonality.sim_pref(returns)
        # Aggregate all pairwise preferences
        all_keys = list(returns.keys())
        agg = {k: 0.0 for k in all_keys}
        for (k1, k2), v in size_prefs.items():
            agg[k1] += v
            agg[k2] += v
        for (k1, k2), v in time_prefs.items():
            agg[k1] += v
            agg[k2] += v
        for (k1, k2), v in sim_prefs.items():
            agg[k1] += v
            agg[k2] += v
        return agg
