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
        time_preference_factor = ValiConfig.TIME_PREFERENCE_FACTOR
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
        size_preference_factor = ValiConfig.SIZE_PREFERENCE_FACTOR
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
    def multi_miner_preference(miner_returns: list[list[float]], metric_function: Callable) -> list[list[float]]:
        """
        For a list of miners’ daily returns, return an n×n NumPy array
        whose strictly upper‑triangular part (j > i) contains the pairwise
        time‑preference values; the diagonal and lower part are zero.
        :param miner_returns: List of miners’ daily returns.
        :return: n×n NumPy array whose strictly upper‑triangular part (j > i) contains the pairwise time‑preference values; the diagonal and lower part are zero.
        """
        n = len(miner_returns)
        prefs = [[0 for _ in range(n)] for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):           # strictly upper triangle
                prefs[i, j] = metric_function(
                    miner_returns[i], miner_returns[j]
                )

        return prefs
        
    @staticmethod
    def multi_miner_time_preference(miner_returns: list[list[float]]) -> list[list[float]]:
        """
        For a list of miners’ daily returns, return an n×n NumPy array
        whose strictly upper‑triangular part (j > i) contains the pairwise
        time‑preference values; the diagonal and lower part are zero.
        :param miner_returns: List of miners’ daily returns.
        :return: n×n NumPy array whose strictly upper‑triangular part (j > i) contains the pairwise time‑preference values; the diagonal and lower part are zero.
        """
        return Orthogonality.multi_miner_preference(miner_returns, Orthogonality.time_preference)
    
    @staticmethod
    def multi_miner_size_preference(miner_returns: list[list[float]]) -> list[list[float]]:
        """
        For a list of miners’ daily returns, return an n×n NumPy array
        whose strictly upper‑triangular part (j > i) contains the pairwise
        size‑preference values; the diagonal and lower part are zero.
        :param miner_returns: List of miners’ daily returns.
        :return: n×n NumPy array whose strictly upper‑triangular part (j > i) contains the pairwise size‑preference values; the diagonal and lower part are zero.
        """
        return Orthogonality.multi_miner_preference(miner_returns, Orthogonality.size_preference)
    
    @staticmethod
    def multi_miner_similarity(miner_returns: list[list[float]]) -> list[list[float]]:
        """
        For a list of miners’ daily returns, return an n×n NumPy array
        whose strictly upper‑triangular part (j > i) contains the pairwise
        similarity values; the diagonal and lower part are zero.
        :param miner_returns: List of miners’ daily returns.
        :return: n×n NumPy array whose strictly upper‑triangular part (j > i) contains the pairwise similarity values; the diagonal and lower part are zero.
        """
        return Orthogonality.multi_miner_preference(miner_returns, Orthogonality.convolutional_similarity)
    
    @staticmethod
    def multi_miner_full_compositional_preference(miner_returns: list[list[float]]) -> list[list[float]]:
        """
        For a list of miners’ daily returns, return an n×n NumPy array
        whose strictly upper‑triangular part (j > i) contains the pairwise
        full compositional preference values; the diagonal and lower part are zero.
        :param miner_returns: List of miners’ daily returns.
        :return: n×n NumPy array whose strictly upper‑triangular part (j > i) contains the pairwise full compositional preference values; the diagonal and lower part are zero.
        """
        miner_size_preferences = Orthogonality.multi_miner_size_preference(miner_returns)
        miner_time_preferences = Orthogonality.multi_miner_time_preference(miner_returns)
        miner_similarity = Orthogonality.multi_miner_similarity(miner_returns)

        return miner_size_preferences + miner_time_preferences + miner_similarity
