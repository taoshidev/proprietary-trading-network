from typing import Dict, Tuple
import numpy as np
import pandas as pd
from typing import Callable, Sequence
from vali_objects.vali_config import ValiConfig
from vali_objects.utils.functional_utils import FunctionalUtils
import bittensor as bt


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
    def correlation_matrix(returns: dict[str, list[float]]) -> tuple[pd.DataFrame, dict[str, float]]:
        """
        Calculate correlation matrix and mean pairwise correlations for miners.
        Filters out miners with insufficient return data and removes all-zero miners.
        :param returns: Dict mapping miner hotkeys to their daily returns
        :param min_returns_threshold: Minimum number of returns required per miner
        :return: Tuple of (correlation_matrix, mean_pairwise_correlations)
        """
        # Safety check: need at least 2 miners for correlation
        if len(returns) < 2:
            bt.logging.debug(f"Orthogonality: Insufficient miners ({len(returns)}) for correlation calculation")
            return pd.DataFrame(), {}
            
        # Convert to DataFrame
        df = pd.DataFrame(returns)

        # Filter miners with enough returns
        miners_with_enough_returns = df.columns[df.count() >= ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N]
        filtered_df = df.loc[:, miners_with_enough_returns]
        
        # Fill NaN with zero and remove miners with all-zero returns
        filtered_df = filtered_df.fillna(0)
        df_nonzero = filtered_df.loc[:, (filtered_df != 0).any(axis=0)]
        
        if df_nonzero.empty or len(df_nonzero.columns) < 2:
            bt.logging.debug(f"Orthogonality: Insufficient valid data after filtering - miners: {len(df_nonzero.columns)}")
            return pd.DataFrame(), {}
            
        # Calculate correlation matrix
        try:
            corr_matrix = df_nonzero.corr()
            
            # Check for invalid correlations
            if corr_matrix.isna().all().all():
                bt.logging.warning("Orthogonality: All correlations are NaN, returning empty matrix")
                return pd.DataFrame(), {}
                
        except Exception as e:
            bt.logging.error(f"Orthogonality: Error calculating correlation matrix: {e}")
            return pd.DataFrame(), {}
        
        # Set diagonal to NaN to ignore self-correlation
        np.fill_diagonal(corr_matrix.values, np.nan)
        
        # Calculate mean pairwise correlation per miner
        mean_corr_per_miner = corr_matrix.mean(skipna=True).to_dict()
        
        return corr_matrix, mean_corr_per_miner

    @staticmethod
    def sliding_similarity_array(
            v1: Sequence[float] | np.ndarray,
            v2: Sequence[float] | np.ndarray,
            window: int = 10
    ) -> np.ndarray:
        """
        Cosine similarity for every right‑hand shift of v1 (0 … max_shift‑1).
        Works for both Python lists and NumPy arrays.

        Returns
        -------
        np.ndarray
            An array of length `max_shift` whose i‑th element is the cosine
            similarity between v1 shifted right by i and the corresponding
            prefix of v2.
        """
        a = np.asarray(v1, dtype=float)
        b = np.asarray(v2, dtype=float)

        # First, go through the process of stripping down the arrays from zeros on the left. We want to keep the same indexing.
        zero_index = max(np.nonzero(a)[0][0], np.nonzero(b)[0][0])
        if zero_index > 0:
            a = a[zero_index:]
            b = b[zero_index:]

        assert len(a) == len(b), "Vectors must be of the same length after stripping zeros."

        if len(a) < window or len(b) < window:
            return np.array([0])

        sims = []
        for i in range(window, len(a)):
            at = a[i - window:i]
            bt = b[i - window:i]

            subset_similarity = Orthogonality.similarity(at, bt)
            sims.append(subset_similarity)

        return np.array(sims, dtype=float)

    @staticmethod
    def sliding_similarity_distillation(similarity_array: np.ndarray) -> float:
        """
        Distill the sliding similarity array into a single value.
        :param similarity_array: Array of cosine similarities.
        :return: Distilled similarity value.
        """
        if len(similarity_array) == 0:
            return 0.0
        return np.clip(np.max(similarity_array), a_min=0, a_max=1)

    @staticmethod
    def sliding_similarity(
            v1: Sequence[float] | np.ndarray,
            v2: Sequence[float] | np.ndarray,
            window: int = None
    ) -> float:
        """
        Calculate the sliding similarity between two vectors.
        :param v1: First vector.
        :param v2: Second vector.
        :param window: Maximum right-hand shift to consider.
        :return: Sliding similarity value.
        """
        if window is None:
            window = ValiConfig.TARGET_SIMILARITY_WINDOW_DAYS

        similarity_array = Orthogonality.sliding_similarity_array(
            v1,
            v2,
            window=window
        )

        return Orthogonality.sliding_similarity_distillation(similarity_array)
    
    @staticmethod
    def duration_metric(v: list[float]) -> float:
        """
        Determine the duration metric of a vector.
        :param v: Vector.
        :return: Duration metric of the vector.
        """
        if type(v) is not np.ndarray:
            v = np.array(v)

        nv = v[v != 0]

        return len(nv)

    @staticmethod
    def size_metric(v: list[float]) -> float:
        """
        Determine the size metric of a vector.
        :param v: Vector.
        :return: Size metric of the vector.
        """
        if type(v) is not np.ndarray:
            v = np.array(v)

        nv = v[v != 0]

        return np.sum(np.abs(nv))

    @staticmethod
    def time_preference(v1: list[float], v2: list[float], max_shift: int = 10) -> float:
        """
        Determine how preferred the first vector is over the second vector based on time longevity.
        In general, we want to prefer
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

        # Return the time preference between the two vectors, should always be normalized between -1 and 1
        return v1_preference - v2_preference
    
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
        v1_metric = Orthogonality.size_metric(v1)
        v2_metric = Orthogonality.size_metric(v2)

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

        return v1_preference - v2_preference

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
        return Orthogonality.pairwise_pref(returns, Orthogonality.sliding_similarity)

    @staticmethod
    def diverging_criteria(preference_factor: float, similarity_factor: float) -> float:
        """
        Diverging criteria based on preference and similarity factors.
        :param preference_factor: Preference factor (e.g., time or size).
        :param similarity_factor: Similarity factor (e.g., from sliding similarity).
        :return: Augmented Similarity Metric.
        """
        B: float = ValiConfig.ORTHOGONALITY_DIVERGING_INTENSITY

        # Input into the divergence function
        x = preference_factor

        # These will set the vertical asymptotes of the divergence function, really just need from 0 to 1
        m = -1
        n = 1

        # Safety checks for edge cases
        if not np.isfinite(preference_factor) or not np.isfinite(similarity_factor):
            return 0.0
            
        if preference_factor == 0:
            return 0.0

        if preference_factor == 1 or preference_factor == -1:
            return 1.0

        # Check bounds to prevent division by zero
        if x <= m or x >= n:
            return 0.0

        try:
            # Divergence factor should be quite high when the preference factor is close to 0 or 1
            divergence_factor = B * np.log((n-m)**2 / (4*(x-m)*(n-x)))
            return np.clip(divergence_factor, 0, 1)
        except (ValueError, ZeroDivisionError, OverflowError) as e:
            bt.logging.warning(f"Orthogonality: Error in diverging_criteria with pref={preference_factor}, sim={similarity_factor}: {e}")
            return 0.0

    @staticmethod
    def full_pref(returns: dict[str, list[float]]) -> dict[str, float]:
        """
        For a dict of miners' daily returns, return a dict mapping hotkey to the sum of all its pairwise compositional preferences (size, time, similarity) with all other miners.
        """
        # Safety checks
        if len(returns) < 2:
            bt.logging.debug(f"Orthogonality: Less than 2 miners ({len(returns)}), returning zero penalties")
            return {miner: 0.0 for miner in returns}
            
        # Check for empty or invalid returns
        valid_miners = []
        for miner, miner_returns in returns.items():
            if miner_returns and len(miner_returns) > 0 and any(r != 0 for r in miner_returns):
                valid_miners.append(miner)
                
        if len(valid_miners) < 2:
            bt.logging.debug(f"Orthogonality: Less than 2 valid miners ({len(valid_miners)}), returning zero penalties")
            return {miner: 0.0 for miner in returns}
        
        try:
            # Pairwise preference dictionaries
            time_prefs = Orthogonality.time_pref(returns)         # {(k1,k2): value}
            sim_prefs = Orthogonality.sim_pref(returns)          # {(k1,k2): score}

            penalty_pairs: Dict[Tuple[str, str], float] = {}

            # For now we can just use the time prefs, but we'll want to fold in size later
            for (k1, k2), v_time in time_prefs.items():
                sim_score = sim_prefs.get((k1, k2), 0.0)
                enhanced = Orthogonality.diverging_criteria(v_time, sim_score)

                # Positive value ⇒ "k1 over k2", negative/zero ⇒ "k2 over k1"
                if v_time > 0:
                    penalty_pairs[(k2, k1)] = enhanced     # k2 owes k1
                else:
                    penalty_pairs[(k1, k2)] = enhanced     # k1 owes k2

            final_penalty: Dict[str, float] = {miner: 0.0 for miner in returns}

            for (debtor, _), value in penalty_pairs.items():
                final_penalty[debtor] = max(final_penalty[debtor], value)
                
            bt.logging.debug(f"Orthogonality penalties calculated for {len(final_penalty)} miners")
            return final_penalty
            
        except Exception as e:
            bt.logging.error(f"Orthogonality: Error calculating penalties: {e}")
            return {miner: 0.0 for miner in returns}
