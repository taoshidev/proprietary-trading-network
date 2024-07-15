# developer: trdougherty
# Copyright © 2024 Taoshi Inc
import numpy as np
from numpy import ndarray
from scipy.sparse import csr_matrix
from scipy import sparse

from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

from vali_config import ValiConfig
from vali_objects.position import Position
from vali_objects.utils.position_utils import PositionUtils

class PlagiarismUtils:
    @staticmethod
    def generate_elimination_mapping(
        positions: list[Position],
        current_time: int
    ) -> dict[str, bool]:
        """
        Args:
            positions: list[Position] - the positions
        """
        # 1. Compute the similarity score between the signals
        # 2. Eliminate miners based on the similarity score

        # Translate the positions list to a list of states
        flattened_positions = PositionUtils.flatten_positions(positions)
        positions_list_translated = PositionUtils.translate_current_leverage(flattened_positions)
        miners, trade_pairs, state_list = PositionUtils.to_state_list(positions_list_translated, current_time=current_time)

        # Build the state matrix for similarity matching
        state_matrix = PlagiarismUtils.build_state_matrix(
            miners=miners,
            trade_pairs=trade_pairs,
            state_list=state_list,
            current_time=current_time
        )

        # intention here is to normalize the state matrix, hedge against different leverage scales
        state_matrix_normalized = PlagiarismUtils.normalize_state_matrix(state_matrix)

        # Compute similarities between the states with multiple time lags
        state_similarities_snapshot = PlagiarismUtils.build_similarities_cascade_lag(state_matrix_normalized)

        # Detect the time lagged similarities
        lag_detection = PlagiarismUtils.detect_lag(state_similarities_snapshot)
        similarities_detection = PlagiarismUtils.detect_similarity(state_similarities_snapshot)

        # Combine the two detections
        state_similarities_booleans = np.logical_and(lag_detection, similarities_detection)

        # log the similarities
        similarities_instances = PlagiarismUtils.log_similarities(
            miners,
            trade_pairs,
            state_similarities_booleans,
            state_similarities_snapshot
        )

        # Distill the similarities
        # state_similarities = PlagiarismUtils.similarities_distillation(state_similarities_snapshot)

        return similarities_instances
    
    @staticmethod
    def log_similarities(
        miners: list[str],
        trade_pairs: list[str],
        state_similarities_booleans: csr_matrix,
        state_similarities_snapshot: csr_matrix
    ):
        """
        Args:
            state_similarities: ndarray - the state similarities matrix
        """
        similarity_condition = state_similarities_booleans == True  # noqa: E712
        victims, plagiarisers, condition = sparse.find(similarity_condition)

        victims_cosine_similarities = state_similarities_snapshot[
            victims, 
            plagiarisers
        ]

        plagiarisers_minerids = np.array([ miners[plagiariser // len(trade_pairs)] for plagiariser in plagiarisers ])
        plagiarisers_tradepairids = np.array([ trade_pairs[plagiariser % len(trade_pairs)] for plagiariser in plagiarisers ])

        victims_minerids = np.array([ miners[victim // len(trade_pairs)] for victim in victims ])
        victims_tradepairids = np.array([ trade_pairs[victim % len(trade_pairs)] for victim in victims ])

        matching_tradepairs = (plagiarisers_tradepairids == victims_tradepairids) & (plagiarisers_minerids != victims_minerids)

        points_of_contention = list(zip(
            plagiarisers_minerids[matching_tradepairs],
            victims_minerids[matching_tradepairs],
            plagiarisers_tradepairids[matching_tradepairs],
            victims_cosine_similarities[matching_tradepairs]
        ))

        return points_of_contention

    @staticmethod
    def detect_lag(
        state_similarities: csr_matrix,
        threshold: float = None
    ) -> np.ndarray:
        """
        Args:
            state_similarities: ndarray - the state similarities matrix
            threshold: float - the threshold for similarity
        """
        # Axis of 1 is the new miners axis
        if threshold is None:
            threshold = ValiConfig.PLAGIARISM_FOLLOWER_TIMELAG_THRESHOLD

        sparse_comparison = np.array(state_similarities / state_similarities.T)
        return sparse_comparison >= threshold
    
    @staticmethod
    def detect_similarity(
        state_similarities: csr_matrix,
        threshold: float = None
    ) -> np.ndarray:
        """
        Args:
            state_similarities: ndarray - the state similarities matrix
            threshold: float - the threshold for similarity
        """
        # Axis of 1 is the new miners axis
        if threshold is None:
            threshold = ValiConfig.PLAGIARISM_FOLLOWER_SIMILARITY_THRESHOLD

        return (state_similarities >= threshold).toarray()
    
    @staticmethod
    def build_state_matrix(
        miners: list[str],
        trade_pairs: list[str],
        state_list: list[dict],
        current_time: int,
        time_resolution: int = None,
        lookback_window: int = None
    ) -> ndarray:
        """
        Args:
            scores: ndarray - the scores for each miner
        """
        if time_resolution is None:
            time_resolution = ValiConfig.PLAGIARISM_MATCHING_TIME_RESOLUTION_MS

        if lookback_window is None:
            lookback_window = ValiConfig.SET_WEIGHT_LOOKBACK_RANGE_MS

        end_time = current_time
        start_time = current_time - lookback_window

        times = np.arange(start_time, end_time, time_resolution)
        leverage_matrix = np.zeros((len(miners), len(trade_pairs), len(times)))

        for state in state_list:
            miner_index = miners.index(state["miner_id"])
            tradepair_index = trade_pairs.index(state["trade_pair"])
            time_criteria = (times >= state["start"]) & (times <= state["end"])
            leverage_matrix[miner_index, tradepair_index, time_criteria] = state["leverage"]

        return leverage_matrix
    
    @staticmethod
    def build_state_matrix_sparse(
        miners: list[str],
        trade_pairs: list[str],
        state_list: list[dict],
        current_time: int,
        time_resolution: int = None
    ) -> ndarray:
        """
        Args:
            scores: ndarray - the scores for each miner
        """
        if time_resolution is None:
            time_resolution = ValiConfig.PLAGIARISM_MATCHING_TIME_RESOLUTION_MS


        start_time = current_time - (ValiConfig.PLAGIARISM_LOOKBACK_RANGE_MS)
        times_length = (current_time - start_time) // time_resolution

        data = []
        row_indices = []
        col_indices = []

        for state in state_list:
            miner_index = miners.index(state["miner_id"])
            tradepair_index = trade_pairs.index(state["trade_pair"])
            
            start_index = ((state["start"] - start_time + time_resolution - 1) // time_resolution)
            end_index = ((state["end"] - start_time) // time_resolution) + 1
            
            time_indices = np.arange(start_index, end_index)
            
            data.extend([state["leverage"]] * len(time_indices))
            row_indices.extend([miner_index * len(trade_pairs) + tradepair_index] * len(time_indices))
            col_indices.extend(time_indices)

        leverage_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(len(miners) * len(trade_pairs), times_length))
        return leverage_matrix
    
    
    @staticmethod
    def similarities_distillation(
        state_similarities_matrix: ndarray
    ) -> ndarray:
        """
        Args:
            state_similarities_matrix: ndarray - the state similarities matrix - shape (n_miners, n_miners)
            return: ndarray - the distilled similarities matrix - shape (n_miners)
        """

        # axis 1 is the time lagged index
        return np.mean(state_similarities_matrix, axis=1)
    
    @staticmethod
    def build_similarities_matrix(
        state_matrix: csr_matrix,
        n_lags: int = 1
    ) -> csr_matrix:
        """
        Args:
            state_matrix: ndarray - the state matrix
        """
        # Compute the similarities between the states
        d1 = state_matrix[:,:-n_lags] # beginning to all the way up to current - n_lags
        d2 = state_matrix[:,n_lags:] # all the way up to current - i.e. "lagged" in history relative to d1

        # return a cosine similarity matrix between the two
        similarities = cosine_similarity(
            d1,
            d2,
            dense_output=False
        )

        # Drop the diagonal
        similarities.setdiag(0)
        return similarities
    
    @staticmethod
    def matrix_divided_by_transpose(matrix):
        transpose = matrix.T
        ratio_matrix = matrix.copy()
        
        # Find the indices of non-zero elements in ratio_matrix
        nonzero_indices = ratio_matrix.nonzero()
        
        # Get the corresponding non-zero elements from the transpose
        transpose_nonzero = transpose[nonzero_indices].A1
        
        # Replace zero elements in transpose_nonzero with 1 to avoid divide by zero
        transpose_nonzero[transpose_nonzero == 0] = 1
        
        # Divide the non-zero elements of ratio_matrix by the corresponding elements of transpose_nonzero
        ratio_matrix.data /= transpose_nonzero
        
        return ratio_matrix

    @staticmethod
    def find_nonsymmetric_elements_sparse(ratio_matrix, threshold=2):
        nonsymmetric_elements = []
        for i, j in zip(*ratio_matrix.nonzero()):
            if ratio_matrix[i, j] > threshold:
                nonsymmetric_elements.append((i, j))
        return nonsymmetric_elements
    
    @staticmethod
    def build_cosine_similarity_matrix(
        state_matrix: csr_matrix,
        n_lags: int = 1
    ) -> csr_matrix:
        """
        Args:
            state_matrix: ndarray - the state matrix
        """
        # Compute the similarities between the states
        similarity = np.dot(state_matrix[:, :-n_lags], state_matrix[:, n_lags:].T)

        # Squared magnitude of preference vectors (number of occurrences)
        square_mag = similarity.diagonal()

        # Inverse squared magnitude
        inv_square_mag = np.zeros_like(square_mag)
        non_zero_mask = square_mag != 0
        inv_square_mag[non_zero_mask] = 1 / square_mag[non_zero_mask]

        # Inverse of the magnitude
        inv_mag = np.sqrt(inv_square_mag)

        # Cosine similarity (elementwise multiply by inverse magnitudes)
        cosine = similarity * inv_mag
        # cosine = cosine.T * inv_mag

        return cosine

    @staticmethod
    def compress_similarity_matrix(similarity_matrix, n_miners, n_signals):
        compressed_data = []
        compressed_row = []
        compressed_col = []

        for i in range(n_miners):
            for j in range(n_miners):
                # Extract the cell block for miners i and j
                cell_block = similarity_matrix[i * n_signals:(i + 1) * n_signals, j * n_signals:(j + 1) * n_signals]
                
                # this is used when we just want to compare like signals
                cell_signals = cell_block.diagonal()

                # Find the maximum value in the cell block
                max_value = cell_signals.max()

                compressed_data.append(max_value)
                compressed_row.append(i)
                compressed_col.append(j)

        # Create the compressed similarity matrix in sparse format
        compressed_matrix = csr_matrix((compressed_data, (compressed_row, compressed_col)), shape=(n_miners, n_miners))

        return compressed_matrix
    
    @staticmethod
    def build_similarities_cascade_lag(state_matrix: csr_matrix, max_lags: int = None) -> csr_matrix:
        """
        Args:
            state_matrix: ndarray - the state matrix
        """
        if max_lags is None:
            max_lags = ValiConfig.PLAGIARISM_MAX_LAGS

        # Initialize the similarities matrix with the first lag
        similarities_matrix = PlagiarismUtils.build_similarities_matrix(state_matrix, n_lags=1)

        # Compute the maximum similarity for each subsequent lag
        for i in range(2, max_lags + 1):
            similarities_matrix_lagged = PlagiarismUtils.build_similarities_matrix(state_matrix, n_lags=i)
            similarities_matrix = similarities_matrix.maximum(similarities_matrix_lagged)

        return similarities_matrix
    
    @staticmethod
    def normalize_state_matrix(
        state_matrix: ndarray
    ) -> ndarray:
        """
        Args:
            state_matrix: ndarray - the state matrix
        """

        # Normalize along the timeseries axis - the rows, not the columns
        return normalize(state_matrix, norm='l2', axis=1)
    
    @staticmethod
    def similarity_compression(
        state_similarities_matrix: ndarray,
        threshold: float = 0.5
    ) -> ndarray:
        """
        Args:
            state_similarities_matrix: ndarray - the state similarities matrix
            threshold: float - the threshold for similarity
        """
        # Axis of 1 is the new miners axis
        similarities_percentile = np.percentile(state_similarities_matrix, 80, axis=1)  # noqa: F841
        return state_similarities_matrix
    
    @staticmethod
    def similarity_threshold(
        score: ndarray,
        threshold: float = None
    ) -> bool:
        """
        Args:
            position: score - current plagiarism score for each miner
            threshold: float - the threshold for plagiarism
        """
        if threshold is None:
            threshold = ValiConfig.PLAGIARISM_THRESHOLD

        return score > threshold
