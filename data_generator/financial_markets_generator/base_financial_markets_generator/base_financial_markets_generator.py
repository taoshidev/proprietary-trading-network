from abc import ABC, abstractmethod
from typing import List


class BaseFinancialMarketsGenerator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_data(self, *args):
        pass

    @abstractmethod
    def get_data_and_structure_data_points(self, *args):
        pass

    @staticmethod
    def convert_output_to_data_points(data_structure: List[List], days_data: List[List], order_to_ds: List[int]):
        """
        return close time, close, high, low, volume
        """
        for tf_row in days_data:
            data_structure[0].append(int(tf_row[order_to_ds[0]]))
            data_structure[1].append(float(tf_row[order_to_ds[1]]))
            data_structure[2].append(float(tf_row[order_to_ds[2]]))
            data_structure[3].append(float(tf_row[order_to_ds[3]]))
            data_structure[4].append(float(tf_row[order_to_ds[4]]))