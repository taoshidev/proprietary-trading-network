import numpy as np


class FunctionalUtils:
    @staticmethod
    def sigmoid(x: float, shift: float = 0, spread: float = 1) -> float:
        """
        Args:
            x: Value to be transformed
            shift: displacement of the sigmoid function
            spread: spread of the sigmoid function

        Returns:
            float - the sigmoid value of the input
        """
        if spread == 0:
            raise ValueError("The spread parameter must be different from 0")

        exp_term = np.clip(spread * (x - shift), -100, 100)
        return float(np.clip(1 / (1 + np.exp(exp_term)), 0, 1))