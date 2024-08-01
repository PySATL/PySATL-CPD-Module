from collections.abc import Iterable

import numpy as np

from CPDShell.Core.algorithms.DensityBasedCPD.abstracts.density_based_algorithm import DensityBasedAlgorithm


class RulsifAlgorithm(DensityBasedAlgorithm):
    """Relative Unconstrained Least-Squares Importance Fitting (RULSIF)
    algorithm for change point detection.

    RULSIF estimates the density ratio between two distributions and uses
    the importance weights for detecting changes in the data distribution.
    """

    def __init__(self, bandwidth: float, regularization_coef: float, threshold: float = 1.1):
        """Initialize the RULSIF algorithm.

        Args:
            bandwidth (float): bandwidth parameter for density estimation.
            regularization_coef (float): regularization parameter.
            threshold (float, optional): threshold for detecting change points.
            Defaults to 1.1.
        """
        self.bandwidth = bandwidth
        self.regularization_coef = regularization_coef
        self.threshold = threshold

    def _loss_function(self, density_ratio: np.ndarray, alpha: np.ndarray) -> float:
        """Loss function for RULSIF.

        Args:
            density_ratio (np.ndarray): estimated density ratio.
            alpha (np.ndarray): coefficients for the density ratio.

        Returns:
            float: the computed loss value.
        """
        return np.mean((density_ratio - 1) ** 2) + self.regularization_coef * np.sum(alpha**2)

    def detect(self, window: Iterable[float | np.float64]) -> int:
        """Detect the number of change points in the given data window
        using RULSIF.

        Args:
            window (Iterable[float]): the data window to detect change points.

        Returns:
            int: the number of detected change points.
        """
        window_sample = np.array(window)
        weights = self._calculate_weights(
            self=self,
            test_value=window_sample,
            reference_value=window_sample,
            bandwidth=self.bandwidth,
            objective_function=self._loss_function,
        )

        return np.count_nonzero(weights > self.threshold)

    def localize(self, window: Iterable[float | np.float64]) -> list[int]:
        """Localize the change points in the given data window using RULSIF.

        Args:
            window (Iterable[float]): the data window to localize change points.

        Returns:
            List[int]: the indices of the detected change points.
        """
        window_sample = np.array(window)
        weights = self._calculate_weights(
            self=self,
            test_value=window_sample,
            reference_value=window_sample,
            bandwidth=self.bandwidth,
            objective_function=self._loss_function,
        )

        return np.where(weights > self.threshold)[0].tolist()