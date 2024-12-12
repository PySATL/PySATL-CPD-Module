"""
Module for Classification CPD algorithm's classifier abstract base class.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2024 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod

import numpy as np


class Classifier(ABC):
    """Classifier's abstract base class."""

    @abstractmethod
    def train(self, sample: list[list[float | np.float64]], barrier: int) -> None:
        """Trains binary classifier on the given sample.
        The observations before barrier belong to the class 0, after barrier --- to the class 1.

        :param sample: sample for training classifier.
        :param barrier: index of observation that splits the given sample.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, sample: list[list[float | np.float64]]) -> np.ndarray:
        """Classifies the elements of a sample into one of two classes, based on training with the barrier.

        :param sample: sample to classify.
        """
        raise NotImplementedError
