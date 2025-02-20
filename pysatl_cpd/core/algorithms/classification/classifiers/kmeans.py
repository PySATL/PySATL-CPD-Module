"""
Module for implementation of k-means classifier for cpd.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2024 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Iterable

import numpy as np
from sklearn.cluster import KMeans

from pysatl_cpd.core.algorithms.classification.abstracts.iclassifier import Classifier


class KMeansAlgorithm(Classifier):
    """
    The class implementing k-means classifier for cpd.
    """

    def __init__(self) -> None:
        """
        Initializes a new instance of k-means classifier for cpd.
        """
        self.__model: KMeans | None = None
        self.__sample: list[float | np.float64] | None = None

    def train(self, sample: Iterable[float | np.float64], barrier: int) -> None:
        """Trains classifier on the given sample.

        :param sample: sample for training classifier.
        :param barrier: index of observation that splits the given sample.
        """
        sample_list = list(sample)
        if self.__sample == sample_list:
            return

        k_means = KMeans(n_clusters=2)
        self.__sample = sample_list
        window_reshaped = np.array(sample_list).reshape(-1, 1)
        self.__model = k_means.fit(window_reshaped)

    def predict(self, sample: list[float | np.float64]) -> np.ndarray:
        """Classifies observations in the given sample based on training with barrier.

        :param sample: sample to classify.
        """
        return self.__model.labels_
