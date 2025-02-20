"""
Module for implementation of knn classifier for cpd.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2024 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

import typing as tp

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from pysatl_cpd.core.algorithms.classification.abstracts.iclassifier import Classifier


class KNNClassifier(Classifier):
    """
    The class implementing knn classifier for cpd.
    """

    def __init__(
        self, k: int, distance: tp.Literal["manhattan", "euclidean", "minkowski", "hamming"] = "minkowski"
    ) -> None:
        """
        Initializes a new instance of knn classifier for cpd.
        :param k: number of neighbours in the knn graph relative to each point.
        :param distance: Metric to use for distance computation.
        Default is "minkowski", which results in the standard Euclidean distance when p = 2.
        """
        self.__k = k
        self.__distance: tp.Literal["manhattan", "euclidean", "minkowski", "hamming"] = distance
        self.__model: KNeighborsClassifier | None = None

    def train(self, sample: list[list[float | np.float64]], barrier: int) -> None:
        """Trains classifier on the given sample.

        :param sample: sample for training classifier.
        :param barrier: index of observation that splits the given sample.
        """
        classes = [0 if i <= barrier else 1 for i in range(len(sample))]
        self.__model = KNeighborsClassifier(n_neighbors=self.__k, metric=self.__distance)
        self.__model.fit(sample, classes)

    def predict(self, sample: list[list[float | np.float64]]) -> np.ndarray:
        """Classifies observations in the given sample based on training with barrier.

        :param sample: sample to classify.
        """
        return self.__model.predict(sample)
