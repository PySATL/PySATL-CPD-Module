"""
Module for implementation of decision tree classifier for cpd.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2024 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import sklearn.tree as sk

from pysatl_cpd.core.algorithms.classification.abstracts.iclassifier import Classifier


class DecisionTreeClassifier(Classifier):
    """
    The class implementing decision tree classifier for cpd.
    """

    def __init__(self) -> None:
        """
        Initializes a new instance of decision tree classifier for cpd.
        """
        self.__model: sk.DecisionTreeClassifier | None = None

    def train(self, sample: list[list[float | np.float64]], barrier: int) -> None:
        """Trains classifier on the given sample.

        :param sample: sample for training classifier.
        :param barrier: index of observation that splits the given sample.
        """
        classes = [0 if i <= barrier else 1 for i in range(len(sample))]
        self.__model = sk.DecisionTreeClassifier()
        self.__model.fit(sample, classes)

    def predict(self, sample: list[list[float | np.float64]]) -> np.ndarray:
        """Classifies observations in the given sample based on training with barrier.

        :param sample: sample to classify.
        """
        return self.__model.predict(sample)
