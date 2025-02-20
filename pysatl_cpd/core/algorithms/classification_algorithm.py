"""
Module for implementation of CPD algorithm based on classification.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2024 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Iterable

import numpy as np

from pysatl_cpd.core.algorithms.abstract_algorithm import Algorithm
from pysatl_cpd.core.algorithms.classification.abstracts.iclassifier import Classifier
from pysatl_cpd.core.algorithms.classification.abstracts.iquality_metric import QualityMetric
from pysatl_cpd.core.algorithms.classification.abstracts.istatistic_test import TestStatistic


class ClassificationAlgorithm(Algorithm):
    """
    The class implementing change point detection algorithm based on classification.
    """

    def __init__(
        self, classifier: Classifier, quality_metric: QualityMetric, test_statistic: TestStatistic, indent_coeff: float
    ) -> None:
        """
        Initializes a new instance of classification based change point detection algorithm.

        :param classifier: Classifier for sample classification.
        :param quality_metric: Metric to assess independence of the two samples
        resulting from splitting the original sample.
        :param test_statistic: Criterion to separate change points from other points in sample.
        :param indent_coeff: Coefficient for evaluating indent from window borders.
        The indentation is calculated by multiplying the given coefficient by the size of window.
        """
        self.__classifier = classifier
        self.__test_statistic = test_statistic
        self.__quality_metric = quality_metric

        self.__shift_coeff = indent_coeff

        self.__change_points: list[int] = []
        self.__change_points_count = 0

    @property
    def test_statistic(self) -> TestStatistic:
        return self.__test_statistic

    @test_statistic.setter
    def test_statistic(self, test_statistic) -> None:
        self.__test_statistic = test_statistic

    def detect(self, window: Iterable[float | np.float64]) -> int:
        """Finds change points in window.

        :param window: part of global data for finding change points.
        :return: the number of change points in the window.
        """
        self.__process_data(window)
        return self.__change_points_count

    def localize(self, window: Iterable[float | np.float64]) -> list[int]:
        """Finds coordinates of change points (localizes them) in window.

        :param window: part of global data for finding change points.
        :return: list of window change points.
        """
        self.__process_data(window)
        return self.__change_points.copy()

    def __process_data(self, window: Iterable[float | np.float64]) -> None:
        """
        Processes a window of data to detect/localize all change points depending on working mode.

        :param window: part of global data for change points analysis.
        """
        sample = list(window)
        sample_size = len(sample)
        if sample_size == 0:
            return

        # Examining each point.
        # Boundaries are always change points.
        first_point = int(sample_size * self.__shift_coeff)
        last_point = int(sample_size * (1 - self.__shift_coeff))
        assessments = []

        for time in range(first_point, last_point):
            train_sample, test_sample = ClassificationAlgorithm.__split_sample(sample)
            self.__classifier.train(train_sample, int(time / 2))
            classes = self.__classifier.predict(test_sample)

            quality = self.__quality_metric.assess_barrier(classes, int(time / 2))
            assessments.append(quality)

        change_points = self.__test_statistic.get_change_points(assessments)

        # Shifting change points coordinates according to their place in window.
        self.__change_points = list(map(lambda x: x + first_point, change_points))
        self.__change_points_count = len(change_points)

    # Splits the given sample into train and test samples.
    # Strategy: even elements goes to the train sample; uneven --- to the test sample
    # Soon classification algorithm will be more generalized: the split strategy will be one of the parameters.
    @staticmethod
    def __split_sample(
        sample: Iterable[float | np.float64],
    ) -> tuple[list[list[float | np.float64]], list[list[float | np.float64]]]:
        train_sample = [[x] for i, x in enumerate(sample) if i % 2 == 0]
        test_sample = [[x] for i, x in enumerate(sample) if i % 2 != 0]

        return train_sample, test_sample
