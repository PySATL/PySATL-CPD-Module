"""
Module for Classification CPD algorithm's test statistic abstract base class.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2024 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod


class TestStatistic(ABC):
    """Test statistic's abstract base class."""

    @abstractmethod
    def get_change_points(self, classifier_assessments: list[float]) -> list[int]:
        """Separates change points from other points in sample based on some criterion.

        :param classifier_assessments: List of quality assessments evaluated in each point of the sample.
        :return: Change points in the current window.
        """
        raise NotImplementedError
