"""
Module for Abstract Scrubber description.
"""

__author__ = "Romanyuk Artem"
__copyright__ = "Copyright (c) 2024 Romanyuk Artem"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence

import numpy

from pysatl_cpd.core.scrubber_scenario import ScrubberScenario


class Scrubber(ABC):
    """A scrubber for dividing data into windows
    and subsequent processing of data windows
    by change point detection algorithms
    """

    def __init__(self) -> None:
        """A scrubber for dividing data into windows
        and subsequent processing of data windows
        by change point detection algorithms

        """
        self._scenario: ScrubberScenario | None = None
        self._data: Sequence[float | numpy.float64] = []
        self.is_running = True
        self.change_points: list[int] = []

    @abstractmethod
    def restart(self) -> None:
        """Function for restarting Scrubber"""
        raise NotImplementedError

    @abstractmethod
    def get_windows(self) -> Iterable[Sequence[float | numpy.float64]]:
        """Function for dividing data into parts to feed into the change point detection algorithm

        :return: Iterator of data windows for change point detection algorithm
        """
        raise NotImplementedError

    @abstractmethod
    def add_change_points(self, window_change_points: list[int]) -> None:
        """Function for mapping window change points to scrubber data part
        :param window_change_points: change points detected on scrubber window
        """
        raise NotImplementedError

    @property
    def scenario(self) -> ScrubberScenario:
        return self._scenario

    @scenario.setter
    def scenario(self, new_scenario) -> None:
        self._scenario = new_scenario

    @property
    def data(self) -> Sequence[float | numpy.float64]:
        return self._data

    @data.setter
    def data(self, new_data) -> None:
        self._data = new_data
        self.restart()
