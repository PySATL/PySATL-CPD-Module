"""
Module for implementation of Data Controller.
"""

__author__ = "Romanyuk Artem"
__copyright__ = "Copyright (c) 2024 Romanyuk Artem"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Iterable

import numpy
from numpy.random.mtrand import Sequence


class DataController:
    """Data Controller for dividing big data to smaller windows for Scrubber"""

    def __init__(self, data: Sequence[float | numpy.float64], scrubber_data_size: int = 2000) -> None:
        """Data Controller for dividing big data to smaller windows for Scrubber

        :param data: values for change point detection
        :param scrubber_data_size: size of data needed to the scrubber
        """

        self._data = data
        self._data_start_index: int = 0
        self.change_points = []
        self._scrubber_data_size = scrubber_data_size

    def restart(self) -> None:
        """Function for restarting Data Controller"""

        self.change_points = []
        self._data_start_index = 0

    def get_data(self) -> Iterable[Sequence[float | numpy.float64]]:
        """Function for getting data pieces

        :return: data pieces iterator"""
        while self._data_start_index < len(self._data):
            cur_index = self._data_start_index
            yield self._data[cur_index : self._data_start_index + self._scrubber_data_size]
            self._data_start_index += self._scrubber_data_size

    def add_change_points(self, window_change_points: list[int]) -> None:
        """Function for mapping window change points to global data

        :param window_change_points: list of change points detected by scrubber"""

        self.change_points += list(map(lambda point: point + self._data_start_index, window_change_points))

    @property
    def data(self) -> Sequence[float | numpy.float64]:
        return self._data

    @data.setter
    def data(self, new_data) -> None:
        self._data = new_data
        self.restart()
