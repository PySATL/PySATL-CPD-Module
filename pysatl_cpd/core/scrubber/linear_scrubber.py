"""
Module for implementation of Linear Scrubber.
"""

__author__ = "Romanyuk Artem"
__copyright__ = "Copyright (c) 2024 Romanyuk Artem"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Iterable, Sequence

import numpy

from pysatl_cpd.core.scrubber.abstract_scrubber import Scrubber


class LinearScrubber(Scrubber):
    """A linear scrubber for dividing data into windows by moving them through data"""

    def __init__(
        self,
        window_length: int = 100,
        shift_factor: float = 1.0 / 3.0,
    ):
        """A linear scrubber for dividing data into windows by moving them through data

        :param window_length: length of data window
        :param shift_factor: how far will the window move relative to the length
        """
        super().__init__()
        self._window_length = window_length
        self._shift_factor = shift_factor
        self._window_start = 0
        self._rewrite_data_index: int = 0

    def restart(self) -> None:
        self.change_points = []
        self.is_running = True
        self._window_start = 0

    def get_windows(self) -> Iterable[Sequence[float | numpy.float64]]:
        while (self._data and self._window_start == 0) or (
            self._window_start + self._window_length <= len(self._data) and self.is_running
        ):
            window_end = self._window_start + self._window_length
            yield self._data[self._window_start : window_end]
            self._window_start += max(1, int(self._window_length * self._shift_factor))

    def add_change_points(self, window_change_points: list[int]) -> None:
        if self.scenario is None:
            raise ValueError("Scrubber has not ScrubberScenario")
        max_change_points = self.scenario.max_window_cp_number
        if self.scenario.to_localize:
            for point in window_change_points[:max_change_points]:
                if self._window_start + point not in self.change_points:
                    self.change_points.append(self._window_start + point)
        else:
            self.change_points += list(
                map(
                    lambda point: self._window_start + point,
                    (window_change_points[:max_change_points]),
                )
            )
