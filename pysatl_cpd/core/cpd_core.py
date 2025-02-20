from collections.abc import Sequence

import numpy

from .algorithms.graph_algorithm import Algorithm
from .data_controller import DataController
from .scrubber.abstract_scrubber import Scrubber
from .scrubber_scenario import ScrubberScenario


class CPDCore:
    """Change Point Detection core"""

    def __init__(
        self,
        scenario: ScrubberScenario,
        data: Sequence[float | numpy.float64],
        scrubber: Scrubber,
        algorithm: Algorithm,
        scrubber_data_size: int = 2000,
    ) -> None:
        """Change Point Detection core

        :param scenario: :class:`Scenario` object with information about the task
        :param data: values for change point detection
        :param scrubber: scrubber for dividing data into windows
            and subsequent processing of data windows
            by change point detection algorithms
        :param algorithm: change point detection algorithm
        :param scrubber_data_size: size of parts into which the initial data will be divided for the scrubber

        """
        self.data_controller = DataController(data, scrubber_data_size)
        self.scrubber = scrubber
        self.scenario = scenario
        self.algorithm = algorithm

    def run(self) -> list[int]:
        """Find change points

        :return: list of change points
        """
        self.scrubber.scenario = self.scenario
        self.scrubber.restart()
        self.data_controller.restart()
        for data in self.data_controller.get_data():
            self.scrubber.data = data
            for window in self.scrubber.get_windows():
                if self.scenario.to_localize:
                    window_change_points = self.algorithm.localize(window)
                else:
                    change_points_number = self.algorithm.detect(window)
                    window_change_points = [0] * change_points_number
                self.scrubber.add_change_points(window_change_points)
            self.data_controller.add_change_points(self.scrubber.change_points)
        return self.data_controller.change_points
