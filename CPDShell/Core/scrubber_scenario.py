from dataclasses import dataclass


@dataclass
class ScrubberScenario:
    """Scenario for scrubber

    :param max_window_cp_number: maximum number of change points on the scrubber window, defaults to 1
    :param to_localize: is it necessary to localize change points, defaults to False
    """

    max_window_cp_number: int = 10**9
    to_localize: bool = True
