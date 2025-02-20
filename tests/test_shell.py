import tempfile
from os import walk
from pathlib import Path

import pytest

from pysatl_cpd.core.algorithms.graph_algorithm import GraphAlgorithm
from pysatl_cpd.core.scrubber.abstract_scrubber import Scrubber
from pysatl_cpd.core.scrubber.linear_scrubber import LinearScrubber
from pysatl_cpd.core.scrubber_scenario import ScrubberScenario
from pysatl_cpd.shell import CPContainer, CPDProblem, CPDResultsAnalyzer, LabeledCPData


def custom_comparison(node1, node2):  # TODO: Remove it everywhere
    arg = 1
    return abs(node1 - node2) <= arg


class TestCPDShell:
    shell_for_setter_getter = CPDProblem([4, 3, 2, 1], cpd_algorithm=GraphAlgorithm(custom_comparison, 4))
    shell_normal = CPDProblem([1, 2, 3, 4], cpd_algorithm=GraphAlgorithm(custom_comparison, 4))
    shell_default = CPDProblem(
        [3, 4, 5, 6], ScrubberScenario(10, True), cpd_algorithm=GraphAlgorithm(custom_comparison, 4)
    )
    shell_marked_data = CPDProblem(
        LabeledCPData([1, 2, 3, 4], [4, 5, 6, 7]),
        cpd_algorithm=GraphAlgorithm(custom_comparison, 4),
    )

    def test_init(self) -> None:
        assert self.shell_normal._data == [1, 2, 3, 4]
        assert self.shell_normal.cpd_core.data_controller.data == [1, 2, 3, 4]
        assert isinstance(self.shell_normal.cpd_core.algorithm, GraphAlgorithm)

        assert isinstance(self.shell_default.cpd_core.algorithm, GraphAlgorithm)
        assert isinstance(self.shell_default.cpd_core.scrubber, LinearScrubber)

        assert isinstance(self.shell_marked_data._data, LabeledCPData)

        assert self.shell_marked_data._data.raw_data == [1, 2, 3, 4]
        assert self.shell_marked_data._data.change_points == [4, 5, 6, 7]
        assert list(self.shell_marked_data.cpd_core.data_controller.data.__iter__()) == [1, 2, 3, 4]

    def test_data_getter_setter(self) -> None:
        assert self.shell_for_setter_getter.data == [4, 3, 2, 1]
        assert self.shell_for_setter_getter.cpd_core.data_controller.data == [4, 3, 2, 1]

        self.shell_for_setter_getter.data = [1, 3, 4]

        assert self.shell_for_setter_getter.data == [1, 3, 4]
        assert self.shell_for_setter_getter.cpd_core.data_controller.data == [1, 3, 4]

    def test_scrubber_setter(self) -> None:
        class TestNewScrubber(Scrubber):
            def restart(self) -> None:
                pass

            def get_windows(self):
                pass

            def add_change_points(self, window_change_points: list[int]) -> None:
                pass

        previous_scrubber = self.shell_for_setter_getter.scrubber
        self.shell_for_setter_getter.scrubber = TestNewScrubber()
        assert isinstance(self.shell_for_setter_getter.scrubber, TestNewScrubber)
        assert self.shell_for_setter_getter.scrubber._data == previous_scrubber._data
        assert self.shell_for_setter_getter.scrubber.scenario == previous_scrubber.scenario

    def test_cpd_algorithm_getter_setter(self) -> None:
        FIVE = 5

        class TestNewAlgo(GraphAlgorithm):
            pass

        self.shell_for_setter_getter.cpd_algorithm = TestNewAlgo(custom_comparison, 5)
        assert isinstance(self.shell_for_setter_getter.cpd_core.algorithm, TestNewAlgo)
        assert self.shell_for_setter_getter.cpd_core.algorithm.threshold == FIVE

    def test_scenario_getter_setter(self) -> None:
        assert self.shell_for_setter_getter.scenario.max_window_cp_number == 10**9
        assert self.shell_for_setter_getter.scenario.to_localize
        self.shell_for_setter_getter.scenario = ScrubberScenario(20, False)
        assert self.shell_for_setter_getter.cpd_core.scenario == ScrubberScenario(20, False)

    def test_change_scenario(self) -> None:
        self.shell_for_setter_getter.change_scenario(15, True)
        assert self.shell_for_setter_getter.scenario == ScrubberScenario(15, True)

    def test_run_cpd(self) -> None:
        res_normal = self.shell_normal.run_cpd()
        res_def = self.shell_default.run_cpd()
        res_marked = self.shell_marked_data.run_cpd()
        assert res_normal.result == []
        assert res_normal.expected_result is None

        assert res_def.result == []
        assert res_def.expected_result is None

        assert res_marked.result == []
        assert res_marked.expected_result == [4, 5, 6, 7]
        assert res_marked.result_diff == [4, 5, 6, 7]


class TestCPDResultsAnalyzer:
    @pytest.mark.parametrize(
        "result1, result2, window, expected",
        [
            ([4, 5, 6, 7], [3, 5, 6], None, (2, 1, 1, 1)),
            ([4, 5, 6, 7], [3, 5, 6], (5, 6), (1, 0, 0, 0)),
            ([4, 5, 6, 7], [3, 5, 6], (0, 100), (2, 97, 2, 1)),
            ([4, 5, 6, 7], [3, 5, 6], (6, 6), (0, 0, 0, 0)),
            ([3, 5, 6, 7], [4, 5, 6], None, (2, 1, 1, 1)),
            ([], [4, 5, 6], None, (0, 0, 0, 2)),
            ([3, 5, 6, 7], [], None, (0, 4, 3, 0)),
        ],
    )
    def test_count_confusion_matrix(self, result1, result2, window, expected):
        assert CPDResultsAnalyzer.count_confusion_matrix(result1, result2, window) == expected

    def test_count_confusion_matrix_exception_case(self):
        with pytest.raises(ValueError):
            CPDResultsAnalyzer.count_confusion_matrix([], [])

    @pytest.mark.parametrize(
        "result1, result2, window, expected",
        [
            ([4, 5, 6, 7], [3, 5, 6], None, 0.6),
            ([4, 5, 6, 7], [3, 5, 6], (5, 6), 1.0),
            ([4, 5, 6, 7], [3, 5, 6], (6, 6), 0.0),
        ],
    )
    def test_count_accuracy(self, result1, result2, window, expected):
        assert CPDResultsAnalyzer.count_accuracy(result1, result2, window) == expected

    @pytest.mark.parametrize(
        "result1, result2, window, expected",
        [
            ([4, 5, 6, 7], [3, 5, 6], None, 2 / 3),
            ([4, 5, 6, 7], [3, 5, 6], (5, 6), 1.0),
            ([4, 5, 6, 7], [3, 5, 6], (6, 6), 0.0),
        ],
    )
    def test_count_precision(self, result1, result2, window, expected):
        assert CPDResultsAnalyzer.count_precision(result1, result2, window) == expected

    @pytest.mark.parametrize(
        "result1, result2, window, expected",
        [
            ([4, 5, 6, 7], [3, 5, 6], None, 2 / 3),
            ([4, 5, 6, 7], [3, 5, 6], (5, 6), 1.0),
            ([4, 5, 6, 7], [3, 5, 6], (6, 6), 0.0),
        ],
    )
    def test_count_recall(self, result1, result2, window, expected):
        assert CPDResultsAnalyzer.count_recall(result1, result2, window) == expected


class TestCPContainer:
    cont_default1 = CPContainer([1] * 15, [1, 2, 3], [2, 3, 4], 10)
    cont_default2 = CPContainer([1] * 15, [1, 2, 3, 6, 8], [2, 3, 4, 6], 20)
    cont_no_expected = CPContainer([1] * 15, [1, 2, 3], None, 5)

    def test_result_diff(self) -> None:
        assert self.cont_default1.result_diff == [1, 4]
        assert self.cont_default2.result_diff == [1, 4, 8]

    def test_result_diff_exeception_case(self) -> None:
        with pytest.raises(ValueError):
            self.cont_no_expected.result_diff

    def test_str_cpcontainer(self) -> None:
        assert (
            str(self.cont_default1)
            == """Located change points: (1;2;3)
Expected change point: (2;3;4)
Difference: (1;4)
Computation time (sec): 10"""
        )

        assert (
            str(self.cont_default2)
            == """Located change points: (1;2;3;6;8)
Expected change point: (2;3;4;6)
Difference: (1;4;8)
Computation time (sec): 20"""
        )

        assert (
            str(self.cont_no_expected)
            == """Located change points: (1;2;3)
Computation time (sec): 5"""
        )

    @pytest.mark.parametrize(
        "data,name",
        (
            (cont_default1, "d_1"),
            (cont_default2, "d_2"),
            (cont_no_expected, "cne"),
        ),
    )
    def test_visualize(self, data, name) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            data.visualize(False, Path(tempdir), name)
            assert [f"{name}.png"] in [file_names for (_, _, file_names) in walk(tempdir)]

    def test_mertic_exception_case(self):
        with pytest.raises(ValueError):
            self.cont_no_expected.count_confusion_matrix()
