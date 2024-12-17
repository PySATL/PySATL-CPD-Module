import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from CPDShell.power import Power


def create_mock_csv(file_path, data, result, expected_result, time_sec):
    """Helper function to create a mock CSV file."""
    with open(file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["data", "result", "expected_result", "time_sec"])
        writer.writerow(
            [
                ",".join(map(str, data)),
                ",".join(map(str, result)),
                ",".join(map(str, expected_result)) if expected_result else "",
                time_sec,
            ]
        )


class TestPower:
    @pytest.mark.parametrize(
        "mock_files, expected_power, test_case_description",
        [
            (
                [
                    {"data": [1.0, 2.0, 3.0], "result": [10], "expected_result": [10], "time_sec": 0.1},
                    {"data": [1.0, 2.0, 3.0], "result": [100], "expected_result": [10], "time_sec": 0.1},
                ],
                0.5,
                "One correct detection, one false negative",
            ),
            (
                [
                    {"data": [1.0, 2.0, 3.0], "result": [10], "expected_result": [10], "time_sec": 0.1},
                    {"data": [1.0, 2.0, 3.0], "result": [20], "expected_result": [20], "time_sec": 0.1},
                ],
                1.0,
                "All correct detections",
            ),
            (
                [
                    {"data": [1.0, 2.0, 3.0], "result": [70], "expected_result": [0], "time_sec": 0.1},
                    {"data": [1.0, 2.0, 3.0], "result": [80], "expected_result": [0], "time_sec": 0.1},
                ],
                0.0,
                "All false negatives",
            ),
            (
                [
                    {"data": [1.0, 2.0, 3.0], "result": [1050], "expected_result": [1000], "time_sec": 0.1},
                    {"data": [1.0, 2.0, 3.0], "result": [950], "expected_result": [1000], "time_sec": 0.1},
                ],
                1.0,
                "Detections within margin range (50 units)",
            ),
            (
                [{"data": [1.0, 2.0, 3.0], "result": [10], "expected_result": [], "time_sec": 0.1}],
                0.0,
                "No expected change points provided",
            ),
        ],
    )
    def test_calculate_power(self, mock_files, expected_power, test_case_description):
        """Parameterized test for Power.calculate_power with various scenarios."""
        with TemporaryDirectory() as temp_dir:
            root_path = Path(temp_dir)

            # Создание mock-файлов CSV
            for i, mock_data in enumerate(mock_files):
                create_mock_csv(
                    root_path / f"test{i + 1}.csv",
                    data=mock_data["data"],
                    result=mock_data["result"],
                    expected_result=mock_data["expected_result"],
                    time_sec=mock_data["time_sec"],
                )

            # Инициализация Power и расчет мощности
            power_calc = Power(root_path)
            try:
                power = power_calc.calculate_power()
                assert power == pytest.approx(expected_power, 0.01), (
                    f"Test case failed: {test_case_description}. " f"Expected power: {expected_power}, but got: {power}"
                )
            except TypeError as e:
                pytest.fail(f"TypeError in test '{test_case_description}': {str(e)!s}")
