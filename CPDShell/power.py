from pathlib import Path

from CPDShell.shell import CPContainer


class Power:
    def __init__(self, root_path: Path):
        self.root_path = root_path

    def __read_all_csv(self):
        """Iterator for reading all CSV files in the specified directory and its subdirectories."""
        for csv_file in self.root_path.rglob("*.csv"):
            data = CPContainer.from_csv(csv_file)
            yield data

    @staticmethod
    def __checking_near_point(container: CPContainer):
        """Checks whether at least one change point is about the expected ones."""
        margin = 50
        detected_points = container.result
        expected_points = container.expected_result

        if not expected_points:
            return False

        for detected in detected_points:
            if any(expected - margin <= detected <= expected + margin for expected in expected_points):
                return True
        return False

    def calculate_power(self):
        """Calculates the power of the algorithm based on hits about the disorder."""
        false_negatives = 0
        total_containers = sum(1 for _ in self.__read_all_csv())
        for container in self.__read_all_csv():
            if not self.__checking_near_point(container):
                false_negatives += 1
        probability_type_two_error = false_negatives / total_containers if total_containers > 0 else 0
        return 1 - probability_type_two_error
