import tempfile
from os import walk
from pathlib import Path

import pytest

from pysatl_cpd.generator.generator import ScipyDatasetGenerator
from pysatl_cpd.generator.saver import DatasetSaver


class TestGenerator:
    config_path = "tests/test_configs/test_config_1.yml"

    @pytest.mark.parametrize(
        "config_path_str,generator,configurations",
        (
            (
                config_path,
                ScipyDatasetGenerator(),
                {
                    "20-normal-0-1-20-normal-10-1": [40, [20]],
                    "20-multivariate_normal-0-0-20-multivariate_normal-10-10": [40, [20]],
                    "20-normal-0-1-no-change-point": [20, []],
                    "20-exponential-1-no-change-point": [20, []],
                    "20-weibull-1-1-no-change-point": [20, []],
                    "20-uniform-0-1-no-change-point": [20, []],
                    "20-beta-1-1-no-change-point": [20, []],
                    "20-gamma-1-1-no-change-point": [20, []],
                    "20-t-2-no-change-point": [20, []],
                    "20-lognorm-1-no-change-point": [20, []],
                    "20-multivariate_normal-0-1-no-change-point": [20, []],
                    "100-normal-0-1-no-change-point": [100, []],
                },
            ),
        ),
    )
    def test_generate_datasets(self, config_path_str, generator, configurations) -> None:
        generated = generator.generate_datasets(Path(config_path_str))
        for name in configurations:
            data_length = len(generated[name][0])
            assert data_length == configurations[name][0]
            assert generated[name][1] == configurations[name][1]

    @pytest.mark.parametrize(
        "config_path_str,generator,configurations",
        (
            (
                config_path,
                ScipyDatasetGenerator(),
                {
                    "20-normal-0-1-20-normal-10-1": [40, [20]],
                    "20-normal-0-1-no-change-point": [20, []],
                    "100-normal-0-1-no-change-point": [100, []],
                },
            ),
        ),
    )
    def test_generate_datasets_save(self, config_path_str, generator, configurations) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            saver = DatasetSaver(Path(tempdir), True)
            generated = generator.generate_datasets(Path(config_path_str), saver)
            for name in configurations:
                data_length = sum(1 for _ in generated[name][0])
                assert data_length == configurations[name][0]
                assert generated[name][1] == configurations[name][1]

            directory = [file_names for (_, _, file_names) in walk(tempdir)]
            for file_names in directory[1:]:
                assert sorted(file_names) == sorted(["changepoints.csv", "sample.adoc", "sample.png", "sample.csv"])
