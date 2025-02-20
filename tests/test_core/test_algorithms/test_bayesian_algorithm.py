import numpy as np
import pytest

from pysatl_cpd.core.algorithms.bayesian.detectors.simple import SimpleDetector
from pysatl_cpd.core.algorithms.bayesian.hazards.constant import ConstantHazard
from pysatl_cpd.core.algorithms.bayesian.likelihoods.gaussian_unknown_mean_and_variance import (
    GaussianUnknownMeanAndVariance,
)
from pysatl_cpd.core.algorithms.bayesian.localizers.simple import SimpleLocalizer
from pysatl_cpd.core.algorithms.bayesian_algorithm import BayesianAlgorithm


def set_seed():
    np.random.seed(1)


def construct_bayesian_algorithm():
    return BayesianAlgorithm(
        learning_steps=50,
        likelihood=GaussianUnknownMeanAndVariance(),
        hazard=ConstantHazard(rate=1.0 / (1.0 - 0.5 ** (1.0 / 500))),
        detector=SimpleDetector(threshold=0.04),
        localizer=SimpleLocalizer(),
    )


@pytest.fixture(scope="function")
def data_params():
    return {
        "num_of_tests": 10,
        "size": 500,
        "change_point": 250,
        "tolerable_deviation": 25,
    }


@pytest.fixture
def generate_data(data_params):
    def _generate_data():
        set_seed()
        return np.concatenate(
            [
                np.random.normal(loc=0, scale=1, size=data_params["change_point"]),
                np.random.normal(loc=5, scale=2, size=data_params["size"] - data_params["change_point"]),
            ]
        )

    return _generate_data


@pytest.fixture(scope="function")
def outer_bayesian_algorithm():
    return construct_bayesian_algorithm()


@pytest.fixture
def inner_algorithm_factory():
    def _factory():
        return construct_bayesian_algorithm()

    return _factory


def test_consecutive_detection(outer_bayesian_algorithm, generate_data, data_params):
    for _ in range(data_params["num_of_tests"]):
        data = generate_data()
        result = outer_bayesian_algorithm.detect(data)
        assert result, "There was undetected change point in data"


def test_correctness_of_consecutive_detection(
    outer_bayesian_algorithm, inner_algorithm_factory, generate_data, data_params
):
    for _ in range(data_params["num_of_tests"]):
        data = generate_data()
        inner_algorithm = inner_algorithm_factory()
        outer_result = outer_bayesian_algorithm.detect(data)
        inner_result = inner_algorithm.detect(data)
        assert outer_result == inner_result, "Consecutive and independent detection should give same results"


def test_consecutive_localization(outer_bayesian_algorithm, generate_data, data_params):
    for _ in range(data_params["num_of_tests"]):
        data = generate_data()
        result = outer_bayesian_algorithm.localize(data)
        assert (
            len(result) > 0
            and data_params["change_point"] - data_params["tolerable_deviation"]
            <= result[0]
            <= data_params["change_point"] + data_params["tolerable_deviation"]
        ), "Incorrect change point localization"


def test_correctness_of_consecutive_localization(
    outer_bayesian_algorithm, inner_algorithm_factory, generate_data, data_params
):
    for _ in range(data_params["num_of_tests"]):
        data = generate_data()
        inner_algorithm = inner_algorithm_factory()
        outer_result = outer_bayesian_algorithm.localize(data)
        inner_result = inner_algorithm.localize(data)
        assert outer_result == inner_result, "Consecutive and independent localization should give same results"


@pytest.mark.parametrize("hazard_rate,max_run_length", [(1.1, 50), (10, 100), (200, 250), (500.325251, 500)])
def test_constant_hazard_for_constants(hazard_rate, max_run_length):
    constant_hazard = ConstantHazard(hazard_rate)
    run_lengths = np.arange(max_run_length)
    hazard_probs = constant_hazard.hazard(run_lengths)
    assert hazard_probs.shape[0] == max_run_length, (
        f"Expected {max_run_length} probabilities, got {hazard_probs.shape[0]}"
    )
    assert np.all(hazard_probs == 1 / hazard_rate), f"Hazard probabilities must be {1 / hazard_rate}"


def test_learning_and_update():
    num_of_tests = 10
    size = 500
    change_point = 250
    learning_steps = 50
    time_after_learning = 51
    threshold_after_learning = 0.01
    time_before_change_point = 249
    threshold_change_point = 0.05

    set_seed()

    for _ in range(num_of_tests):
        data = np.concatenate(
            [
                np.random.normal(loc=0, scale=1, size=change_point),
                np.random.normal(loc=5, scale=2, size=size - change_point),
            ]
        )
        likelihood = GaussianUnknownMeanAndVariance()
        likelihood.learn(list(data[:learning_steps]))

        for time in range(learning_steps, size):
            observation = float(data[time])
            pred_probs = np.array(likelihood.predict(observation))

            if time == time_after_learning:
                assert pred_probs[-1] > threshold_after_learning, (
                    f"Too small predictive probability after learning: {pred_probs[-1]}"
                )
            if time == time_before_change_point:
                assert pred_probs[-1] >= threshold_change_point, (
                    f"Too small predictive probability before change: {pred_probs[-1]}"
                )
            if time == change_point:
                assert pred_probs[-1] < threshold_change_point, (
                    f"Too large predictive probability after change: {pred_probs[-1]}"
                )

            expected_size = time - learning_steps + 1
            assert pred_probs.shape[0] == expected_size, (
                f"Expected {expected_size} probabilities, got {pred_probs.shape[0]}"
            )

            likelihood.update(observation)


def test_clear():
    set_seed()

    size = 51
    data = np.random.normal(loc=0, scale=1, size=size)
    likelihood = GaussianUnknownMeanAndVariance()

    likelihood.learn(list(data[:-2]))
    first = likelihood.predict(float(data[-1]))

    likelihood.clear()

    likelihood.learn(list(data[:-2]))
    second = likelihood.predict(float(data[-1]))

    assert first == second, f"Results differ after clear: {first} vs {second}"


def test_detector_detection():
    run_lengths = np.full(100, 0.01)
    detector = SimpleDetector(threshold=0.04)
    assert detector.detect(run_lengths), "Change point should be detected"


def test_detector_clear():
    run_lengths = np.full(100, 0.01)
    detector = SimpleDetector(threshold=0.04)

    result1 = detector.detect(run_lengths)
    detector.clear()
    result2 = detector.detect(run_lengths)

    assert result1 == result2, f"Results differ after clear: {result1} vs {result2}"


def test_localizer_localization():
    change_point = 5
    run_lengths = np.full(11, 0.05)
    run_lengths[change_point] = 0.5
    localizer = SimpleLocalizer()
    result = localizer.localize(run_lengths)
    assert result == change_point, f"Expected change at {change_point}, got {result}"
