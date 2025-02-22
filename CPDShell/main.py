from pathlib import Path

from CPDShell.Core.algorithms.bayesian_algorithm import BayesianAlgorithm
from CPDShell.Core.algorithms.BayesianCPD.detectors.drop_detector import DropDetector
from CPDShell.Core.algorithms.BayesianCPD.detectors.simple_detector import SimpleDetector
from CPDShell.Core.algorithms.BayesianCPD.hazards.constant_hazard import ConstantHazard
from CPDShell.Core.algorithms.BayesianCPD.likelihoods.gaussian_unknown_mean_and_variance import (
    GaussianUnknownMeanAndVariance,
)
from CPDShell.Core.algorithms.BayesianCPD.localizers.simple_localizer import SimpleLocalizer
from CPDShell.Core.algorithms.ClassificationBasedCPD.test_statistics.threshold_overcome import ThresholdOvercome
from CPDShell.Core.algorithms.knn_algorithm import KNNAlgorithm
from CPDShell.Core.scrubber.linear_scrubber import LinearScrubber
from CPDShell.generator.generator import ScipyDatasetGenerator
from CPDShell.generator.saver import DatasetSaver
from CPDShell.shell import CPDProblem

path_string = "tests/test_CPDShell/test_configs/test_config_exp.yml"
distributions_name = "exp"

saver = DatasetSaver(Path(), True)
generated = ScipyDatasetGenerator().generate_datasets(Path(path_string), saver)
data, expected_change_points = generated[distributions_name]

print("Expected change points:", expected_change_points)

# Graph algorithm demo
graph_cpd = CPDProblem(data)
graph_cpd.scrubber.window_length = 150
graph_cpd.scrubber.movement_k = 2.0 / 3.0

res_graph = graph_cpd.run_cpd()
res_graph.visualize(True)
print("Graph algorithm")
print(res_graph)


# k-NN based algorithm demo
def metric(obs1: float, obs2: float) -> float:
    return abs(obs1 - obs2)


K = 7
THRESHOLD = 2.5
WINDOW_SIZE = 48
MOVEMENT_COEFF = 0.5
INDENT_COEFF = 0.25
CHANGE_POINT_NUMBER = 40

statistic = ThresholdOvercome(THRESHOLD)
knn_algorithm = KNNAlgorithm(metric, statistic, INDENT_COEFF, K)
scrubber = LinearScrubber(WINDOW_SIZE, MOVEMENT_COEFF)
knn_cpd = CPDProblem(data, cpd_algorithm=knn_algorithm, scrubber=scrubber)

res_knn = knn_cpd.run_cpd()
res_knn.visualize(True)
print("k-NN based algorithm")
print(res_knn)


# Bayesian algorithm demo
BAYESIAN_THRESHOLD = 0.1
NUM_OF_SAMPLES = 1000
SAMPLE_SIZE = 500
BERNOULLI_PROB = 1.0 - 0.5 ** (1.0 / SAMPLE_SIZE)
HAZARD_RATE = 1 / BERNOULLI_PROB
LEARNING_SAMPLE_SIZE = 50
BAYESIAN_DROP_THRESHOLD = 0.7

constant_hazard = ConstantHazard(HAZARD_RATE)
gaussian_likelihood = GaussianUnknownMeanAndVariance()

simple_detector = SimpleDetector(BAYESIAN_THRESHOLD)
drop_detector = DropDetector(BAYESIAN_DROP_THRESHOLD)

simple_localizer = SimpleLocalizer()

bayesian_algorithm = BayesianAlgorithm(
    learning_steps=LEARNING_SAMPLE_SIZE,
    likelihood=gaussian_likelihood,
    hazard=constant_hazard,
    detector=simple_detector,
    localizer=simple_localizer,
)

bayesian_cpd = CPDProblem(data, cpd_algorithm=bayesian_algorithm)
bayesian_cpd.scrubber.window_length = 500
bayesian_cpd.scrubber.movement_k = 2.0 / 3.0

res_bayes = bayesian_cpd.run_cpd()
res_bayes.visualize(True)
print("Bayesian algorithm")
print(res_bayes)
