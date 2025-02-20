"""
Module for implementation of classifier's quality metric based on Matthews correlation coefficient.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2024 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

from math import sqrt

from numpy import ndarray

from pysatl_cpd.core.algorithms.classification.abstracts.iquality_metric import QualityMetric


class MCC(QualityMetric):
    """
    The class implementing quality metric based on Matthews correlation coefficient.
    """

    def assess_barrier(self, classes: ndarray, time: int) -> float:
        """Evaluates quality function based on classificator in the specified point.

        :param classes: Classes of observations, predicted by the classifier.
        :param time: Index of barrier in the given sample to calculate quality.
        :return: Quality assessment.
        """
        before = classes[:time]
        after = classes[time:]
        after_length = len(after)
        before_length = time

        true_positive = after.sum()
        false_positive = before.sum()
        true_negative = before_length - false_positive
        false_negative = after_length - true_positive
        positive = true_positive + false_negative
        negative = false_positive + true_negative
        pp = true_positive + false_positive
        pn = false_negative + true_negative

        tpr = true_positive / positive
        tnr = true_negative / negative
        ppv = true_positive / pp
        npv = true_negative / pn
        fnr = false_negative / positive
        fpr = false_positive / negative
        fo_rate = false_negative / pn
        fdr = false_positive / pp

        return sqrt(tpr * tnr * ppv * npv) - sqrt(fnr * fpr * fo_rate * fdr)
