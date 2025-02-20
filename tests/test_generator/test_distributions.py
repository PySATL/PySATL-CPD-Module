import pytest

import pysatl_cpd.generator.distributions as dstr


class TestDistributions:
    @pytest.mark.parametrize(
        "distribution, params, error",
        [
            (dstr.Distributions.NORMAL, {"mean": "0"}, ValueError),
            (dstr.Distributions.NORMAL, {"mean": "0", "var": "1"}, KeyError),
            (dstr.Distributions.NORMAL, {"mean": "0", "variance": "1", "x": "5"}, ValueError),
            (dstr.Distributions.NORMAL, {"mean": "0", "variance": "-1"}, ValueError),
            (dstr.Distributions.EXPONENTIAL, {}, ValueError),
            (dstr.Distributions.EXPONENTIAL, {"rt": "1"}, KeyError),
            (dstr.Distributions.EXPONENTIAL, {"rate": "1", "x": "5"}, ValueError),
            (dstr.Distributions.EXPONENTIAL, {"rate": "-1"}, ValueError),
            (dstr.Distributions.WEIBULL, {"shape": "0"}, ValueError),
            (dstr.Distributions.WEIBULL, {"shape": "0", "var": "1"}, KeyError),
            (dstr.Distributions.WEIBULL, {"shape": "1", "scale": "1", "x": "5"}, ValueError),
            (dstr.Distributions.WEIBULL, {"shape": "-1", "scale": "1"}, ValueError),
            (dstr.Distributions.WEIBULL, {"shape": "1", "scale": "-1"}, ValueError),
            (dstr.Distributions.UNIFORM, {"min": "0"}, ValueError),
            (dstr.Distributions.UNIFORM, {"min": "-1", "MAX": "1"}, KeyError),
            (dstr.Distributions.UNIFORM, {"min": "-1", "max": "1", "x": "5"}, ValueError),
            (dstr.Distributions.UNIFORM, {"min": "1", "max": "-1"}, ValueError),
            (dstr.Distributions.BETA, {"alpha": "1"}, ValueError),
            (dstr.Distributions.BETA, {"alpha": "1", "x": "1"}, KeyError),
            (dstr.Distributions.BETA, {"alpha": "1", "beta": "1", "x": "5"}, ValueError),
            (dstr.Distributions.BETA, {"alpha": "-1", "beta": "1"}, ValueError),
            (dstr.Distributions.BETA, {"alpha": "1", "beta": "-1"}, ValueError),
            (dstr.Distributions.GAMMA, {"alpha": "1"}, ValueError),
            (dstr.Distributions.GAMMA, {"alpha": "1", "x": "1"}, KeyError),
            (dstr.Distributions.GAMMA, {"alpha": "1", "beta": "1", "x": "5"}, ValueError),
            (dstr.Distributions.GAMMA, {"alpha": "-1", "beta": "1"}, ValueError),
            (dstr.Distributions.GAMMA, {"alpha": "1", "beta": "-1"}, ValueError),
            (dstr.Distributions.T, {}, ValueError),
            (dstr.Distributions.T, {"N": "1"}, KeyError),
            (dstr.Distributions.T, {"n": "1", "x": "5"}, ValueError),
            (dstr.Distributions.T, {"n": "-1"}, ValueError),
            (dstr.Distributions.LOGNORM, {}, ValueError),
            (dstr.Distributions.LOGNORM, {"S": "1"}, KeyError),
            (dstr.Distributions.LOGNORM, {"s": "1", "x": "5"}, ValueError),
            (dstr.Distributions.LOGNORM, {"s": "-1"}, ValueError),
            (dstr.Distributions.MULTIVARIATIVE_NORMAL, {}, ValueError),
            (dstr.Distributions.MULTIVARIATIVE_NORMAL, {"Mean": "[0.0, 0.0]"}, KeyError),
            (dstr.Distributions.MULTIVARIATIVE_NORMAL, {"mean": "[0.0, 0.0]", "x": "5"}, ValueError),
            (dstr.Distributions.MULTIVARIATIVE_NORMAL, {"mean": "[]"}, ValueError),
        ],
    )
    def test_distribution_params_validation_fail(self, distribution, params, error):
        sample_len = 100
        with pytest.raises(error):
            d = dstr.Distribution.from_str(str(distribution), params)
            assert isinstance(d, dstr.ScipyDistribution)
            assert len(d.scipy_sample(sample_len)) == sample_len

    @pytest.mark.parametrize(
        "distribution, params",
        [
            (dstr.Distributions.NORMAL, {"mean": "0", "variance": "1"}),
            (dstr.Distributions.EXPONENTIAL, {"rate": "1"}),
            (dstr.Distributions.WEIBULL, {"shape": "1", "scale": "1"}),
            (dstr.Distributions.UNIFORM, {"min": "0", "max": "1"}),
            (dstr.Distributions.BETA, {"alpha": "1", "beta": "1"}),
            (dstr.Distributions.GAMMA, {"alpha": "1", "beta": "1"}),
            (dstr.Distributions.T, {"n": "1"}),
            (dstr.Distributions.LOGNORM, {"s": "1"}),
            (dstr.Distributions.MULTIVARIATIVE_NORMAL, {"mean": "[0.0, 0.1]"}),
        ],
    )
    def test_distribution_generate(self, distribution, params):
        sample_len = 100
        d = dstr.Distribution.from_str(str(distribution), params)
        assert isinstance(d, dstr.ScipyDistribution)
        assert len(d.scipy_sample(sample_len)) == sample_len
