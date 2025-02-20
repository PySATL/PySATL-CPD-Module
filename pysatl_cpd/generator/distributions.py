import ast
from enum import Enum
from typing import Final, Protocol

import numpy as np
import scipy.stats as ss


class Distributions(Enum):
    NORMAL = "normal"
    EXPONENTIAL = "exponential"
    WEIBULL = "weibull"
    UNIFORM = "uniform"
    BETA = "beta"
    GAMMA = "gamma"
    T = "t"
    LOGNORM = "lognorm"
    MULTIVARIATIVE_NORMAL = "multivariate_normal"

    def __str__(self):
        return self.value


class Distribution(Protocol):
    """
    An interface for all distributions.
    Allows to create instance with name and params dictionary.
    """

    @property
    def name(self) -> str:
        """
        :return: Name of the distribution.
        """
        return ""

    @property
    def params(self) -> dict[str, str]:
        """
        :return: Parameters of the distribution.
        """
        return {}

    @staticmethod
    def from_str(name: str, params: dict[str, str]) -> "Distribution":
        distributions = {
            Distributions.NORMAL.value: NormalDistribution,
            Distributions.EXPONENTIAL.value: ExponentialDistribution,
            Distributions.WEIBULL.value: WeibullDistribution,
            Distributions.UNIFORM.value: UniformDistribution,
            Distributions.BETA.value: BetaDistribution,
            Distributions.GAMMA.value: GammaDistribution,
            Distributions.T.value: TDistribution,
            Distributions.LOGNORM.value: LogNormDistribution,
            Distributions.MULTIVARIATIVE_NORMAL.value: MultivariateNormalDistribution,
        }
        return distributions[name].from_params(params)


class ScipyDistribution(Distribution):
    """
    Distribution supporting sample generation with SciPy methods.
    """

    def scipy_sample(self, length: int) -> np.ndarray:
        """
        Generate sample using SciPy.

        :return: Generated sample.
        """
        raise NotImplementedError()


class NormalDistribution(ScipyDistribution):
    """
    Description for the normal distribution with mean and variance parameters.
    """

    MEAN_KEY: Final[str] = "mean"
    VAR_KEY: Final[str] = "variance"

    mean: float
    variance: float

    def __init__(self, mean=0.0, var=1.0):
        self.mean = mean
        self.variance = var

    @property
    def name(self) -> str:
        return str(Distributions.NORMAL)

    @property
    def params(self) -> dict[str, str]:
        return {
            NormalDistribution.MEAN_KEY: str(self.mean),
            NormalDistribution.VAR_KEY: str(self.variance),
        }

    def scipy_sample(self, length: int) -> np.ndarray:
        return ss.norm(loc=self.mean, scale=self.variance).rvs(size=length)

    @staticmethod
    def from_params(params: dict[str, str]) -> "NormalDistribution":
        parameter_number = 2
        if len(params) != parameter_number:
            raise ValueError(
                "Normal distribution must have 2 parameters: "
                + f"{NormalDistribution.MEAN_KEY}, {NormalDistribution.VAR_KEY}"
            )
        mean: float = float(params[NormalDistribution.MEAN_KEY])
        var: float = float(params[NormalDistribution.VAR_KEY])
        if var < 0:
            raise ValueError("Variance cannot be less than 0")
        return NormalDistribution(mean, var)


class ExponentialDistribution(ScipyDistribution):
    """
    Description of exponential distribution with intensity parameter.
    """

    RATE_KEY: Final[str] = "rate"

    rate: float

    def __init__(self, rate: float = 1.0):
        if rate <= 0:
            raise ValueError("Rate must be greater than 0")
        self.rate = rate

    @property
    def name(self) -> str:
        return str(Distributions.EXPONENTIAL)

    @property
    def params(self) -> dict[str, str]:
        return {
            ExponentialDistribution.RATE_KEY: str(self.rate),
        }

    def scipy_sample(self, length: int) -> np.ndarray:
        return ss.expon(scale=1 / self.rate).rvs(size=length)

    @staticmethod
    def from_params(params: dict[str, str]) -> "ExponentialDistribution":
        if len(params) != 1:
            raise ValueError("Exponential distribution must have 1 parameter: " + f"{ExponentialDistribution.RATE_KEY}")
        rate: float = float(params[ExponentialDistribution.RATE_KEY])
        if rate <= 0:
            raise ValueError("Rate must be greater than 0")
        return ExponentialDistribution(rate)


class WeibullDistribution(ScipyDistribution):
    """
    Description of weibull distribution with intensity parameter.
    """

    SHAPE_KEY: Final[str] = "shape"
    SCALE_KEY: Final[str] = "scale"

    shape: float
    scale: float

    def __init__(self, shape: float = 1.0, scale: float = 1.0):
        if shape <= 0 or scale <= 0:
            raise ValueError("Shape and scale must be greater than 0")
        self.shape = shape
        self.scale = scale

    @property
    def name(self) -> str:
        return str(Distributions.WEIBULL)

    @property
    def params(self) -> dict[str, str]:
        return {
            WeibullDistribution.SHAPE_KEY: str(self.shape),
            WeibullDistribution.SCALE_KEY: str(self.scale),
        }

    def scipy_sample(self, length: int) -> np.ndarray:
        return ss.weibull_min(c=self.shape, scale=1 / self.scale).rvs(size=length)

    @staticmethod
    def from_params(params: dict[str, str]) -> "WeibullDistribution":
        num_params = 2
        if len(params) != num_params:
            raise ValueError(
                "Exponential distribution must have 2 parameters: "
                + f"{WeibullDistribution.SHAPE_KEY}"
                + f"{WeibullDistribution.SCALE_KEY}"
            )
        shape: float = float(params[WeibullDistribution.SHAPE_KEY])
        scale: float = float(params[WeibullDistribution.SCALE_KEY])
        if shape <= 0 or scale <= 0:
            raise ValueError("Parameters must be greater than 0")
        return WeibullDistribution(shape, scale)


class UniformDistribution(ScipyDistribution):
    """
    Description of uniform distribution with intensity parameter.
    """

    MIN_KEY: Final[str] = "min"
    MAX_KEY: Final[str] = "max"

    max: float
    min: float

    def __init__(self, min_value: float, max_value: float):
        if min_value >= max_value:
            raise ValueError("Max must be greater than min value")
        self.min = min_value
        self.max = max_value

    @property
    def name(self) -> str:
        return str(Distributions.UNIFORM)

    @property
    def params(self) -> dict[str, str]:
        return {
            UniformDistribution.MIN_KEY: str(self.min),
            UniformDistribution.MAX_KEY: str(self.max),
        }

    def scipy_sample(self, length: int) -> np.ndarray:
        return ss.uniform(loc=self.min, scale=self.max - self.min).rvs(size=length)

    @staticmethod
    def from_params(params: dict[str, str]) -> "UniformDistribution":
        num_params = 2
        if len(params) != num_params:
            raise ValueError(
                "Uniform distribution must have 2 parameters: "
                + f"{UniformDistribution.MIN_KEY}"
                + f"{UniformDistribution.MAX_KEY}"
            )
        min_value: float = float(params[UniformDistribution.MIN_KEY])
        max_value: float = float(params[UniformDistribution.MAX_KEY])
        if min_value >= max_value:
            raise ValueError("Max must be greater than min value")
        return UniformDistribution(min_value, max_value)


class BetaDistribution(ScipyDistribution):
    """
    Description of beta distribution with intensity parameter.
    """

    ALPHA_KEY: Final[str] = "alpha"
    BETA_KEY: Final[str] = "beta"

    alpha: float
    beta: float

    def __init__(self, alpha_value: float, beta_value: float):
        if alpha_value <= 0 or beta_value <= 0:
            raise ValueError("Alpha and beta must be greater than zero")
        self.alpha = alpha_value
        self.beta = beta_value

    @property
    def name(self) -> str:
        return str(Distributions.BETA)

    @property
    def params(self) -> dict[str, str]:
        return {
            BetaDistribution.ALPHA_KEY: str(self.alpha),
            BetaDistribution.BETA_KEY: str(self.beta),
        }

    def scipy_sample(self, length: int) -> np.ndarray:
        return ss.beta(a=self.alpha, b=self.beta).rvs(size=length)

    @staticmethod
    def from_params(params: dict[str, str]) -> "BetaDistribution":
        num_params = 2
        if len(params) != num_params:
            raise ValueError(
                f"Beta distribution must have 2 parameters: {BetaDistribution.ALPHA_KEY}, {BetaDistribution.BETA_KEY}"
            )
        alpha: float = float(params[BetaDistribution.ALPHA_KEY])
        beta: float = float(params[BetaDistribution.BETA_KEY])
        if alpha <= 0 or beta <= 0:
            raise ValueError("Alpha and beta must be greater than zero")
        return BetaDistribution(alpha, beta)


class GammaDistribution(ScipyDistribution):
    """
    Description of gamma distribution with shape and scale parameters.
    """

    ALPHA_KEY: Final[str] = "alpha"
    BETA_KEY: Final[str] = "beta"

    alpha: float
    beta: float

    def __init__(self, alpha_value: float, beta_value: float):
        if alpha_value <= 0 or beta_value <= 0:
            raise ValueError("Alpha and beta must be greater than zero")
        self.alpha = alpha_value
        self.beta = beta_value

    @property
    def name(self) -> str:
        return str(Distributions.GAMMA)

    @property
    def params(self) -> dict[str, str]:
        return {
            GammaDistribution.ALPHA_KEY: str(self.alpha),
            GammaDistribution.BETA_KEY: str(self.beta),
        }

    def scipy_sample(self, length: int) -> np.ndarray:
        return ss.gamma(a=self.alpha, scale=1 / self.beta).rvs(size=length)

    @staticmethod
    def from_params(params: dict[str, str]) -> "GammaDistribution":
        num_params = 2
        if len(params) != num_params:
            raise ValueError(
                f"Gamma  must have 2 parameters: {GammaDistribution.ALPHA_KEY}, {GammaDistribution.BETA_KEY}"
            )
        alpha = float(params[GammaDistribution.ALPHA_KEY])
        beta = float(params[GammaDistribution.BETA_KEY])
        if alpha <= 0 or beta <= 0:
            raise ValueError("Alpha and beta for gamma distributions must be greater than zero")
        return GammaDistribution(alpha, beta)


class TDistribution(ScipyDistribution):
    """
    Description of Student's t-distribution with the degrees of freedom parameter.
    """

    N_KEY: Final[str] = "n"

    n: int

    def __init__(self, n_value: int) -> None:
        if n_value <= 0:
            raise ValueError("Degrees of freedom must be positive integer number")
        self.n = n_value

    @property
    def name(self) -> str:
        return str(Distributions.T)

    @property
    def params(self) -> dict[str, str]:
        return {
            TDistribution.N_KEY: str(self.n),
        }

    def scipy_sample(self, length: int) -> np.ndarray:
        return ss.t(df=self.n).rvs(size=length)

    @staticmethod
    def from_params(params: dict[str, str]) -> "TDistribution":
        num_params = 1
        if len(params) != num_params:
            raise ValueError(f"Student's distribution must have 1 parameter: {TDistribution.N_KEY}")
        n = int(params[TDistribution.N_KEY])
        if n <= 0:
            raise ValueError("n (degrees of freedom) must be positive integer")
        return TDistribution(n)


class LogNormDistribution(ScipyDistribution):
    """
    Description of log normal distributionn with one parameter
    """

    S_KEY: Final[str] = "s"

    s: float

    def __init__(self, s_value: float) -> None:
        if s_value <= 0:
            raise ValueError("S parameter must be positive number")
        self.s = s_value

    @property
    def name(self) -> str:
        return str(Distributions.LOGNORM)

    @property
    def params(self) -> dict[str, str]:
        return {
            LogNormDistribution.S_KEY: str(self.s),
        }

    def scipy_sample(self, length: int) -> np.ndarray:
        return ss.lognorm(s=self.s).rvs(size=length)

    @staticmethod
    def from_params(params: dict[str, str]) -> "LogNormDistribution":
        num_params = 1
        if len(params) != num_params:
            raise ValueError(f"Log normal distribution must have 1 parameter: {LogNormDistribution.S_KEY}")
        s = float(params[LogNormDistribution.S_KEY])
        return LogNormDistribution(s)


class MultivariateNormalDistribution(ScipyDistribution):
    """
    Description of multivariate normal distribution with det(covarianse matrix) == 1
    """

    MEAN_KEY: Final[str] = "mean"

    mean: list[float]
    cov: np.ndarray

    def __init__(self, mean_value: list[float]) -> None:
        if len(mean_value) == 0:
            raise ValueError("Mean cannot have dimention less than 1")
        self.mean = mean_value
        self.cov = np.diag(v=[1 for _ in range(len(mean_value))])

    @property
    def name(self) -> str:
        return str(Distributions.MULTIVARIATIVE_NORMAL)

    @property
    def params(self) -> dict[str, str]:
        return {MultivariateNormalDistribution.MEAN_KEY: str(self.mean)}

    def scipy_sample(self, length: int) -> np.ndarray:
        return ss.multivariate_normal(mean=self.mean, cov=self.cov).rvs(size=length)

    @staticmethod
    def from_params(params: dict[str, str]) -> "MultivariateNormalDistribution":
        num_params = 1
        if len(params) != num_params:
            raise ValueError(
                f"Multivariative normal distribution must have 1 parameter: {MultivariateNormalDistribution.MEAN_KEY}"
            )
        mean = list(map(float, ast.literal_eval(params[MultivariateNormalDistribution.MEAN_KEY])))
        return MultivariateNormalDistribution(mean)
