# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


class DistributionType(Enum):
    int = 1
    float = 2
    categorical = 3


class Direction(Enum):
    minimize = 1
    maximize = 2


@dataclass
class SamplerConfig:
    _target_: str = MISSING


@dataclass
class GridSamplerConfig(SamplerConfig):
    """
    https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.GridSampler.html
    """

    _target_: str = "optuna.samplers.GridSampler"
    # search_space will be populated at run time based on hydra.sweeper.params
    _partial_: bool = True


@dataclass
class TPESamplerConfig(SamplerConfig):
    """
    https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.TPESampler.html
    """

    _target_: str = "optuna.samplers.TPESampler"
    seed: Optional[int] = None

    consider_prior: bool = True
    prior_weight: float = 1.0
    consider_magic_clip: bool = True
    consider_endpoints: bool = False
    n_startup_trials: int = 10
    n_ei_candidates: int = 24
    multivariate: bool = False
    warn_independent_sampling: bool = True


@dataclass
class RandomSamplerConfig(SamplerConfig):
    """
    https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.RandomSampler.html
    """

    _target_: str = "optuna.samplers.RandomSampler"
    seed: Optional[int] = None


@dataclass
class BOSamplerConfig(SamplerConfig):
    _target_: str = "bcm.samplers.bayesian_optimization.BOSampler"
    seed: Optional[int] = None
    # Steps of random exploration
    init_points: Optional[int] = 5


@dataclass
class GpyOptSamplerConfig(SamplerConfig):
    _target_: str = "bcm.samplers.gpyopt.BOSampler"
    seed: Optional[int] = None
    # type of model to use as surrogate:
    # - ‘GP’, standard Gaussian process.
    # - ‘GP_MCMC’, Gaussian process with prior in the hyper-parameters.
    # - ‘sparseGP’, sparse Gaussian process.
    # - ‘warperdGP’, warped Gaussian process.
    # - ‘InputWarpedGP’, input warped Gaussian process
    # - ‘RF’, random forest (scikit-learn).
    model_type: Optional[str] = "GP"
    # number of initial points that are collected jointly before start running the optimization.
    initial_design_numdata: Optional[int] = 5
    # type of initial design:
    # - ‘random’, to collect points in random locations.
    # - ‘latin’, to collect points in a Latin hypercube (discrete variables are sampled randomly.)
    initial_design_type: Optional[str] = "random"
    # type of acquisition function to use.
    # - ‘EI’, expected improvement.
    # - ‘EI_MCMC’, integrated expected improvement (requires GP_MCMC model).
    # - ‘MPI’, maximum probability of improvement.
    # - ‘MPI_MCMC’, maximum probability of improvement (requires GP_MCMC model).
    # - ‘LCB’, GP-Lower confidence bound.
    # - ‘LCB_MCMC’, integrated GP-Lower confidence bound (requires GP_MCMC model).
    acquisition_type: Optional[str] = "EI"
    # type of acquisition function to use.
    # - ‘lbfgs’: L-BFGS.
    # - ‘DIRECT’: Dividing Rectangles.
    # - ‘CMA’: covariance matrix adaptation.
    acquisition_optimizer_type: Optional[str] = "lbfgs"


@dataclass
class PyMooSamplerConfig(SamplerConfig):
    _target_: str = "bcm.samplers.pymoo.PyMooSampler"
    seed: Optional[int] = None
    algorithm_name: Optional[str] = "de"
    sampling_name: Optional[str] = "lhs"
    pop_size: Optional[int] = 2


@dataclass
class BoTorchSamplerConfig(SamplerConfig):
    _target_: str = "bcm.samplers.botorch.BoTorchSampler"
    seed: Optional[int] = None


@dataclass
class CmaEsSamplerConfig(SamplerConfig):
    """
    https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.CmaEsSampler.html
    """

    _target_: str = "optuna.samplers.CmaEsSampler"
    seed: Optional[int] = None

    x0: Optional[Dict[str, Any]] = None
    sigma0: Optional[float] = None
    independent_sampler: Optional[Any] = None
    warn_independent_sampling: bool = True
    consider_pruned_trials: bool = False
    restart_strategy: Optional[Any] = None
    inc_popsize: int = 2
    use_separable_cma: bool = False
    source_trials: Optional[Any] = None


@dataclass
class NSGAIISamplerConfig(SamplerConfig):
    """
    https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.NSGAIISampler.html
    """

    _target_: str = "optuna.samplers.NSGAIISampler"
    seed: Optional[int] = None

    population_size: int = 50
    mutation_prob: Optional[float] = None
    crossover_prob: float = 0.9
    swapping_prob: float = 0.5
    constraints_func: Optional[Any] = None


@dataclass
class MOTPESamplerConfig(SamplerConfig):
    """
    https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.MOTPESampler.html
    """

    _target_: str = "optuna.samplers.MOTPESampler"
    seed: Optional[int] = None

    consider_prior: bool = True
    prior_weight: float = 1.0
    consider_magic_clip: bool = True
    consider_endpoints: bool = False
    n_startup_trials: int = 10
    n_ehvi_candidates: int = 24


@dataclass
class DistributionConfig:
    # Type of distribution. "int", "float" or "categorical"
    type: DistributionType

    # Choices of categorical distribution
    # List element type should be Union[str, int, float, bool]
    choices: Optional[List[Any]] = None

    # Lower bound of int or float distribution
    low: Optional[float] = None

    # Upper bound of int or float distribution
    high: Optional[float] = None

    # If True, space is converted to the log domain
    # Valid for int or float distribution
    log: bool = False

    # Discritization step
    # Valid for int or float distribution
    step: Optional[float] = None


defaults = [{"sampler": "tpe"}]


@dataclass
class OptunaSweeperConf:
    _target_: str = "hydra_plugins.hydra_custom_sweeper.CustomSweeper"
    defaults: List[Any] = field(default_factory=lambda: defaults)

    # Sampling algorithm
    # Please refer to the reference for further details
    # https://optuna.readthedocs.io/en/stable/reference/samplers.html
    sampler: SamplerConfig = MISSING

    # Direction of optimization
    # Union[Direction, List[Direction]]
    direction: Any = Direction.minimize

    # Storage URL to persist optimization results
    # For example, you can use SQLite if you set 'sqlite:///example.db'
    # Please refer to the reference for further details
    # https://optuna.readthedocs.io/en/stable/reference/storages.html
    storage: Optional[Any] = None

    # Name of study to persist optimization results
    study_name: Optional[str] = None

    # Total number of function evaluations
    n_trials: int = 20

    # Number of parallel workers
    n_jobs: int = 2

    # Maximum authorized failure rate for a batch of parameters
    max_failure_rate: float = 0.0

    search_space: Optional[Dict[str, Any]] = None

    params: Optional[Dict[str, str]] = None

    # Allow custom trial configuration via Python methods.
    # If given, `custom_search_space` should be a an instantiate-style dotpath targeting
    # a callable with signature Callable[[DictConfig, optuna.trial.Trial], None].
    # https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/002_configurations.html
    custom_search_space: Optional[str] = None


ConfigStore.instance().store(
    group="hydra/sweeper",
    name="custom_sweeper",
    node=OptunaSweeperConf,
    provider="custom_sweeper",
)

ConfigStore.instance().store(
    group="hydra/sweeper/sampler",
    name="tpe",
    node=TPESamplerConfig,
    provider="custom_sweeper",
)

ConfigStore.instance().store(
    group="hydra/sweeper/sampler",
    name="random",
    node=RandomSamplerConfig,
    provider="custom_sweeper",
)

ConfigStore.instance().store(
    group="hydra/sweeper/sampler",
    name="cmaes",
    node=CmaEsSamplerConfig,
    provider="custom_sweeper",
)

ConfigStore.instance().store(
    group="hydra/sweeper/sampler",
    name="nsgaii",
    node=NSGAIISamplerConfig,
    provider="custom_sweeper",
)

ConfigStore.instance().store(
    group="hydra/sweeper/sampler",
    name="motpe",
    node=MOTPESamplerConfig,
    provider="custom_sweeper",
)

ConfigStore.instance().store(
    group="hydra/sweeper/sampler",
    name="grid",
    node=GridSamplerConfig,
    provider="custom_sweeper",
)

ConfigStore.instance().store(
    group="hydra/sweeper/sampler",
    name="bo",
    node=BOSamplerConfig,
    provider="custom_sweeper",
)

ConfigStore.instance().store(
    group="hydra/sweeper/sampler",
    name="gpyopt",
    node=GpyOptSamplerConfig,
    provider="custom_sweeper",
)

ConfigStore.instance().store(
    group="hydra/sweeper/sampler",
    name="pymoo",
    node=PyMooSamplerConfig,
    provider="custom_sweeper",
)

ConfigStore.instance().store(
    group="hydra/sweeper/sampler",
    name="botorch",
    node=BoTorchSamplerConfig,
    provider="custom_sweeper",
)
