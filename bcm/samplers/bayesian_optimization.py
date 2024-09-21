from typing import Any, List, Optional

from bayes_opt import BayesianOptimization

from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.core.plugins import Plugins
from omegaconf import DictConfig, OmegaConf


class BOSampler:
    is_optuna = False

    def __init__(self, seed, init_points):
        # Args and kwargs come from BOSamplerConfig
        self.seed = seed
        self.init_points = init_points
        self.optimizer = None

    def add_arguments(
        self,
        direction: Any,
        storage: Optional[Any],
        study_name: Optional[str],
        n_trials: int,
        n_jobs: int,
        max_failure_rate: float,
        search_space: Optional[DictConfig],
        custom_search_space: Optional[str],
        params: Optional[DictConfig],
    ):
        if direction == "minimize":
            # Bayesian optimization is maximizing
            self.sign = -1
        elif direction == "maximize":
            self.sign = 1
        else:
            raise ValueError(f"Direction {direction} not supported")
        self.storage = storage
        self.study_name = study_name
        self.n_trials = n_trials
        self.max_failure_rate = max_failure_rate
        assert self.max_failure_rate >= 0.0
        assert self.max_failure_rate <= 1.0
        self.search_space = search_space
        self.params = params
        self.job_idx: int = 0
        if n_jobs != 1:
            print(f"Ignoring n_jobs={n_jobs}. Running a single thread")

    def setup(self, hydra_context, task_function, config):
        self.config = config
        self.hydra_context = hydra_context
        self.launcher = Plugins.instance().instantiate_launcher(
            config=config, hydra_context=hydra_context, task_function=task_function
        )
        self.sweep_dir = config.hydra.sweep.dir

    def _parse_config(self) -> List[str]:
        params_conf = []
        for k, v in self.params.items():
            params_conf.append(f"{k}={v}")
        return params_conf

    def _query_function(self, **kwargs):
        overrides = [
            tuple(f"{name}={val}" for name, val in {**self.fixed, **kwargs}.items())
        ]
        # overrides can have more than one element if multiple jobs in parallel
        # (although this functionality is not supported by BOpt - see also assertion below)
        returns = self.launcher.launch(overrides, self.job_idx)
        self.job_idx += len(returns)
        assert len(returns) == 1, "Only single thread supported so far"
        ret = returns[0]
        assert (
            ret.status.value == 1
        ), f"Job has not been successfully completed: {ret.status}"
        return self.sign * ret.return_value

    @staticmethod
    def _overrides_to_pbounds(overrides):
        pbounds = {}
        fixed = {}
        for o in overrides:
            val = o.value()
            # Only care about overrides of sweep type
            if o.is_sweep_override():
                if o.is_interval_sweep():
                    pbounds[o.get_key_element()] = (val.start, val.end)
                else:
                    raise ValueError(f"Overrides {val} not supported")
            else:
                fixed[o.get_key_element()] = val
        return pbounds, fixed

    def sweep(self, arguments):
        assert self.config is not None
        assert self.launcher is not None
        assert self.hydra_context is not None
        assert self.job_idx is not None

        params_conf = self._parse_config()
        params_conf.extend(arguments)

        parser = OverridesParser.create(config_loader=self.hydra_context.config_loader)
        overrides = parser.parse_overrides(params_conf)
        pbounds, self.fixed = self._overrides_to_pbounds(overrides)

        self.optimizer = BayesianOptimization(
            f=self._query_function,
            pbounds=pbounds,
            random_state=self.seed,
            verbose=0,
        )
        self.optimizer.maximize(init_points=self.init_points, n_iter=self.n_trials)

        all_res = []
        for res in self.optimizer.res:
            all_res.append(
                {
                    "target": self.sign * float(res["target"]),
                    "params": {k: float(v) for k, v in res["params"].items()},
                }
            )
        OmegaConf.save(
            OmegaConf.create(all_res),
            f"{self.config.hydra.sweep.dir}/optimization_log.yaml",
        )
        OmegaConf.save(
            OmegaConf.create(
                {
                    "target": self.sign * float(self.optimizer.max["target"]),
                    "params": {
                        k: float(v) for k, v in self.optimizer.max["params"].items()
                    },
                }
            ),
            f"{self.config.hydra.sweep.dir}/optimization_best.yaml",
        )
