from typing import Any, List, Optional

import numpy as np
from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.core.plugins import Plugins
from omegaconf import DictConfig, OmegaConf

from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.termination import get_termination


class PyMooProblem(ElementwiseProblem):
    def __init__(self, bounds, fixed, launcher, direction):
        # Do not overwrite bounds variable
        self.job_idx: int = 0
        self.names_and_bounds = bounds
        self.fixed = fixed
        self.launcher = launcher
        self.sign = -1 if direction == "maximize" else 1
        xl = np.array([b["xl"] for b in self.names_and_bounds])
        xu = np.array([b["xu"] for b in self.names_and_bounds])
        super().__init__(n_var=len(self.names_and_bounds), n_obj=1, xl=xl, xu=xu)

    def _evaluate(self, x, out):
        # Create query function as the optimizer needs a Callable
        # GPyOpt needs a function with a single variable
        # Values are unnamed and structured in a NumPy array [batch_size, num_params]
        # preserving the order of self.bounds
        kwargs = {self.names_and_bounds[i]["name"]: val for i, val in enumerate(x)}
        overrides = [
            tuple(f"{name}={val}" for name, val in {**self.fixed, **kwargs}.items())
        ]
        # This line calls the main program with the parameters in overrides
        returns = self.launcher.launch(overrides, self.job_idx)

        self.job_idx += len(returns)
        assert len(returns) == 1
        out["F"] = self.sign * returns[0].return_value


class PyMooSampler:
    is_optuna = False

    def __init__(
        self,
        seed,
        algorithm_name,
        sampling_name,
        pop_size,
    ):
        # Args and kwargs come from PyMooSamplerConfig in hydra_plugins/config.py
        self.seed = seed
        self.algorithm_name = algorithm_name
        self.sampling_name = sampling_name
        self.pop_size = pop_size
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
        # This method is called in hydra_plugins/hydra_custom_sweeper.py initialization
        self.direction = direction
        assert self.direction in [
            "maximize",
            "minimize",
        ], f"Direction {self.direction} not supported"
        self.storage = storage
        self.study_name = study_name
        self.n_trials = n_trials
        self.max_failure_rate = max_failure_rate
        assert self.max_failure_rate >= 0.0
        assert self.max_failure_rate <= 1.0
        self.search_space = search_space
        self.params = params
        self.n_jobs = n_jobs

    def setup(self, hydra_context, task_function, config):
        # This method is called in hydra_plugins/hydra_custom_sweeper.py setup
        # The setup is run by hydra implicitly before calling sweep
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

    @staticmethod
    def _overrides_to_bounds(overrides):
        # Overrides are all the parameters in the command line that we add
        # to replace the default parameters in the hydra config
        # To that, the sweep parameters (intervals, discrete spaces, etc.)
        # are also added
        bounds = []
        fixed = {}
        for o in overrides:
            val = o.value()
            # Only care about overrides of sweep type
            if o.is_sweep_override():
                if o.is_interval_sweep():
                    bounds.append(
                        {
                            "name": o.get_key_element(),
                            "xl": val.start,
                            "xu": val.end,
                        }
                    )
                else:
                    raise ValueError(f"Overrides {val} not supported")
            else:
                fixed[o.get_key_element()] = val
        return bounds, fixed

    def sweep(self, arguments):
        # This method is called in hydra_plugins/hydra_custom_sweeper.py sweep
        # The sweep is run by hydra implicitly
        assert self.config is not None
        assert self.launcher is not None
        assert self.hydra_context is not None

        params_conf = self._parse_config()
        params_conf.extend(arguments)

        parser = OverridesParser.create(config_loader=self.hydra_context.config_loader)
        overrides = parser.parse_overrides(params_conf)
        bounds, fixed = self._overrides_to_bounds(overrides)

        problem = PyMooProblem(bounds, fixed, self.launcher, self.direction)

        if self.sampling_name == "lhs":
            from pymoo.operators.sampling.lhs import LHS

            sampling = LHS()
        else:
            raise ValueError(f"Sampling {self.sampling_name} not supported")

        if self.algorithm_name == "de":
            from pymoo.algorithms.soo.nonconvex.de import DE

            algorithm = DE(pop_size=self.pop_size, sampling=sampling)
        else:
            raise ValueError(f"Algorithm {self.algorithm_name} not supported")

        termination = get_termination("n_eval", self.n_trials)

        res = minimize(
            problem,
            algorithm,
            termination=termination,
            seed=self.seed,
            save_history=True,
        )

        all_res = []
        all_vals = []
        for hist in res.history:
            for f, x in zip(hist.pop.get("F"), hist.pop.get("X")):
                assert (
                    len(f) == 1
                ), "Multi-objective optimization not supported for save"
                assert len(x) == len(
                    bounds
                ), "Parameters do not correspond to the bounds"
                all_vals.append(f[0])
                all_res.append(
                    {
                        "target": float(f[0]),
                        "params": {
                            param_i["name"]: float(x_i)
                            for param_i, x_i in zip(bounds, x)
                        },
                    }
                )

        OmegaConf.save(
            OmegaConf.create(all_res),
            f"{self.config.hydra.sweep.dir}/optimization_log.yaml",
        )
        OmegaConf.save(
            OmegaConf.create(all_res[np.argmin(np.array(all_vals))]),
            f"{self.config.hydra.sweep.dir}/optimization_best.yaml",
        )
