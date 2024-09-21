from typing import Any, List, Optional

import numpy as np
from GPyOpt.methods import BayesianOptimization
from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.core.plugins import Plugins
from omegaconf import DictConfig, OmegaConf




class BOHydra(BayesianOptimization):
    """ Override the Bayesian Optimization to save the results
    during the run"""
    def __init__(self, hydra_sweep_dir, **kwargs):
        self.hydra_sweep_dir = hydra_sweep_dir
        super(BOHydra, self).__init__(**kwargs)

    def evaluate_objective(self):
        "This is run during evaluation  in the while iteration loop l151 "
        # First evaluate the objective
        self.Y_new, cost_new = self.objective.evaluate(self.suggested_sample)

        # Update the cost model / GP
        self.cost.update_cost_model(self.suggested_sample, cost_new)

        # Add the new values
        self.Y = np.vstack((self.Y, self.Y_new))

        # Save the new values
        if (self.num_acquisitions + 1) % 10 == 0:
            self.save_results_optuna()

    def save_results_optuna(self):
        # all_res = Dict[str, Any]
        # all_res = List[Trial]
        all_res = []
        for x, y in zip(self.X, self.Y):
            assert len(y) == 1, "Multi-objective optimization not supported for save"
            assert len(x) == len(
                self.domain
            ), "The parameters do not correspond to the domain"
            all_res.append(
                {
                    "target": str(y[0]),
                    "params": {
                        param_i["name"]: float(x_i) for param_i, x_i in zip(self.domain, x)
                    },
                }
            )
        OmegaConf.save(
            OmegaConf.create(all_res),
            f"{self.hydra_sweep_dir}/optimization_log.yaml",
        )
        OmegaConf.save(
            OmegaConf.create(all_res[np.argmin(self.Y)]),
            f"{self.hydra_sweep_dir}/optimization_best.yaml",
        )

class BOSampler:
    is_optuna = False

    def __init__(
        self,
        seed,
        model_type,
        initial_design_numdata,
        initial_design_type,
        acquisition_type,
        acquisition_optimizer_type,
    ):
        # Args and kwargs come from GpyOptSamplerConfig in hydra_plugins/config.py
        self.seed = seed
        self.model_type = model_type
        self.initial_design_numdata = initial_design_numdata
        self.initial_design_type = initial_design_type
        self.acquisition_type = acquisition_type
        self.acquisition_optimizer_type = acquisition_optimizer_type
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
        self.job_idx: int = 0
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

    def _query_function(self, params):
        # Create query function as the optimizer needs a Callable
        # GPyOpt needs a function with a single variable
        # Values are unnamed and structured in a NumPy array [batch_size, num_params]
        # preserving the order of self.domain
        overrides = []
        for param in params:
            kwargs = {self.domain[i]["name"]: val for i, val in enumerate(param)}
            overrides.append(
                tuple(f"{name}={val}" for name, val in {**self.fixed, **kwargs}.items())
            )

        # This line calls the main program with the parameters in overrides
        returns = self.launcher.launch(overrides, self.job_idx)

        self.job_idx += len(returns)
        return [float(ret.return_value) for ret in returns]

    @staticmethod
    def _overrides_to_domain(overrides):
        # Overrides are all the parameters in the command line that we add
        # to replace the default parameters in the hydra config
        # To that, the sweep parameters (intervals, discrete spaces, etc.)
        # are also added
        domain = []
        fixed = {}
        for o in overrides:
            val = o.value()
            # Only care about overrides of sweep type
            if o.is_sweep_override():
                if o.is_interval_sweep():
                    domain.append(
                        {
                            "name": o.get_key_element(),
                            "type": "continuous",
                            "domain": (val.start, val.end),
                        }
                    )
                elif o.is_range_sweep():
                    # Treat range sweep the same as interval (ignore step and shuffle)
                    domain.append(
                        {
                            "name": o.get_key_element(),
                            "type": "continuous",
                            "domain": (val.start, val.stop),
                        }
                    )
                else:
                    raise ValueError(f"Overrides {val} not supported")
            else:
                fixed[o.get_key_element()] = val
        return domain, fixed

    def sweep(self, arguments):
        # This method is called in hydra_plugins/hydra_custom_sweeper.py sweep
        # The sweep is run by hydra implicitly
        assert self.config is not None
        assert self.launcher is not None
        assert self.hydra_context is not None
        assert self.job_idx is not None

        params_conf = self._parse_config()
        params_conf.extend(arguments)

        parser = OverridesParser.create(config_loader=self.hydra_context.config_loader)
        overrides = parser.parse_overrides(params_conf)
        self.domain, self.fixed = self._overrides_to_domain(overrides)
        self.optimizer = BOHydra(
            f=self._query_function,
            domain=self.domain,
            model_type=self.model_type,
            initial_design_numdata=self.initial_design_numdata,
            initial_design_type=self.initial_design_type,
            acquisition_type=self.acquisition_type,
            acquisition_optimizer_type=self.acquisition_optimizer_type,
            batch_size=self.n_jobs,
            maximize=self.direction == "maximize",
            hydra_sweep_dir=self.config.hydra.sweep.dir
        )
        self.optimizer.run_optimization(max_iter=self.n_trials)

        self.optimizer.save_results_optuna()

