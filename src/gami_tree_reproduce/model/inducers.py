from abc import ABC, abstractmethod
from os import environ as oenv
from pathlib import Path
from typing import Literal, get_args

import numpy as np
from gaminet import GAMINetClassifier as GamiNetC
from gaminet import GAMINetRegressor as GamiNetR
from interpret.glassbox import ExplainableBoostingClassifier as EBMC
from interpret.glassbox import ExplainableBoostingRegressor as EBMR
from omegaconf import OmegaConf
from sklearn.metrics import log_loss, mean_squared_error
from typing_extensions import override
from xgboost import XGBClassifier as XGBC  # xgbs scikit learn api
from xgboost import XGBRegressor as XGBR  # xgbs scikit learn api

from gami_tree_reproduce.model.params import (
    EBMParams,
    GamiNetParams,
    Params,
    XGBParams,
)
from gami_tree_reproduce.utils import get_project_paths

from .params import get_parameter_class

project_paths = get_project_paths()

Task = Literal["classification", "regression"]


class BaseInducer(ABC):
    """ """

    @property
    @abstractmethod
    def classifier_class(self): ...

    @property
    @abstractmethod
    def regressor_class(self): ...

    @property
    def task(self) -> str:
        return self._task

    @property
    def name(self) -> str:
        return self._name

    @property
    @abstractmethod
    def param_class(self) -> type[Params]: ...

    @property
    def model(self):
        return self._model

    @property
    def params_wrapper(self):
        return self._params_wrapper

    def __init__(self, task: Task, params_wrapper: Params):
        """
        Create an Inducer object, based on parameters and task.
        Depending on task, a classsification or regression object from the implementing library is chosen.
        At initialization the constructor sets `_params` to the dictioary of the parameter wrapper and
        `_model` to the actual model wrapped in the inducer class.

        Args:
            params (Params): The parameters should be wrapped in a Params object, which validates the parameters.
            task (Task): The actual task, either "regression" or "classification".
        """
        if not isinstance(params_wrapper, self.param_class):
            msg = f"{self.__class__.__name__} expects params of type {self.param_class.__name__}, but got {type(params_wrapper).__name__}."
            raise TypeError(msg)

        if task not in get_args(Task):
            msg = f"Expected task to be type 'Task' but got {type(task)}"
            raise TypeError(msg)

        self._params_wrapper = params_wrapper
        self._task = task

        model_class = (
            self.classifier_class if task == "classification" else self.regressor_class
        )

        self._model = model_class(**self._params_wrapper.params)

    def hpo_pending(self) -> bool:
        return self.params_wrapper.hpo_pending

    def do_hpo(self, X_val, y_val) -> dict:
        """
        Optimize hyperparameters stored in self._hpo_setting of parameter wrapper

        Interact with parameter wrapper using mediator object handling hpo process.

        Args:
            X_val : Validationset (covariates)
            y_val : Validationset (labels)
        """
        hpo_settings = self.params_wrapper.hpo_settings
        HPOmediator.check_hpo_methods_set(hpo_settings)

        hpo_conf = {}
        hpo_runs = {}
        for hpo_param, hpo_setting in self.params_wrapper.hpo_settings.items():
            hpo_method = hpo_setting["method"]
            HPOmediator.check_hpo_method_registered(hpo_method)
            hpo_result, hpo_config = HPOmediator.do_hpo(
                hpo_param=hpo_param,
                method=hpo_method,
                hpo_setting=hpo_setting,
                inducer=self,
                X_val=X_val,
                y_val=y_val,
            )
            hpo_conf.update({hpo_param: hpo_result})
            hpo_runs[hpo_param] = hpo_config

        self.set_params_inducer(hpo_conf)
        self.params_wrapper._hpo_pending = False

        return hpo_runs

    @abstractmethod
    def set_params_inducer(self, param_dict: dict) -> None:
        """
        Set parameters for inducer object.

        First validate parameters (implemented in underlying package) and set for parameter wrapper.
        Subclasses must set parameters for the internal model since this is model/API specific.

        Args:
            param_dict (dict): Plain dictinary with parameter name as key and value.
        """
        self.params_wrapper._validate_params(param_dict)
        self.params_wrapper.set_params(param_dict)

    @abstractmethod
    def train(self, X_train, y_train) -> np.ndarray: ...

    @abstractmethod
    def predict(self, X_new) -> np.ndarray: ...


# =======================================================================================
#                           Derived Inducer Classes
# =======================================================================================
class EBMinducer(BaseInducer):
    @property
    def classifier_class(self) -> type[EBMC]:
        return EBMC

    @property
    def regressor_class(self) -> type[EBMR]:
        return EBMR

    @property
    def name(self) -> str:
        return "ebm"

    @property
    def param_class(self):
        return EBMParams

    @override
    def set_params_inducer(self, param_dict):
        super().set_params_inducer(param_dict)
        self._model.set_params(**param_dict)

    @override
    def train(self, X, y) -> np.ndarray:
        self._model.fit(X, y)
        if self._task == "regression":
            loss_trace = mean_squared_error(self._model.predict(X), y)
        else:
            loss_trace = log_loss(self._model.predict(X), y)
        return loss_trace

    @override
    def predict(self, X):
        return self._model.predict(X)


class XGBinducer(BaseInducer):
    @property
    def classifier_class(self) -> type[XGBC]:
        return XGBC

    @property
    def regressor_class(self) -> type[XGBR]:
        return XGBR

    @property
    def name(self) -> str:
        return "xgb"

    @property
    def param_class(self):
        return XGBParams

    @override
    def set_params_inducer(self, param_dict):
        super().set_params_inducer(param_dict)
        self._model.set_params(**param_dict)

    @override
    def train(self, X_train, y_train) -> np.array:
        self._model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)
        loss = self._model.evals_result_["validation_0"]["logloss"]
        return np.array(loss)

    @override
    def predict(self, X_new) -> np.ndarray:
        return self._model.predict(X_new)


class GamiNetInducer(BaseInducer):
    @property
    def name(self) -> str:
        return "gaminet"

    @property
    def classifier_class(self) -> type[GamiNetC]:
        return GamiNetC

    @property
    def regressor_class(self) -> type[GamiNetR]:
        return GamiNetR

    @property
    def param_class(self):
        return GamiNetParams

    @override
    def set_params_inducer(self, param_dict):
        super().set_params_inducer(param_dict)
        self._model.set_params(**param_dict)

    @override
    def train(self, X, y) -> None:
        self._model.fit(X, y)

    @override
    def predict(self, X) -> np.ndarray:
        return self._model.predict(X)


INDUCER_REGISTRY = {
    "ebm": EBMinducer,
    "xgb": XGBinducer,
    "gaminet": GamiNetInducer,
}


def get_inducer_class(name: str) -> callable:
    key = name.lower()
    if key not in INDUCER_REGISTRY:
        msg = f"There is no inducer '{key}' that is registered."
        raise KeyError(msg)

    return INDUCER_REGISTRY[key]


# =======================================================================================
#                           HPO management
# =======================================================================================


conf = OmegaConf.load(Path(oenv["PROJECT_ROOT"], "conf", "config.yaml"))
rng = np.random.default_rng(conf.seed)


class HPOmediator(ABC):
    @abstractmethod
    def check_hpo_methods_set(hpo_settings: dict):
        for hpo_param, hpo_setting in hpo_settings.items():
            if "method" not in hpo_setting:
                msg = f"Expected 'method: <sth>' enty for hpo parameter '{hpo_param}'"
                raise KeyError(msg)

    @abstractmethod
    def check_hpo_method_registered(method_name: str):
        if method_name not in HPO_METHOD_REGISTRY:
            msg = f"Method '{method_name}' is not implemented and/or registered in method registry"
            raise KeyError(msg)

    @abstractmethod
    def do_grid_search(
        hpo_param: str, grid: list, inducer, X_val: np.array, y_val: np.array
    ):
        configurations = {}
        inducer_name = inducer.name
        task = inducer.task
        default_params = inducer.params_wrapper.params
        best_loss = float("inf")
        for candidate_value in grid:
            new_params = default_params.update({hpo_param: candidate_value})
            candidate_params_wrapper = get_parameter_class(inducer_name)(
                params=new_params, task=task
            )
            candidate_inducer = get_inducer_class(inducer_name)(
                task=task, params_wrapper=candidate_params_wrapper
            )
            candidate_inducer.set_params_inducer({hpo_param: candidate_value})
            loss_trace = candidate_inducer.train(X_val, y_val)
            if isinstance(
                loss_trace, float | int
            ):  # if no trace but just final loss provided
                current_loss = loss_trace
            else:
                current_loss = loss_trace[-1]
            configurations.update({candidate_value: current_loss})
            if current_loss < best_loss:
                best_loss = current_loss
                best_value = candidate_value

        return (best_value, configurations)

    @abstractmethod
    def do_random_search(
        hpo_param,
        distribution: np.random.Generator,
        params: dict,
        inducer,
        X_val,
        y_val,
    ):
        random_grid = distribution(**params)
        return HPOmediator.do_grid_search(hpo_param, random_grid, inducer, X_val, y_val)

    @abstractmethod
    def do_hpo(hpo_param: str, method: str, hpo_setting: dict, inducer, X_val, y_val):
        if method == "random":
            if "distribution" not in hpo_setting:
                msg = "For method 'random' expected 'distribution' keyword that matches numpy generator."
                raise KeyError(msg)
            if not hasattr(rng, distribution := hpo_setting["distribution"]):
                msg = f"Numpy rng hast no implementation of desired distibution {distribution}"
                raise KeyError(msg)
            if "params" not in hpo_setting:
                msg = f"Expected 'params' key to initialize distribution {hpo_setting['distribution']}"
                raise KeyError(msg)

            distribution = getattr(rng, hpo_setting["distribution"])
            params = hpo_setting["params"]
            hpo_result = HPOmediator.do_random_search(
                hpo_param, distribution, params, inducer, X_val, y_val
            )
        elif method == "grid":
            hpo_result = HPOmediator.do_grid_search(
                hpo_param, hpo_setting["grid"], inducer, X_val, y_val
            )
        else:
            raise KeyError
        return hpo_result


HPO_METHOD_REGISTRY = {
    "grid": HPOmediator.do_grid_search,
    "random": HPOmediator.do_random_search,
}
