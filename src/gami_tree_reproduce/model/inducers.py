"""
Wrap each inducer in a wrapper class that provides common API.
Since most of the ML algorithms used are not standard each inducer has its own API that requires different implementation
for parameter tracking, training and prediction.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal, get_args

import mlflow
import numpy as np
from gaminet import GAMINet
from interpret.glassbox import ExplainableBoostingClassifier as EBMC
from interpret.glassbox import ExplainableBoostingRegressor as EBMR
from xgboost import XGBClassifier as XGBC
from xgboost import XGBRegressor as XGBR

from gami_tree_reproduce.model.params import EBMParams, GamiNetParams, Params, XGBParams

Task = Literal["classification", "regression"]

ROOT = Path.cwd()
ASSETS = ROOT / "assets"
mlflow.set_tracking_uri(ASSETS)


class BaseInducer(ABC):
    """
    Class template for inducer classes.
    This serves as a wrapper for the various imported inducer algorithms.
    """

    @property
    @abstractmethod
    def classifier_class(self): ...

    @property
    @abstractmethod
    def regressor_class(self): ...

    @property
    @abstractmethod
    def param_class(self) -> type[Params]: ...

    @property
    def model(self):
        return self._model

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

        self._params = params_wrapper.get_params()
        self._task = task

        model_class = (
            self.classifier_class if task == "classification" else self.regressor_class
        )

        self._model = model_class(**self._params)

    def get_params(self) -> dict:
        return self._params

    def get_task(self) -> str:
        return self._task

    @abstractmethod
    def train(self, X: Any, y: Any) -> np.ndarray: ...

    @abstractmethod
    def predict(self, X: Any) -> np.ndarray: ...


class EBMinducer(BaseInducer):
    """
    Docstring for EBMinducer
    """

    @property
    def classifier_class(self) -> type[EBMC]:
        return EBMC

    @property
    def regressor_class(self) -> type[EBMR]:
        return EBMR

    @property
    def param_class(self):
        return EBMParams

    def train(self, X, y) -> None:
        self._model.fit(X, y)
        return self

    def predict(self, X):
        pass


class XGBinducer(BaseInducer):
    """
    Docstring for XGBinducer
    """

    @property
    def classifier_class(self) -> type[XGBC]:
        return XGBC

    @property
    def regressor_class(self) -> type[XGBR]:
        return XGBR

    @property
    def param_class(self):
        return XGBParams

    def train(self, X, y) -> None:
        with mlflow.start_run(run_name="xgb"):
            mlflow.log_params(self._params)
            self._model.fit(X, y)

    def predict(self, X) -> np.ndarray:
        return self._model.predict(X)


class GamiNetInducer(BaseInducer):
    """ "
    Docstring
    """

    @property
    def classifier_class(self) -> type[GAMINet]:
        return GAMINet

    @property
    def regressor_class(self) -> type[GAMINet]:
        return GAMINet

    @property
    def param_class(self):
        return GamiNetParams

    def train(self, X, y) -> None:
        self._model.fit(X, y)

    def predict(self, X) -> np.ndarray:
        return self._model.predict(X)


######################################################################################
# for usage only access should be done via imported getter function and class registry
######################################################################################

INDUCER_REGISTRY = {"ebm": EBMinducer, "xgb": XGBinducer, "gaminet": GamiNetInducer}


def get_inducer(name: str) -> callable:
    key = name.lower()
    if key not in INDUCER_REGISTRY:
        raise ValueError

    return INDUCER_REGISTRY[key]
