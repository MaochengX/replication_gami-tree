from abc import ABC, abstractmethod
from typing import Literal, get_args

import numpy as np
import pandas as pd

# from gaminet import GAMINet
from interpret.glassbox import ExplainableBoostingClassifier as EBMC
from interpret.glassbox import ExplainableBoostingRegressor as EBMR
from typing_extensions import override
from xgboost import XGBClassifier as XGBC  # xgbs scikit learn api
from xgboost import XGBRegressor as XGBR  # xgbs scikit learn api

from gami_tree_reproduce.model.params import BaseParams, EBMParams, Params, XGBParams
from gami_tree_reproduce.utils import get_project_paths

project_paths = get_project_paths()

Task = Literal["classification", "regression"]


class BaseInducer(ABC):
    """
    Docstring for BaseInducer

    :var Args: Description
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

        self._params_wrapper = params_wrapper
        self._task = task
        self._is_trained = False

        self._X_train = None
        self._y_train = None
        self._X_val = None
        self._y_val = None
        self._X_test = None
        self._y_test = None

        model_class = (
            self.classifier_class if task == "classification" else self.regressor_class
        )

        self._model = model_class(**self._params_wrapper.get_params())

    @abstractmethod
    def do_hpo(self) -> None: ...

    @abstractmethod
    def train(self) -> None: ...

    @abstractmethod
    def test(self) -> np.ndarray: ...

    def set_data(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_val: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
    ) -> None:
        self._X_train = X_train
        self._y_train = y_train
        self._X_val = X_val
        self._y_val = y_val
        self._X_test = X_test
        self._y_test = y_test

    # -----------------
    #       Checks
    # -----------------

    def _assert_data_is_set(self):
        assert self._X_train is not None
        assert self._y_train is not None
        assert self._X_val is not None
        assert self._y_val is not None
        assert self._X_test is not None
        assert self._y_test is not None

    def _assert_no_hpo_params(self, hpo_keyword: str = "tune"):
        param_values = self._params_wrapper.get_params().values()
        hpo_candidates = [item for item in param_values if item == hpo_keyword]
        assert len(hpo_candidates) == 0

    # -----------------
    #       Getters
    # -----------------

    def get_params_wrapper(self) -> BaseParams:
        return self._params_wrapper

    def get_task(self) -> str:
        return self._task

    def get_model(self) -> dict:
        return self._model

    def get_data_train(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        self._assert_data_is_set()
        return self._X_train, self._y_train

    def get_data_val(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        self._assert_data_is_set()
        return self._X_val, self._y_val

    def get_data_test(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        self._assert_data_is_set()
        return self._X_test, self._y_test


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

    @override
    def do_hpo(self) -> None:
        self._assert_data_is_set()

        params_object = self.get_params_wrapper()
        optimized_params = {}
        for hpo_param, param_config in params_object.get_hpo_params().items():
            optimal_value = param_config[param_config["method"]][0]
            optimized_params.update({hpo_param: optimal_value})
        self._params_wrapper.set_optimized_hpo_params(optimized_params)
        self._model.set_params(**optimized_params)

    @override
    def train(self) -> None:
        self._assert_data_is_set()
        self._assert_no_hpo_params()

        self._model.fit(self._X_train, self._y_train)

    @override
    def test(self) -> np.ndarray:
        self._assert_data_is_set()
        return self._model.predict(self._X_test)


# class GamiNetInducer(BaseInducer):
#     """ "
#     Docstring
#     """

#     @property
#     def classifier_class(self) -> type[GAMINet]:
#         return GAMINet

#     @property
#     def regressor_class(self) -> type[GAMINet]:
#         return GAMINet

#     @property
#     def param_class(self):
#         return GamiNetParams

#     def train(self, X, y) -> None:
#         self._model.fit(X, y)

#     def predict(self, X) -> np.ndarray:
#         return self._model.predict(X)


INDUCER_REGISTRY = {"ebm": EBMinducer, "xgb": XGBinducer}


def get_inducer_class(name: str) -> callable:
    key = name.lower()
    if key not in INDUCER_REGISTRY:
        msg = f"There is no inducer '{key}' that is registered."
        raise KeyError(msg)

    return INDUCER_REGISTRY[key]
