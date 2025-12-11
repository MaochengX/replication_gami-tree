"""
For each inducer wrap parameter dictionary in a wrapper class.
At initialization check whether the configuration is valid (key match expected keys from API)
and for some parameters set to default if no user input is provided.
"""

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, TypeVar, get_args

from gaminet import GAMINet
from interpret.glassbox import ExplainableBoostingClassifier as EBMC
from interpret.glassbox import ExplainableBoostingRegressor as EBMR
from xgboost import XGBClassifier as XGBC
from xgboost import XGBRegressor as XGBR


# TODO: Forward declarations for now until implemented
class GAMITR: ...


class GAMITC: ...


Task = Literal["regression", "classification"]


@dataclass
class BaseParams(ABC):
    """Base class for all parameter types."""

    def __init__(self, task: Task, params: dict | None = None) -> None:
        """ """
        if params is None:
            params = {}

        expected_tasks = get_args(Task)
        if task not in expected_tasks:
            msg = f"Expected task to be in {expected_tasks} but got '{task}'."
            raise TypeError(msg)
        self._task = task

        if not isinstance(params, dict):
            msg = f"Expected parameters to be a dictionary, got {type(params)}"
            raise TypeError(msg)

        # validate the keys
        self._validate_params(params)
        # Set default and overwrite
        params = self._fill_defaults(params)
        self._params = params

    @abstractmethod
    def _validate_params(self, params: dict) -> bool: ...

    @abstractmethod
    def _fill_defaults(self, params) -> dict: ...

    def get_params(self) -> dict:
        return self._params


Params = TypeVar("Params", bound=BaseParams)


class EBMParams(BaseParams):
    """
    Docstring for EBMParams Class
    """

    def _validate_params(self, params: Params) -> None:
        """ """
        keys_to_check = list(params.keys())
        task = self._task

        if task == "regression":
            expected_parameter_keys = list(EBMR().get_params().keys())
        elif task == "classification":
            expected_parameter_keys = list(EBMC().get_params().keys())
        else:
            msg = f"Got unexpected task {task}."
            raise ValueError(msg)

        conditions = [key in expected_parameter_keys for key in keys_to_check]
        faulty_keys = [
            key
            for key, condition in zip(keys_to_check, conditions, strict=True)
            if not condition
        ]
        if not all(conditions):
            msg = f"Expected parameters to be in {expected_parameter_keys} but got {faulty_keys}"
            raise ValueError(msg)

    def _fill_defaults(self, params) -> dict:
        if self._task == "classification":
            params_defaults = EBMC().get_params()
        elif self._task == "regression":
            params_defaults = EBMR().get_params()
        params_defaults.update(params)
        return params_defaults


class XGBParams(BaseParams):
    """
    Docstring for XGBParams Class
    """

    def _validate_params(self, params: Params) -> None:
        """ """
        keys_to_check = list(params.keys())
        task = self._task

        if task == "regression":
            expected_parameter_keys = list(XGBR().get_params().keys())
        elif task == "classification":
            expected_parameter_keys = list(XGBC().get_params().keys())

        conditions = [key in expected_parameter_keys for key in keys_to_check]
        faulty_keys = [
            key
            for key, condition in zip(keys_to_check, conditions, strict=True)
            if not condition
        ]
        if not all(conditions):
            msg = f"Expected parameters to be in {expected_parameter_keys} but got {faulty_keys}"
            raise ValueError(msg)

    def _fill_defaults(self, params: dict):
        if self._task == "regression":
            params_default = XGBR().get_params()
        elif self._task == "classification":
            params_default = XGBC().get_params()
        params_default.update(params)
        return params_default


class GAMITParams(BaseParams):
    def _validate_params(self, params: dict) -> None:
        pass


class GamiNetParams(BaseParams):
    def _validate_params(self, params: dict) -> None:
        keys_to_check = list(params.keys())
        signature = get_signature(GAMINet)
        expected_parameter_keys = list(signature.keys())

        conditions = [key in expected_parameter_keys for key in keys_to_check]
        faulty_keys = [
            key
            for key, condition in zip(keys_to_check, conditions, strict=True)
            if not condition
        ]

        if not all(conditions):
            msg = f"Expected parameters to be in {expected_parameter_keys} but got {faulty_keys}"
            raise ValueError(msg)

    def __init__(self, task: Task, params: dict):
        # API uses Catital case, see https://github.com/ZebinYang/gaminet
        if task == "regression":
            params["task_type"] = "Regression"
        else:
            params["task_type"] = "Classification"


PARAMS_REGISTRY = {"ebm": EBMParams, "xgb": XGBParams}


def get_parameter(name: str) -> callable:
    key = name.lower()
    if key not in PARAMS_REGISTRY:
        raise ValueError

    return PARAMS_REGISTRY[key]


def get_signature(cls):
    sig = inspect.signature(cls)
    args = {}
    for name, param in sig.parameters.items():
        if param.default is inspect._empty:
            args[name] = None  # no default value
        else:
            args[name] = param.default
    return args
