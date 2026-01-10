"""
For each inducer wrap parameter dictionary in a wrapper class.
At initialization check whether the configuration is valid (key match expected keys from API)
and for some parameters set to default if no user input is provided.
"""

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from numbers import Number
from typing import Literal, TypeVar, get_args

# from gaminet import GAMINet
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
        self._set_params(params.copy())  # copy here due to pop operation
        # HPO parameters have key value tune by default and no further information
        # additional information of HPO is stored in self._hpo_params
        # when HPO is performed the values in self._params are set from 'tune' to the optimal value
        self._set_initial_hpo_params(params)

    # -----------------
    #       Setters
    # -----------------

    @abstractmethod
    def _set_params() -> None: ...

    @abstractmethod
    def _set_initial_hpo_params() -> None: ...

    def set_optimized_hpo_params(self, optimal_configuration: dict) -> None:
        for param, optimal_value in optimal_configuration.items():
            if param not in self._params:
                msg = (
                    f"Found no hyperparameter {param} in initialized defaul dictionary"
                )
                raise KeyError(msg)
            self._params[param] = optimal_value

    # -----------------
    #       Checks
    # -----------------

    # needs inducer specific implementation
    @abstractmethod
    def _validate_params(self, params: dict) -> bool: ...

    def _check_param_key_names(self, keys: list[str], keys_expected: list[str]) -> None:
        unknown_keys = [key for key in keys if key not in keys_expected]
        if len(unknown_keys) >= 1:
            msg = f"Got unexpected parameter keywords: {unknown_keys}"
            raise KeyError(msg)

    def _assert_hpo_method_implemented(
        self, hpo_param: str, param_method: str, param_conf: dict
    ):
        action_space = param_conf[param_method]

        match param_method:
            case "grid":
                if not isinstance(action_space, list):
                    msg = f"For HPO method 'grid'' expected a list but got {type(action_space)} for param '{hpo_param}'"
                    raise KeyError(msg)
                if not all(isinstance(item, Number) for item in action_space):
                    msg = f"Expected action space for 'grid' to be numbers in '{hpo_param}'"
                    raise KeyError(msg)
            case "categorical":
                if not isinstance(action_space, list):
                    msg = f"For HPO method 'categorical'' expected a list but got {type(action_space)} for param '{hpo_param}'"
                    raise KeyError(msg)
                if not all(isinstance(item, str) for item in action_space):
                    msg = f"Expected action space for 'categorical' to be strings in '{hpo_param}'"
                    raise KeyError(msg)

    def _assert_hpo_params(self, hpo_params: dict):
        # check if each hpo param has provided method
        supported_methods = ["grid", "categorical"]
        for hpo_param, param_conf in hpo_params.items():
            if "method" not in param_conf:
                msg = f"Expected to find 'method' entry in configuration for parameter {hpo_param}"
                raise KeyError(msg)
            param_method = param_conf["method"]
            if param_method not in supported_methods:
                msg = f"Got method '{param_method}', but support only for {supported_methods}"
                raise KeyError(msg)
            if param_method not in param_conf:
                msg = f"Expected to find entry in form of '{param_method}: <action space>' but found no '{param_method}' key for param '{hpo_param}'"
                raise KeyError(msg)

            self._assert_hpo_method_implemented(hpo_param, param_method, param_conf)

    # -----------------
    #       Getters
    # -----------------

    def get_params(self) -> dict:
        return self._params

    def get_hpo_params(self) -> dict:
        return self._hpo_params


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

    def _validate_params(self, params: Params, hpo_keyword: str = "tune") -> None:
        hpo_params = params.pop(hpo_keyword)
        hpo_keys = list(hpo_params.keys())
        param_keys = list(params.keys())
        task = self._task

        if task == "regression":
            expected_parameter_keys = list(XGBR().get_params().keys())
            self._check_param_key_names(param_keys, expected_parameter_keys)
        elif task == "classification":
            expected_parameter_keys = list(XGBC().get_params().keys())
            self._check_param_key_names(param_keys, expected_parameter_keys)

        self._check_param_key_names(hpo_keys, expected_parameter_keys)
        params.update({hpo_keyword: hpo_params})

    def _set_params(self, params: dict, hpo_keyword: str = "tune") -> None:
        hpo_params = params.pop(hpo_keyword) if hpo_keyword in params else {}

        if self._task == "regression":
            params_default = XGBR().get_params()
        elif self._task == "classification":
            params_default = XGBC().get_params()

        params_default.update(params)
        params_default.update(dict.fromkeys(hpo_params, hpo_keyword))

        self._params = params_default

    def _set_initial_hpo_params(self, params: dict, hpo_keyword: str = "tune") -> None:
        if hpo_keyword in params:
            hpo_params = params.pop(hpo_keyword)
            self._assert_hpo_params(hpo_params)
            self._hpo_params = hpo_params
        else:
            self._hpo_params = {}


class GamiTreeParams(BaseParams):
    def _validate_params(self, params: dict) -> None:
        pass


# class GamiNetParams(BaseParams):
#     def _validate_params(self, params: dict) -> None:
#         keys_to_check = list(params.keys())
#         signature = get_signature(GAMINet)
#         expected_parameter_keys = list(signature.keys())

#         conditions = [key in expected_parameter_keys for key in keys_to_check]
#         faulty_keys = [
#             key
#             for key, condition in zip(keys_to_check, conditions, strict=True)
#             if not condition
#         ]

#         if not all(conditions):
#             msg = f"Expected parameters to be in {expected_parameter_keys} but got {faulty_keys}"
#             raise ValueError(msg)

#     def __init__(self, task: Task, params: dict):
#         # API uses Catital case, see https://github.com/ZebinYang/gaminet
#         if task == "regression":
#             params["task_type"] = "Regression"
#         else:
#             params["task_type"] = "Classification"

#     def _fill_defaults(self, params: dict, task: str) -> dict:
#         defaults = get_gaminet_defaults(task)
#         params.update(defaults)


# def get_gaminet_defaults(task: str) -> dict:
#     signature = inspect.signature(GAMINet)
#     defaults = {
#         name: param.default
#         for name, param in signature.parameters.items()
#         if param.default is not inspect._empty()
#     }
#     if task == "regression":
#         defaults["task_type"] = "Regression"
#     else:
#         defaults["task_type"] = "Classification"
#     return defaults


PARAMS_REGISTRY = {"ebm": EBMParams, "xgb": XGBParams}


def get_parameter_class(name: str) -> callable:
    key = name.lower()
    if key not in PARAMS_REGISTRY:
        msg = f"No Parameter Class defined for given keyword '{key}'"
        raise KeyError(msg)

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
