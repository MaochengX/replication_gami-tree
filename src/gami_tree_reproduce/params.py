from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, TypeVar, get_args

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

    def __init__(self, task: Task, params: dict) -> None:
        """ """
        expected_tasks = get_args(Task)
        if task not in expected_tasks:
            msg = f"Expected task to be in {expected_tasks} but got '{task}'."
            raise TypeError(msg)

        if not isinstance(params, dict):
            msg = f"Expected parameters to be a dictionary, got {type(params)}"
            raise TypeError(msg)

        self._validate_params(task, params)
        self._parms = params

    @abstractmethod
    def _validate_params(self, task: Task, params: dict) -> bool: ...


Params = TypeVar("Params", bound=BaseParams)


class EBMParams(BaseParams):
    def _validate_params(self, task: Task, params: Params) -> None:
        """ """
        keys_to_check = list(params.keys())

        if task == "regression":
            expected_parameter_keys = list(EBMC().get_params().keys())
        elif task == "classification":
            expected_parameter_keys = list(EBMR().get_params().keys())
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

    def __init__(self, task: Task, params: Params):
        """Initialize Parameter diictionary for Explainable Boosting Machine."""
        super().__init__(task, params)


class XGBParams(BaseParams):
    def _validate_params(self, task: Task, params: Params) -> None:
        """ """
        keys_to_check = list(params.keys())

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

    def __init__(self, task: Task, params: Params):
        """Initialize Parameter dictionary for Expreme Gradient Boosting."""
        super().__init__(task, params)


class GAMITParams(BaseParams):
    def _validate_params(self, task: Task, params: dict) -> None:
        pass

    def __init__(self, task: Task, params: dict):
        pass
