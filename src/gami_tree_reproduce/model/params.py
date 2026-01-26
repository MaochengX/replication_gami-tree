"""
Each Parameter class inherits from BaseParams to provide a common API.
The initialization method is implemented in BaseParams which calls methods that need to be implemented in the individual
sub-classes since implementation dependent.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, TypeVar, get_args

from gaminet import GAMINetClassifier as GamiNetC
from gaminet import GAMINetRegressor as GamiNetR
from interpret.glassbox import ExplainableBoostingClassifier as EBMC
from interpret.glassbox import ExplainableBoostingRegressor as EBMR
from xgboost import XGBClassifier as XGBC
from xgboost import XGBRegressor as XGBR

Task = Literal["regression", "classification"]


@dataclass
class BaseParams(ABC):
    """Base class for all parameter types."""

    def __init__(
        self, task: Task, params: dict | None = None, hpo_keyword="tune"
    ) -> None:
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

        # set hpo configurations if applicable, fill method will overwrite HPO params with internal defaults
        self._hpo_settings = params.get(hpo_keyword, {})
        if self._hpo_settings == {}:
            self._hpo_pending = False
        else:
            self._hpo_pending = True

        self._validate_params(params.copy())
        default_params = self._get_default_params(params.copy())
        self._params = default_params

    @abstractmethod
    def _validate_params(self, params: dict): ...

    @abstractmethod
    def _get_default_params(self, params: dict): ...

    # ---------------------------------------------------------------------------------------
    #                                       API
    # ---------------------------------------------------------------------------------------
    def set_params(self, param_dict) -> None:
        self._validate_params(param_dict)
        self._params.update(param_dict)

    @property
    def hpo_settings(self) -> dict:
        return self._hpo_settings

    @property
    def params(self) -> dict:
        return self._params

    @property
    def hpo_pending(self) -> bool:
        return self._hpo_pending


# =======================================================================================
#                           Derived Parameter Classes
# =======================================================================================
Params = TypeVar("Params", bound=BaseParams)


class EBMParams(BaseParams):
    # ---------------------------------------------------------------------------------------
    #                                   Init methods
    # ---------------------------------------------------------------------------------------
    def _validate_params(self, params_user: Params, hpo_keyword: str = "tune") -> None:
        """
        Validate user given parameters with respect to internal implementation in package.

        Args:
            params_user (dict): Dictionary of parameter_name value pairs give by user. HPO params have special values.
            hpo_keyword (str, optional): Key name for HPO params in params_user. Defaults to "tune".
        """
        if self._task == "regression":
            expected_parameter_keys = list(EBMR().get_params().keys())
        elif self._task == "classification":
            expected_parameter_keys = list(EBMC().get_params().keys())
        compare_param_dict(params_user, expected_parameter_keys, hpo_keyword)

    def _get_default_params(self, params_user: dict, hpo_keyword="tune") -> None:
        """
        Fill parameter dictionary with package specific defaults if not given by user in params_user dictionary.
        HPO parameters are also overwritten with internal defaults.

        Args:
            params_user (dict): Dictionary of parameter_name value paris given by user. HPO params have special values
            hpo_keyword (str, optional): Key name for HPO params in params_user. Defaults to "tune".
        """

        if self._task == "classification":
            params_defaults = EBMC().get_params()
        elif self._task == "regression":
            params_defaults = EBMR().get_params()
        hpo_defaults = pop_hpo_with_default(params_user, params_defaults, hpo_keyword)
        # update internal default with hpo defaults or user defaults
        params_defaults.update(hpo_defaults)
        params_defaults.update(params_user)
        return params_defaults


class XGBParams(BaseParams):
    # ---------------------------------------------------------------------------------------
    #                                   Init methods
    # ---------------------------------------------------------------------------------------
    def _validate_params(self, params_user: Params, hpo_keyword: str = "tune") -> None:
        """
        Validate user given parameters with respect to internal implementation in package.
        For XGB internal defaults are None in python.

        Args:
            params_user (dict): Dictionary of parameter value pairs as give by user. HPO params have special value.
            hpo_keyword (str, optional): Key name for HPO params in params_user. Defaults to "tune".
        """
        if self._task == "regression":
            expected_parameter_keys = list(XGBR().get_params().keys())
        elif self._task == "classification":
            expected_parameter_keys = list(XGBC().get_params().keys())
        compare_param_dict(params_user, expected_parameter_keys, hpo_keyword)

    def _get_default_params(self, params_user: dict, hpo_keyword="tune") -> None:
        """
        Fill parameter dictionary with package specific defaults if not given by user in params_user dictionary.
        HPO parameters are also overwritten with internal defaults.

        Args:
            params_user (dict): Dictionary of parameter_name value paris given by user. HPO params have special values
            hpo_keyword (str, optional): Key name for HPO params in params_user. Defaults to "tune".
        """

        if self._task == "classification":
            params_defaults = XGBC().get_params()
        elif self._task == "regression":
            params_defaults = XGBR().get_params()
        hpo_defaults = pop_hpo_with_default(params_user, params_defaults, hpo_keyword)
        # update internal default with hpo defaults or user defaults
        params_defaults.update(hpo_defaults)
        params_defaults.update(params_user)
        return params_defaults


class GamiNetParams(BaseParams):
    # ---------------------------------------------------------------------------------------
    #                                   Init methods
    # ---------------------------------------------------------------------------------------
    def _validate_params(self, params_user: dict, hpo_keyword: str = "tune") -> None:
        """
        Validate user given parameters with respect to internal implementation in package.

        Args:
            params_user (dict): Dictionary of parameter value pairs as give by user. HPO params have special value.
            hpo_keyword (str, optional): Key name for HPO params in params_user. Defaults to "tune".
        """

        if self._task == "regression":
            expected_parameter_keys = list(GamiNetR().get_params().keys())
        elif self._task == "classification":
            expected_parameter_keys = list(GamiNetC().get_params().keys())
        compare_param_dict(params_user, expected_parameter_keys, hpo_keyword)

    def _get_default_params(self, params_user: dict, hpo_keyword: str = "tune") -> None:
        """
        Fill parameter dictionary with package specific defaults if not given by user in params_user dictionary.
        HPO parameters are also overwritten with internal defaults.

        Args:
            params_user (dict): Dictionary of parameter_name value paris given by user. HPO params have special values
            hpo_keyword (str, optional): Key name for HPO params in params_user. Defaults to "tune".
        """

        if self._task == "classification":
            params_defaults = EBMC().get_params()
        elif self._task == "regression":
            params_defaults = EBMR().get_params()
        hpo_defaults = pop_hpo_with_default(params_user, params_defaults, hpo_keyword)
        # update internal default with hpo defaults or user defaults
        params_defaults.update(hpo_defaults)
        params_defaults.update(params_user)
        return params_defaults


# ---------------------------------------------------------------------------------------
#                                   Helper functions
# ---------------------------------------------------------------------------------------


def compare_param_dict(params: dict, params_default: list, hpo_keyword="tune") -> None:
    params_user = params.copy()
    hpo_params_user = params_user.pop(hpo_keyword, {})
    non_hpo_params_user = params_user  # pop inplace

    wrong_non_hpo_keys = [key for key in hpo_params_user if key not in params_default]
    wrong_hpo_keys = [key for key in non_hpo_params_user if key not in params_default]

    wrong_keys = wrong_hpo_keys + wrong_non_hpo_keys
    if len(wrong_keys) >= 1:
        msg = f"Unexpected parameter keywords: {wrong_keys}. Expected to be in {params_default}"
        raise KeyError(msg)


def pop_hpo_with_default(
    params_user: dict, params_default: list, hpo_keyword="tune"
) -> dict:
    hpo_params = params_user.pop(hpo_keyword, {})
    return {key: value for key, value in params_default.items() if key in hpo_params}


PARAMS_REGISTRY = {"ebm": EBMParams, "xgb": XGBParams, "gaminet": GamiNetParams}


def get_parameter_class(name: str) -> callable:
    key = name.lower()
    if key not in PARAMS_REGISTRY:
        msg = f"No Parameter Class defined for given keyword '{key}'"
        raise KeyError(msg)
    return PARAMS_REGISTRY[name]
