from abc import abstractmethod
from collections.abc import Iterable
from dataclasses import asdict
from typing import (
    Any,
    ClassVar,
    Literal,
    Protocol,
    override,
    runtime_checkable,
)

import numpy as np
from interpret.glassbox import ExplainableBoostingClassifier as EBMC
from interpret.glassbox import ExplainableBoostingRegressor as EBMR
from xgboost import XGBClassifier as XGBC
from xgboost import XGBRegressor as XGBR

from src.gami_tree_reproduce.params import EBMParams, GAMITParams, Params, XGBParams


# TODO: Forward declarations for now until implemented
class GAMITR: ...


class GAMITC: ...


# Protocol Definitions


@runtime_checkable
class Classifier(Protocol):
    def train(self, X, y) -> None: ...
    def predict_proba(self, X) -> np.ndarray | Iterable[float]: ...


@runtime_checkable
class Regressor(Protocol):
    def train(self, X, y) -> None: ...
    def predict(self, X) -> np.ndarray | Iterable[float]: ...


Model = Classifier | Regressor


####################
# Registry Metaclass
####################
class InducerRegistryBase(type):
    """
    Metaclass automatically registering Inducer subclasses.
    """

    inducer_registry: ClassVar[dict[str, type["BaseInducerClass"]]] = {}

    @override
    def __new__(cls, name, bases, attrs):
        new_cls = super().__new__(cls, name, bases, attrs)
        if name != "BaseInducerClass":
            cls.inducer_registry[name] = new_cls
        return new_cls

    @classmethod
    def get_registry(cls) -> dict[str, type["BaseInducerClass"]]:
        return dict(cls.inducer_registry)


##############
# Base Inducer
##############
class BaseInducerClass(metaclass=InducerRegistryBase):
    """
    Base class for model inducers.
    """

    @property
    @abstractmethod
    def classifier_class(self) -> type[Classifier]: ...

    @property
    @abstractmethod
    def regressor_class(self) -> type[Regressor]: ...

    def instantiate(
        self,
        task: Literal["classification", "regression"],
        params: Params | dict | None = None,
    ) -> Model:
        model_class = (
            self.classifier_class if task == "classification" else self.regressor_class
        )

        # Convert dataclass parameters to dict
        if params is None:
            params_dict = {}
        elif hasattr(params, "__dataclass_fields__"):
            params_dict = asdict(params)
        else:
            params_dict = dict(params)

        self._params = params_dict

        # Return a NEW INSTANCE EVERY TIME
        return model_class(**params_dict)

    def train(self, X: Any, y: Any, params: Params | dict) -> np.ndarray:
        """ " """
        if self._trained:
            msg = "Cannot retrain an already trained model."
            raise AssertionError(msg)

        self._trained = True
        self._params = params

        self._model.train(X, y)


###################
# Concrete Inducers
###################
class EBMinducer(BaseInducerClass):
    param_class = EBMParams

    @property
    def classifier_class(self) -> type[EBMC]:
        return EBMC

    @property
    def regressor_class(self) -> type[EBMR]:
        return EBMR


class XGBinducer(BaseInducerClass):
    param_class = XGBParams

    @property
    def classifier_class(self) -> type[XGBC]:
        return XGBC

    @property
    def regressor_class(self) -> type[XGBR]:
        return XGBR


class GAMITinducer(BaseInducerClass):
    param_class = GAMITParams

    @property
    def classifier_class(self) -> type[GAMITC]:
        return GAMITC

    @property
    def regressor_class(self) -> type[GAMITR]:
        return GAMITR


# Registry Access

inducer_registry = InducerRegistryBase.get_registry()
ebm_base = EBMinducer()
xgb_base = XGBinducer()
gamit_base = GAMITinducer()
