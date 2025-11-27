from dataclasses import asdict, dataclass
from typing import Any, ClassVar, Literal, Protocol, TypeVar

from interpret.glassbox import ExplainableBoostingClassifier as EBMC
from interpret.glassbox import ExplainableBoostingRegressor as EBMR
from xgboost import XGBClassifier as XGBC
from xgboost import XGBRegressor as XGBR


# forward declaration, later to be imported
class GAMITR: ...  # for regression task


class GAMITC: ...  # for classification task


@dataclass
class EBMParams:
    """ """


@dataclass
class XGBParams:
    """ """


@dataclass
class GAMITParams:
    """
    GAMIT = GAMI-Tree
    """


class Classifier(Protocol):
    """
    Protocol for Classifier variables.
    """

    def train(self):
        pass

    def predict_proba(self):
        pass


class Regressor(Protocol):
    """
    Protocol for Regressor variables.
    """

    def train(self):
        pass

    def predict(self):
        pass


Params = TypeVar("Params", bound=EBMParams | XGBParams | GAMITParams)
Model = Classifier | Regressor


class InducerRegistryBase(type):
    """
    Metaclass for registration of Inducer classes.
    """

    inducer_registry: ClassVar[dict[str | type["BaseInducerClass"]]] = {}

    def __new__(cls, name, bases, attrs):
        """
        Register a new Inducer at definition time here.
        """
        new_cls = super().__new__(cls, name, bases, attrs)
        if name != "BaseInducerClass":
            cls.inducer_registry[new_cls.__name__] = new_cls
        return new_cls

    @classmethod
    def get_registry(cls) -> dict:
        """Return (copy of) inducer registry."""
        return dict(cls.inducer_registry)


class BaseInducerClass(metaclass=InducerRegistryBase):
    """ """

    def __init__(self, params: Params):
        """ """
        self.params = params
        self._model: Model | None = None

    def train(self, X: Any, Y: Any) -> Model:
        """ """
        raise NotImplementedError

    def instantiate(self, task: Literal["classification", "regression"]) -> Model:
        """
        Instantiae untrained model.
        """
        is_classification = task == "classification"
        model_class = (
            self.classifier_class if is_classification else self.regressor_class
        )

        if model_class is None:
            msg = f"{self.__class__.__name__} does not have a classifier or regressor"
            raise ValueError(msg)

        params_dict = (
            asdict(self.params) if hasattr(self.params, "__dataclass_fields__") else {}
        )
        return model_class(**params_dict)


class EBMinducer(BaseInducerClass):
    classifier_class = EBMC
    regressor_class = EBMR


class XGBinducer(BaseInducerClass):
    classifier_class = XGBC
    regressor_class = XGBR


class GAMITinducer(BaseInducerClass):
    classifier_class = GAMITC
    regressor_class = GAMITR
