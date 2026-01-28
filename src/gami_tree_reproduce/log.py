import time
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.metrics import log_loss, mean_squared_error

from gami_tree_reproduce.model.inducers import BaseInducer


def npnum_to_pynum(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: npnum_to_pynum(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [npnum_to_pynum(v) for v in obj]
    return obj


class ExperimentMediator:
    def __init__(self):
        self._inducer = None
        self._time_train = None
        self._time_predict = None
        self._loss_train = None
        self._loss_test = None

    def train(
        self, inducer: BaseInducer, X_train: np.ndarray, y_train: np.ndarray
    ) -> None:
        self._inducer = inducer

        start_train = time.perf_counter()
        # ebm: single number ok
        loss_train = inducer.train(X_train, y_train)
        end_train = time.perf_counter()

        self._time_train = end_train - start_train
        self._loss_train = loss_train

    def predict(self, inducer: BaseInducer, X_test, y_test) -> None:
        start_predict = time.perf_counter()
        # ebm: predictions ok
        y_hat = inducer.predict(X_test)
        end_predict = time.perf_counter()

        self._time_predict = end_predict - start_predict
        loss = mean_squared_error if inducer.task == "regression" else log_loss
        loss_test = loss(y_hat, y_test)
        self._loss_test = loss_test

    def log(self, destination_folder: Path, inducer) -> None:
        destination_folder.mkdir(exist_ok=True, parents=True)

        total_config = {}
        total_config.update(
            {"loss_train": self._loss_train, "loss_test": self._loss_test}
        )
        total_config.update({"hpo_settings": inducer.params_wrapper.hpo_settings})
        total_config.update({"params": inducer.params_wrapper.params})
        total_config.update(
            {"time_train": self._time_train, "time_predict": self._time_predict}
        )
        with Path(destination_folder, "results.yaml").open("w") as f:
            total_config = npnum_to_pynum(total_config)
            yaml.safe_dump(total_config, f)

        joblib.dump(inducer, Path(destination_folder, "model.gz"))
