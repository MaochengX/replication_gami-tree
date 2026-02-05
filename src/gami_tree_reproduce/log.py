import time
from datetime import datetime
from pathlib import Path
from typing import Any

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


class LogMediator:
    def __init__(self):
        self._inducer = None
        self._time_train = None
        self._time_train_perf = None

        self._time_predict = None
        self._time_predict_perf = None

        self._loss_train = None
        self._loss_test = None

        self._time_hpo = None
        self._time_hpo_perf = None

    def train(
        self, inducer: BaseInducer, X_train: np.ndarray, y_train: np.ndarray
    ) -> None:
        self.is_numpy_format(X_train, y_train)
        self._inducer = inducer

        start_train = datetime.now()  # noqa: DTZ005
        start_train_perf = time.perf_counter()
        # ebm: single number ok
        loss_train = inducer.train(X_train, y_train)
        end_train_perf = time.perf_counter()
        end_train = datetime.now()  # noqa: DTZ005

        self._time_train = self.timedelta_to_minute(end_train - start_train)
        self._time_train_perf = end_train_perf - start_train_perf
        self._loss_train = loss_train

    def predict(
        self, inducer: BaseInducer, X_test: np.ndarray, y_test: np.ndarray
    ) -> None:
        self.is_numpy_format(X_test, y_test)

        start_predict = datetime.now()  # noqa: DTZ005
        start_predict_perf = time.perf_counter()
        # ebm: predictions ok
        y_hat = inducer.predict(X_test)
        end_predict_perf = time.perf_counter()
        end_predict = datetime.now()  # noqa: DTZ005

        self._time_predict = self.timedelta_to_minute(end_predict - start_predict)
        self._time_predict_perf = end_predict_perf - start_predict_perf

        if y_test is not None:
            loss = mean_squared_error if inducer.task == "regression" else log_loss
            loss_test = loss(y_hat, y_test)
            self._loss_test = loss_test

    def do_hpo(
        self, inducer: BaseInducer, X_val: np.ndarray, y_val: np.ndarray
    ) -> None:
        start_hpo = datetime.now()  # noqa: DTZ005
        start_hpo_perf = time.perf_counter()
        inducer.do_hpo(X_val, y_val)
        end_hpo_perf = time.perf_counter()
        end_hpo = datetime.now()  # noqa: DTZ005
        self._time_hpo = self.timedelta_to_minute(end_hpo - start_hpo)
        self._time_hpo_perf = end_hpo_perf - start_hpo_perf

    def log(self, destination_folder: Path, inducer: BaseInducer) -> None:
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
        total_config.update(
            {
                "time_train_perf": self._time_train_perf,
                "time_predict_perf": self._time_predict_perf,
            }
        )

        with Path(destination_folder, "results.yaml").open("w") as f:
            total_config = npnum_to_pynum(total_config)
            yaml.safe_dump(total_config, f)

        joblib.dump(inducer, Path(destination_folder, "model.gz"))

    def is_numpy_format(self, X_train: Any, y_train: Any | None = None) -> bool:
        if y_train is not None and not isinstance(y_train, np.ndarray):
            msg = f"Expected y_train to be np.ndarray, got {type(y_train)}"
            raise KeyError(msg)
        if not isinstance(X_train, np.ndarray):
            msg = f"Expected X_train to be np.ndarray, got {type(X_train)}"
            raise KeyError(msg)

    def timedelta_to_minute(self, timedelta) -> float:
        """
        _summary_

        Args:
            timedelta   (datetime.timedelta): Timedelta object consisting of days, seconds and microseconds
        """
        days_seconds = timedelta.days * 24 * 60 * 60
        seconds = timedelta.seconds
        microseconds_seconds = timedelta.microseconds * 1e-6
        return days_seconds + seconds + microseconds_seconds
