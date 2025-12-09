from ast import literal_eval
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.datasets import fetch_openml
from sklearn.model_selection import ParameterGrid

root = Path.cwd()
path_data = root / "data"


def download_data_openml(id: int, file_name: str) -> Path:
    destination = root / path_data
    if not Path.is_dir(destination):
        msg = f"{destination} is not an existing directory."
        raise NotADirectoryError(msg)

    data = fetch_openml(data_id=id, as_frame=True, parser="auto")
    frame = data.frame

    full_path = destination / Path(f"{file_name}.csv")
    frame.to_csv(full_path)
    return full_path


def pop_configuration_variables(conf: dict, prefix: str):
    """
    Assumes the dictionary keys to be popped are prefixed in some way to identify.

    Args:
        conf (dict):
        prefix (str):

    Returns:
        A tuple, the first element to reduced original configuration and the second element
        the prefix-fetched elements.
    """
    conf_popped = {k: conf.pop(k) for k in list(conf.keys()) if k.startswith(prefix)}
    conf_popped = {k.replace(prefix, "", 1): v for k, v in conf_popped.items()}
    return conf, conf_popped


def expand_param(param, size) -> np.ndarray:
    return np.full(size, param)


def make_equicorrelated_cov(corr, var, size) -> np.ndarray:
    cov = np.full((size, size), corr * var)
    np.fill_diagonal(cov, var)
    return cov


def get_partial_generator(conf: dict, expand=True) -> np.ndarray:
    default_generator = np.random.default_rng()

    if "distribution" not in conf:
        msg = "Expected key 'distribution' for numpy generator but did not find."
        raise ValueError(msg)

    if hasattr(default_generator, distribution_name := conf.pop("distribution")):
        generator = getattr(default_generator, distribution_name)
    else:
        msg = "Non valid distribution name provided. Expected numpy.random.Generator method name."
        raise ValueError(msg)

    _, generator_params = pop_configuration_variables(conf, "distribution_")

    if expand:
        size = generator_params.pop("size")
        generator_params["mean"] = expand_param(generator_params["mean"], size)
        generator_params["cov"] = make_equicorrelated_cov(
            corr=generator_params.pop("correlation"),
            var=generator_params.pop("variance"),
            size=size,
        )

    try:
        partial_generator = partial(
            generator, **generator_params
        )  # sample size not set here
    except ValueError:
        msg = f"Could not initialize generator with params {generator_params}"
        raise

    return partial_generator


def make_experiment_data(conf: dict):
    """
    Describe yaml interpretation here!!!
    """

    conf, conf_x1 = pop_configuration_variables(conf, "X1_")
    conf, conf_x2 = pop_configuration_variables(conf, "X2_")
    conf, conf_y = pop_configuration_variables(conf, "Y_")

    truncate_tuple = literal_eval(conf["truncation"])
    n_sample = conf["n_sample"]

    x1_generator = get_partial_generator(conf_x1, n_sample)
    data_x1 = x1_generator(size=n_sample)
    data_x1 = data_x1[
        :, : conf["n_clip"]
    ]  # maintain only the first 'n_clip' variables of X1
    data_x1 = np.clip(data_x1, min=truncate_tuple[0], max=truncate_tuple[1])

    colnames = [f"x_1_{idx}" for idx in range(1, data_x1.shape[1] + 1)]
    data_x1 = pd.DataFrame(data=data_x1, columns=colnames)

    x2_generator = get_partial_generator(conf_x2, n_sample)
    data_x2 = x2_generator(size=n_sample)
    data_x2 = np.clip(data_x2, min=truncate_tuple[0], max=truncate_tuple[1])

    colnames = [f"x_2_{idx}" for idx in range(1, data_x2.shape[1] + 1)]
    data_x2 = pd.DataFrame(data=data_x2, columns=colnames)

    y_generator = get_partial_generator(conf_y, expand=False)
    data_y = y_generator(size=n_sample)
    colnames = ["y"]
    data_y = pd.DataFrame(data=data_y, columns=colnames)

    if not data_x2.shape[0] == data_x1.shape[0] == data_y.shape[0]:
        msg = "Dimension misatch"
        raise ValueError(msg)

    return pd.concat([data_y, data_x1, data_x2], axis=1)


def make_experiment(conf):
    grid = list(ParameterGrid(conf))
    for idx, configuration in enumerate(grid):
        data = make_experiment_data(configuration)
        filename = f"experiment_{idx + 1}.csv"
        full_path = path_data / filename
        data.to_csv(full_path)


def make_data_from_conf(conf_data_folder: Path = root / "conf" / "datasets") -> None:
    for config_file in conf_data_folder.rglob("*.yaml"):
        with Path.open(config_file) as f:
            cfg = yaml.safe_load(f)
        if "openml_id" in cfg:
            id = cfg["openml_id"]
            file_name = config_file.stem
            download_data_openml(id, file_name)
        elif config_file.name == "experiment.yaml":
            make_experiment(cfg)


make_data_from_conf()
