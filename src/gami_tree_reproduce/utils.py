import os
from itertools import product
from pathlib import Path

import pandas as pd
import yaml
from omegaconf import DictConfig, OmegaConf
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import MinMaxScaler

project_root = Path(__file__).resolve().parents[2]
os.environ.setdefault("PROJECT_ROOT", str(project_root))


def get_project_paths(
    path_to_config: Path = Path(project_root / "conf" / "config.yaml"),
    create: bool = True,
) -> dict:
    config = OmegaConf.load(path_to_config)
    OmegaConf.resolve(config)

    path_folders = config.paths.folder
    path_files = config.paths.file
    paths = {key: Path(path_str) for key, path_str in path_folders.items()}

    if create:
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)

    path_files = {key: Path(path_str) for key, path_str in path_files.items()}

    paths.update(path_files)

    return paths


def config_to_grid(cfg_params: OmegaConf, hpo_keyword: str = "tune") -> list[dict]:
    """
    Create a list of all possible parameter configurations.

    Args:
        cfg_params (OmegaConf): Configuration from yaml read into OmegaConf.
        no_combo (str, optional): Value for keys that are excluded in calculation of all grid-combinations. Defaults to "tune".

    Returns:
        list[dict]: A list of dictionaries, where each dictionary corresponds to a certain configuration of parameters.

    Example:
        > param_config_to_grid({"alpha": [1,2,3], "beta": [1, 2], "gamma": "tune", "delata": 1})
        > [{"alpha": 1, "beta": 1, "gamma": "tune", "delata": 1}, {"alpha": 2, "beta": 1, "gamma": "tune", "delata": 1},
             {"alpha": 3, "beta": 1, "gamma": "tune", "delata": 1}, {"alpha": 1, "beta": 2, "gamma": "tune", "delata": 1},
             {"alpha": 2, "beta": 2, "gamma": "tune", "delata": 1}, {"alpha": 3, "beta": 2, "gamma": "tune", "delata": 1}]
    """
    if isinstance(cfg_params, DictConfig):
        cfg_params = OmegaConf.to_container(cfg_params)

    # tunable hyperparameters are just added in the end to each combination
    hpo_dict = {}
    if hpo_keyword in cfg_params:
        hpo_dict = {hpo_keyword: cfg_params[hpo_keyword]}
        cfg_params.pop("tune")

    param_grid = {
        k: [v] if not isinstance(v, list) else v for k, v in cfg_params.items()
    }
    param_grid = [
        dict(zip(param_grid.keys(), combination, strict=True))
        for combination in product(*param_grid.values())
    ]

    return [{**d, **hpo_dict} for d in param_grid]


def add_list_to_grid(key_name: str, value_list: list, grid: list[dict]) -> list[dict]:
    """
    Expand Cartesian Product Dictionary with  new single-key many-value combinations.

    Args:
        key_name (str): key name of dictionary entry to be added
        value_list (list): values of dictionary entries to be added
        grid (list[dict]): the original grid that should be expanded

    Returns:
        list[dict]: a new list of dictionaries taking the cartesian product over value_list

    Example:
        > add_list_to_key("data", ["s1", "s2"], [{"max_depth": 3, "size":100}, {"max_depth":3, "size":200"}])
        > [{"max_depth": 3, "size":100, "data":"s1"},
            {"max_depth": 3, "size":100, "data":"s2"},
            {"max_depth":3, "size":200", "data": "s1"},
            {"max_depth":3, "size":200", "data": "s2}]
    """
    return [{**config, f"{key_name}": value} for config in grid for value in value_list]


project_paths = get_project_paths()
with Path("conf", "config.yaml").open("r") as f:
    conf = yaml.safe_load(f)


def make_metainfo(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """
    Create dictionary with information on covariates.
    Information includes the data type and the scaler function used to normalize input data to [0,1]

    Args:
        X_train (pd.DataFrame): Covariates for training
        y_train (pd.Series): Labels for training

    Returns:
        dict: Dictionary that maps each variable to its type (str) and scaler function
    """

    def dtype_to_gaminetformat(dtype):
        return (
            {"type": "continous"}
            if is_numeric_dtype(dtype)
            else {"type": "categorical"}
        )

    meta_info = {
        col: dtype_to_gaminetformat(dtype)
        for col, dtype in dict(X_train.dtypes).items()
    }
    meta_info.update({"y": {"type": "target"}})
    all_0_1_train = y_train.isin([0, 1]).all().all()

    for _, (key, item) in enumerate(meta_info.items()):
        if item["type"] == "target":
            if not all_0_1_train:
                sy = MinMaxScaler((0, 1))
                y_train = sy.fit_transform(y_train.to_frame())
            else:
                sy = None
            meta_info[key]["scaler"] = sy

        else:
            sx = MinMaxScaler((0, 1))
            x_train = sx.fit_transform(X_train[[key]])
            X_train[[key]] = x_train
            meta_info[key]["scaler"] = sx
    return meta_info


def get_seed() -> int:
    return conf["seed"]
