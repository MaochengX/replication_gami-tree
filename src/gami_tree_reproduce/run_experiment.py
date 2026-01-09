from os import environ as oenv
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split

from gami_tree_reproduce.model.inducers import get_inducer_class
from gami_tree_reproduce.model.params import get_parameter_class
from gami_tree_reproduce.utils import config_to_grid, get_project_paths

project_paths = get_project_paths()
config = Path(oenv["PROJECT_ROOT"])
config = OmegaConf.load(Path("conf/config.yaml"))
OmegaConf.resolve(config)
seed = config.seed


def get_inducer_dictionary_grid(inducers: list[Path]) -> dict:
    """
    Create configuration grids for all inducers for which there is a configuration file.

    It is assumed that the name of the yaml file matches the names in the inducer register so that the yaml file name
    can later be used to pick the desired inducer from the registry.

    Args:
        inducers (list[Path]): List of all paths to configurations.

    Returns:
        dict: Dictionary with inducer name as key and a dictionary of possible configurations as value.
    """
    total_grid = {}
    for inducer_yaml in inducers:
        inducer_name = inducer_yaml.stem
        inducer_cfg = OmegaConf.load(inducer_yaml)
        inducer_grid = config_to_grid(inducer_cfg)
        total_grid[inducer_name] = inducer_grid

    return total_grid


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


def make_train_val_test(
    source_file: Path = project_paths["data_raw"],
    destination_folder: Path = project_paths["data_preprocessed"],
    config: dict = config,
) -> None:
    data_name = source_file.stem

    test_size = config["test_size"]
    val_size = config["val_size"]
    data = pd.read_parquet(source_file)
    data_train_val, data_test = train_test_split(
        data, test_size=test_size, shuffle=True, random_state=seed
    )

    val_size = val_size / (1 - test_size)
    data_train, data_val = train_test_split(
        data_train_val, test_size=val_size, shuffle=True, random_state=seed
    )

    destination_subfolder = destination_folder / data_name
    Path(destination_subfolder).mkdir(exist_ok=True, parents=True)

    data_train.to_parquet(
        Path(destination_subfolder / Path(data_name + "_train")).with_suffix(".pq")
    )
    data_test.to_parquet(
        Path(destination_subfolder / Path(data_name + "_test")).with_suffix(".pq")
    )
    data_val.to_parquet(
        Path(destination_subfolder / Path(data_name + "_val")).with_suffix(".pq")
    )


def get_train_val_test(
    subfolder: str | Path | None = None,
    source: Path = project_paths["data_preprocessed"],
    tuples=True,
):
    if isinstance(subfolder, str):  # only name provided and default path assumed
        assert Path(source / subfolder).exists()
        source = source / subfolder
    else:  # explicitly folder path given
        assert source.exists()

    train_file = source / f"{subfolder}_train.pq"
    test_file = source / f"{subfolder}_test.pq"
    val_file = source / f"{subfolder}_val.pq"

    assert train_file.exists()
    assert test_file.exists()
    assert val_file.exists()

    data_train = pd.read_parquet(train_file)
    data_val = pd.read_parquet(val_file)
    data_test = pd.read_parquet(test_file)

    if not tuples:
        return data_train, data_val, data_test

    X_train, y_train = data_train.drop(columns="y"), data_train["y"]
    X_val, y_val = data_val.drop(columns="y"), data_val["y"]
    X_test, y_test = data_test.drop(columns="y"), data_test["y"]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


dataset_paths = list(project_paths["data_raw"].glob("*.pq"))
configs_inducer = list(Path(project_paths["conf_inducer"]).glob("*.yaml"))

inducers_dictionary_grid = get_inducer_dictionary_grid(configs_inducer)

for raw_dataset in dataset_paths:
    make_train_val_test(raw_dataset)

for dataset_path in dataset_paths:
    # dataset names are stores as parquet files and end in 'c' or 'r' for categorization/regression tasks
    experiment_name = dataset_path.stem
    if experiment_name[-1] == "c":
        task = "classification"
    else:
        taks = "regression"

    # get class and instantiate parameter and inducer objects
    for inducer_name in inducers_dictionary_grid:
        configurations = inducers_dictionary_grid[inducer_name]
        for current_configuration in configurations:
            # hpo params need to be tuned and all other parameters are either default or set in config
            hpo_params = {
                key: current_configuration.pop(key)  # need to do hpo before params
                for key, value in list(current_configuration.items())
                if value == "tune"
            }
            hpo_params = hpo_params.keys()

            params = get_parameter_class(inducer_name)(
                params=current_configuration, task=task
            )

            inducer = get_inducer_class(inducer_name)(task=task, params_wrapper=params)
            # data = pd.read_parquet(dataset_path)
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_train_val_test()
            inducer.estimate_hpo_params(X_val, y_val)
            inducer.fit()
            # TODO: train model on dataset
