from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from gami_tree_reproduce.model.inducers import get_inducer_class
from gami_tree_reproduce.model.params import get_parameter_class
from gami_tree_reproduce.utils import config_to_grid, get_project_paths

project_paths = get_project_paths()


def get_inducer_dictionary_grid(inducers: list[Path]) -> dict:
    """
    Create configuration grids for all inducers for which there is a configuration file.

    Args:
        inducers (list[Path]): List of all paths to configurations.

    Returns:
        dict: Dictionary with inducer name as key and a dictionary of possible configurations as value.
    """
    total_grid = {}
    for inducer_yaml in inducers:
        inducer_name = inducer_yaml.stem
        inducer_cfg = OmegaConf.load(inducer_yaml)
        inducer_grid = config_to_grid(inducer_cfg.parameters)
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


datasets_paths = list(project_paths["data"].glob("*.pq"))
configs_inducer = list(Path(project_paths["conf_inducer"]).glob("*.yaml"))

inducers_dictionary_grid = get_inducer_dictionary_grid(configs_inducer)

for dataset_path in datasets_paths:
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
            hpo_params = {
                key: current_configuration.pop(key)
                for key, value in list(current_configuration.items())
                if value == "tune"
            }
            hpo_params = hpo_params.keys()

            params = get_parameter_class(inducer_name)(
                params=current_configuration, task=task
            )
            inducer = get_inducer_class(inducer_name)(taske=task, params_wrapper=params)

            # TODO: train hpo_params (set to default in params)
            data = pd.read_parquet(dataset_path)
            # TODO: train model on dataset
