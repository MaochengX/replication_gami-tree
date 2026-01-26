from os import environ as oenv

import pandas as pd

oenv["MPLBACKEND"] = "Agg"
from pathlib import Path

from omegaconf import OmegaConf

from gami_tree_reproduce.data.preprocess_utils import (
    assert_preprocess_data_available,
    get_train_val_test,
)
from gami_tree_reproduce.log import ExperimentMediator
from gami_tree_reproduce.model.inducers import get_inducer_class
from gami_tree_reproduce.model.params import get_parameter_class
from gami_tree_reproduce.utils import (
    config_to_grid,
    get_project_paths,
)

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


preprocessed_folder = assert_preprocess_data_available()
assert_preprocess_data_available()

dataset_folders = [path for path in preprocessed_folder.glob("*") if path.is_dir()]
configs_inducer = list(Path(project_paths["conf_inducer"]).glob("*.yaml"))

inducers_dictionary_grid = get_inducer_dictionary_grid(configs_inducer)

for dataset_folder in dataset_folders:
    # dataset names are stores as parquet files and end in 'c' or 'r' for categorization/regression tasks
    dataset_name = dataset_folder.stem
    if dataset_name[-1] == "c":
        task = "classification"
    else:
        taks = "regression"

    for inducer_name in inducers_dictionary_grid:
        if inducer_name != "ebm":
            continue

        configurations = inducers_dictionary_grid[inducer_name]
        for experiment_count, current_configuration in enumerate(configurations):
            # location for results
            path_results = Path(
                project_paths["assets_conf_experiments"],
                dataset_name
                + "_"
                + inducer_name
                + "_experiment"
                + str(experiment_count + 1),
            )

            params = get_parameter_class(inducer_name)(
                params=current_configuration, task=task
            )

            inducer = get_inducer_class(inducer_name)(task=task, params_wrapper=params)
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_train_val_test(
                dataset_name
            )
            if inducer.hpo_pending():
                hpo_configs = inducer.do_hpo(X_val, y_val)
            else:
                X_train = pd.concat([X_train, X_val])
                y_train = pd.concat([y_train, y_val])

            log = ExperimentMediator()
            log.train(inducer, X_train, y_train)
            log.predict(inducer, X_test, y_test)
            log.log(path_results, inducer)
