from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import ParameterGrid

from gami_tree_reproduce.inducers import get_inducer
from gami_tree_reproduce.params import get_parameter

ROOT = Path.cwd()
PATH_CONF = ROOT / "conf"
PATH_DATA = ROOT / "data"


def train_all(cfg: dict) -> None:
    """Train all models.

    For each model and each dataset train the model on the dataset and document
    results in assets folder. Configurations for each model are expected to be in a
    configuration file next to `main.yaml`.

    Args:
        cfg (dict): confinguration file for main script
    """
    main_grid = ParameterGrid(cfg)

    for _, main_conf in enumerate(main_grid):
        name_dataset = main_conf["data"]
        name_inducer = main_conf["inducer"]
        name_task = main_conf["task"]

        with Path.open(
            PATH_CONF / "data" / Path(name_dataset).with_suffix(".yaml")
        ) as f:
            conf_data = yaml.safe_load(f)
        target = conf_data["target"]

        data = pd.read_csv(PATH_DATA / Path(name_dataset).with_suffix(".csv"))
        y = data[target]
        X = data.drop(columns=[target])

        with Path.open(
            PATH_CONF / "inducer" / Path(name_inducer).with_suffix(".yaml")
        ) as f:
            param_grid = yaml.safe_load(f)
            param_grid = ParameterGrid(param_grid)

        for _, param_conf in enumerate(param_grid):
            params = get_parameter(name_inducer)(name_task, param_conf)
            inducer = get_inducer(name_inducer)(name_task, params)
            inducer.train(X, y)


if __name__ == "__main__":
    with Path.open(PATH_CONF / "main.yaml") as f:
        cfg = yaml.safe_load(f)
    train_all(cfg)
