import argparse
from itertools import product
from pathlib import Path

from omegaconf import OmegaConf


def get_parameter_grid(cfg_params: OmegaConf, section_name="fix") -> None:
    if section_name in cfg_params:
        params_experiment = cfg_params[section_name]
        parameter_experiment_grid = [
            dict(zip(params_experiment, value, strict=True))
            for value in product(*params_experiment.values())
        ]
    else:
        msg = f"Expected to find '{section_name}' section in config but found none."
        raise KeyError(msg)

    return parameter_experiment_grid


def get_filename_path(filename: str, dir_walk: list[Path]) -> Path:
    """
    _summary_

    Args:
        filename (str): _description_
        walk_list (list[Path]): _description_

    Returns:
        Path: _description_
    """
    return next(p for p in dir_walk if p.stem == filename)


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


ROOT = Path.cwd()
CONF = ROOT / "conf"
DATA = ROOT / "data"
ASSETS = ROOT / "assets"

parser = argparse.ArgumentParser()
parser.add_argument("--inducer", type=str, default="all")
parser.add_argument("--data", type=str, default="all")
parser.add_argument("--param_fix", type=str, default="")
parser.add_argument("--param_tune", type=str, default="")
args, unknown = parser.parse_known_args()

arg_data = args.data
arg_inducer = args.inducer


datasets = list(DATA.glob("*.pq"))
dataset_names = [p.stem for p in datasets]
inducer = "ebm"
path_conf_inducer = CONF / "inducer" / Path(inducer).with_suffix(".yaml")
cfg_inducer = OmegaConf.load(path_conf_inducer)
params = cfg_inducer.parameters
inducer_experiment_grid = get_parameter_grid(params)
# TODO: iterate over grid
