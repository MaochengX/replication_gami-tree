import os
from itertools import product
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

project_root = Path(__file__).resolve().parents[2]
os.environ.setdefault("PROJECT_ROOT", str(project_root))


def get_project_paths(
    path_to_config: Path = Path(project_root / "conf" / "config.yaml"),
    create: bool = True,
):
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


def config_to_grid(cfg_params: OmegaConf, no_combo="tune") -> list[dict]:
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
    hyperparameter_dict = {}
    if no_combo in cfg_params:
        hyperparameter_dict = dict.fromkeys(cfg_params[no_combo], no_combo)
        cfg_params.pop("tune")
    param_grid = {
        k: [v] if not isinstance(v, list) else v for k, v in cfg_params.items()
    }  # convert to list (iterable) with possibly single element
    # create the actual grid ad add hyperaprameters
    param_grid = [
        dict(zip(param_grid.keys(), combination, strict=True))
        for combination in product(*param_grid.values())
    ]

    return [{**d, **hyperparameter_dict} for d in param_grid]
