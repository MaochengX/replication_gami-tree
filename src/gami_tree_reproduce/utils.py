import json
from itertools import product
from pathlib import Path

import h5py
import pandas as pd
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
ASSET = ROOT / "assets" / "conf" / "data"
CONF_SIM = ROOT / "conf" / "data" / "simulation.yaml"


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


def save_hdf5(
    name: str, destination_folder: Path, data: pd.DataFrame, metadata: dict
) -> None:
    fullname = Path(destination_folder, Path(name).with_suffix(".hdf5"))
    with h5py.File(fullname, "w") as f:
        dset = f.create_dataset(data=data, name=name)
        dset.attrs["metadata"] = json.dumps(metadata)
