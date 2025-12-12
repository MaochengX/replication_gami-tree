import argparse
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from gami_tree_reproduce.inducers import get_inducer
from gami_tree_reproduce.params import get_parameter

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


def parse_data(args):
    """
    The user can individually specify  datanames on which the models are trained.
    Check whether valid dataset chosen (exists in data folder)
    """


def parse_inducer(args):
    """
    The user can individually specify modelnames which is trained on the datasets.
    Check whether valid inducer chosen (accessable from inducer_registry via 'get_inducer')
    """


def gather_parameternames(cfg: DictConfig):
    if "parameters" not in cfg:
        raise KeyError

    params_fix = params_tune = []
    if "fix" in cfg.parameters:
        params_fix = list(cfg.parameters.fix.keys())
    if "tune" in cfg.parameters:
        params_tune = list(cfg.parameters.tune.keys())

    return params_fix + params_tune


task = "regression"
inducer = "xgb"
dataset = "sim4_model4"
path_conf_inducer = CONF / "inducer" / Path(inducer).with_suffix(".yaml")
cfg_inducer = OmegaConf.load(path_conf_inducer)


param_class = get_parameter(inducer)
param_config = param_class(task)
inducer_configuration = get_inducer(inducer)(task, param_config)
data = pd.read_parquet(DATA / Path(dataset).with_suffix(".pq"))
if {"X_1", "y"}.issubset(set(data.columns)):
    y = data.y
    X = data.drop(columns=["y"])
inducer_configuration.train(X, y)
