import argparse
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from gami_tree_reproduce.data.utils import (
    model1,
    model2,
    model3,
    model4,
    set_y,
)

ROOT = Path.cwd()
DATA = ROOT / "data"
SIMULATION_CONF = ROOT / "src" / "gami_tree_reproduce" / "data" / "simulation.yaml"

DATA.mkdir(exist_ok=True, parents=True)

parser = argparse.ArgumentParser()
parser.add_argument("--filenameprefix", type=str, default="")
args, unknown = parser.parse_known_args()
filenameprefix = args.filenameprefix


def numpy_resolver(func_name, *args):
    fn = getattr(np, func_name)
    return fn(*[int(a) if str(a).isdigit() else a for a in args])


OmegaConf.register_new_resolver("np", numpy_resolver)


def equicov(k, corr, var):
    cov = np.full((k, k), corr)
    np.fill_diagonal(cov, var)
    return cov


OmegaConf.register_new_resolver(
    "equicov", lambda k, var, cor: equicov(int(k), float(var), float(cor))
)


cfg = OmegaConf.load(SIMULATION_CONF)


# merge command line overrides: key=value pairs
cli_cfg = OmegaConf.from_dotlist(sys.argv[1:])
cfg = OmegaConf.merge(cfg, cli_cfg)

rng = np.random.default_rng(cfg.SEED)


generator1 = getattr(rng, cfg.x1_generator.name)
X1 = generator1(**cfg.x1_generator.params)

generator2 = getattr(rng, cfg.x2_generator.name)
X2 = generator2(**cfg.x2_generator.params)

covariates = np.hstack([X1[:, :10], X2])


y_generator = cfg.y_generator.name
y_generator_params = cfg.y_generator.params

data_model1_r = set_y(
    covariates, "regression", model1, y_generator, y_generator_params, rng
)
data_model1_c = set_y(
    covariates, "classification", model1, y_generator, y_generator_params, rng
)

data_model2_r = set_y(
    covariates, "regression", model2, y_generator, y_generator_params, rng
)
data_model2_c = set_y(
    covariates, "classification", model2, y_generator, y_generator_params, rng
)

data_model3_r = set_y(
    covariates, "regression", model3, y_generator, y_generator_params, rng
)
data_model3_c = set_y(
    covariates, "classification", model3, y_generator, y_generator_params, rng
)

data_model4_r = set_y(
    covariates, "regression", model4, y_generator, y_generator_params, rng
)
data_model4_c = set_y(
    covariates, "classification", model4, y_generator, y_generator_params, rng
)

def out_name(stem: str) -> str:
    return f"{filenameprefix}_{stem}.pq" if filenameprefix else f"{stem}.pq"


for stem, df in (
    ("model1c", data_model1_c),
    ("model1r", data_model1_r),
    ("model2c", data_model2_c),
    ("model2r", data_model2_r),
    ("model3c", data_model3_c),
    ("model3r", data_model3_r),
    ("model4c", data_model4_c),
    ("model4r", data_model4_r),
):
    fname = out_name(stem)
    df.to_parquet(DATA / fname)

