import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

ROOT = Path.cwd()
DATA = ROOT / "data"
ASSET = ROOT / "assets" / "conf" / "data"
SIMULATION_CONF = ROOT / "conf" / "data" / "simulation.yaml"

DATA.mkdir(exist_ok=True, parents=True)
ASSET.mkdir(exist_ok=True, parents=True)

parser = argparse.ArgumentParser()
parser.add_argument("--filenameprefix", type=str, default="")
args, unknown = parser.parse_known_args()
filenameprefix = args.filenameprefix


def model1(X) -> np.ndarray:
    term1 = X[:, 0:5].sum(axis=1)

    term2 = 0.5 * np.sum(X[:, 5:8] ** 2, axis=1)

    idx = np.where(X[:, 8:10] > 0, X[:, 8:10], 0)
    term3 = idx.sum(axis=1)

    term4 = 0.2 * np.sum(X * X, axis=1)

    return term1 + term2 + term3 + term4


def model2(covariate): ...


def model3(covariates): ...


def model4(covariates): ...


def write_data(
    cfg, covariates, modelfunc, prefix=filenameprefix, folderpath=DATA
) -> None:
    y = modelfunc(covariates)
    filename_data = prefix + modelfunc.__name__ + ".csv"
    filename_asset = prefix + modelfunc.__name__ + ".yaml"

    cols = [f"X_{i + 1}" for i in range(covariates.shape[1])]
    data = pd.DataFrame(covariates, columns=cols)
    data["y"] = y
    data.to_csv(Path(folderpath, filename_data), index=False)
    write_configuration(cfg, filename_asset)


def write_configuration(config, filename, folderpath=ASSET):
    OmegaConf.save(config, Path(folderpath, filename))


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

write_data(cfg, covariates, model1)
