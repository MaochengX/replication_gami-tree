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


def model2(X) -> np.ndarray:
    # term1: sum_{j=1..5} x_j
    term1 = X[:, 0:5].sum(axis=1)

    # term2: sum_{j=6..8} 0.5 * x_j^2
    term2 = 0.5 * np.sum(X[:, 5:8] ** 2, axis=1)

    # term3: sum_{j=9..10} x_j * 1(x_j > 0)
    term3 = np.sum(np.where(X[:, 8:10] > 0, X[:, 8:10], 0), axis=1)

    # term4: 0.25 * x1*x2
    term4 = 0.25 * X[:, 0] * X[:, 1]

    # term5: 0.25 * x1 * x3^2
    term5 = 0.25 * X[:, 0] * (X[:, 2] ** 2)

    # term6: 0.25 * x4^2 * x5^2
    term6 = 0.25 * (X[:, 3] ** 2) * (X[:, 4] ** 2)

    # term7: exp(x4 * x6 / 3)
    term7 = np.exp(X[:, 3] * X[:, 5] / 3)

    # term8: x5*x6 * 1(x5>0) * 1(x6>0)
    term8 = X[:, 4] * X[:, 5] * (X[:, 4] > 0) * (X[:, 5] > 0)

    # term9: clip(x7 + x8, -1, 0)
    term9 = np.clip(X[:, 6] + X[:, 7], -1, 0)

    # term10: clip(x7 * x9, -1, 1)
    term10 = np.clip(X[:, 6] * X[:, 8], -1, 1)

    # term11: 1(x8>0) * 1(x9>0)
    term11 = (X[:, 7] > 0) * (X[:, 8] > 0)

    return (
        term1
        + term2
        + term3
        + term4
        + term5
        + term6
        + term7
        + term8
        + term9
        + term10
        + term11
    )


def model3(X) -> np.ndarray:
    # term1: sum_{j=1..5} x_j
    term1 = X[:, 0:5].sum(axis=1)

    # term2: sum_{j=6..8} 0.5 * x_j^2
    term2 = 0.5 * np.sum(X[:, 5:8] ** 2, axis=1)

    # term3: sum_{j=9..10} x_j * 1(x_j > 0)
    term3 = np.sum(np.where(X[:, 8:10] > 0, X[:, 8:10], 0), axis=1)

    # term4: 0.25 * x1^2 * x2^2
    term4 = 0.25 * (X[:, 0] ** 2) * (X[:, 1] ** 2)

    # term5: 2 * (x3 - 0.5)_+
    term5 = 2 * np.maximum(X[:, 2] - 0.5, 0)

    # term6: (x4 - 0.5)_+
    term6 = np.maximum(X[:, 3] - 0.5, 0)

    # term7: 0.5 * sin(pi*x5) * sin(pi*x6)
    term7 = 0.5 * np.sin(np.pi * X[:, 4]) * np.sin(np.pi * X[:, 5])

    # term8: 0.5 * sin(pi*(x7 + x8))
    term8 = 0.5 * np.sin(np.pi * (X[:, 6] + X[:, 7]))

    return term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8


def model4(X) -> np.ndarray:
    # term1: sum_{j=1..5} x_j
    term1 = X[:, 0:5].sum(axis=1)

    # term2: sum_{j=6..8} 0.5 * x_j**2
    term2 = 0.5 * np.sum(X[:, 5:8] ** 2, axis=1)

    # term3: sum_{j=9..10} x_j * 1(x_j > 0)
    term3 = np.sum(np.where(X[:, 8:10] > 0, X[:, 8:10], 0), axis=1)

    # shorthand variables for readability
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    x4, x5, x6 = X[:, 3], X[:, 4], X[:, 5]

    # term4: interactions x1x2 + x1x3 + x2x3
    term4 = x1 * x2 + x1 * x3 + x2 * x3

    # term5: 0.5 * x1*x2*x3
    term5 = 0.5 * x1 * x2 * x3

    # term6: x4x5 + x4x6 + x5x6
    term6 = x4 * x5 + x4 * x6 + x5 * x6

    # term7: 0.5 * 1(x4>0) * x5 * x6
    term7 = 0.5 * (x4 > 0) * x5 * x6

    return term1 + term2 + term3 + term4 + term5 + term6 + term7


def write_data(covariates, modelfunc, prefix=filenameprefix, folderpath=DATA) -> None:
    # TODO: add noise to y
    y = modelfunc(covariates)
    filename_data = prefix + "_" + modelfunc.__name__ + ".csv"

    cols = [f"X_{i + 1}" for i in range(covariates.shape[1])]
    data = pd.DataFrame(covariates, columns=cols)
    data["y"] = y
    data.to_csv(Path(folderpath, filename_data), index=False)


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

write_data(covariates, model1)
write_data(covariates, model2)
write_data(covariates, model3)
write_data(covariates, model4)
write_configuration(
    cfg, filename=Path(ASSET, Path(filenameprefix).with_suffix(".yaml"))
)
