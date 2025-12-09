from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

ROOT = Path.cwd()


def get_model1(X) -> np.ndarray:
    """
    g(x_i) = \sum_{j=1}^5 x_{ij} +
             \sum_{j=6}^{8} 0.5 x_{ij}^2 +
             \sum_{j=9}^10 x_{ij}\mathbb{I}(x_{ij}>0) +
             \sum_{j=1}^10 \sum_{k=j+1}^10 0.2 x_{ij} x_{ik}
    """
    term1 = X[:, 0:5].sum(axis=1)

    term2 = 0.5 * np.sum(X[:, 5:8] ** 2, axis=1)

    idx = np.where(X[:, 8:10] > 0, X[:, 8:10], 0)
    term3 = idx.sum(axis=1)

    term4 = 0.2 * np.sum(X * X, axis=1)

    return term1 + term2 + term3 + term4


def get_model2(covariate): ...


def get_model3(covariates): ...


def get_model4(covariates): ...


def write_data(covariates, y, filename, folderpath=ROOT / "data") -> None:
    cols = [f"X_{i + 1}" for i in range(covariates.shape[1])]
    data = pd.DataFrame(covariates, columns=cols)
    data["y"] = y
    data.to_csv(Path(folderpath, filename), index=False)


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


cfg = OmegaConf.load(ROOT / "conf" / "data" / "simulation.yaml")
rng = np.random.default_rng(cfg.SEED)


generator1 = getattr(rng, cfg.x1_generator.name)
X1 = generator1(**cfg.x1_generator.params)

generator2 = getattr(rng, cfg.x2_generator.name)
X2 = generator2(**cfg.x2_generator.params)

covariates = np.hstack([X1[:, :10], X2])

y = get_model1(covariates)
write_data(covariates, y, "simulation_1.csv")
