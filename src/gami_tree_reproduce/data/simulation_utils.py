from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf

from gami_tree_reproduce.data.simulation_models import (
    model1,
    model2,
    model3,
    model4,
    set_y_c,
    set_y_r,
)


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


def yaml_to_omegaconf(cfg_path: Path) -> OmegaConf:
    return OmegaConf.load(cfg_path)


def dict_to_omegaconf(data_dict: dict) -> OmegaConf:
    return OmegaConf.create(data_dict)


def get_generator_params(cfg: DictConfig) -> list[dict]:
    cfg_resolved = OmegaConf.to_container(cfg, resolve=True)

    x1_params = cfg_resolved["x1_generator"]["params"]
    x2_params = cfg_resolved["x2_generator"]["params"]
    y_c_params = cfg_resolved["y_generator_c"]["params"]
    y_r_params = cfg_resolved["y_generator_r"]["params"]

    return x1_params, x2_params, y_c_params, y_r_params


def get_generators(cfg: DictConfig) -> list:
    cfg_resolved = OmegaConf.to_container(cfg, resolve=True)

    rng = np.random.default_rng(cfg_resolved["SEED"])
    x1_generator = getattr(rng, cfg_resolved["x1_generator"]["name"])
    x2_generator = getattr(rng, cfg_resolved["x2_generator"]["name"])
    y_generator_c = getattr(rng, cfg_resolved["y_generator_c"]["name"])
    y_generator_r = getattr(rng, cfg_resolved["y_generator_r"]["name"])

    return x1_generator, x2_generator, y_generator_c, y_generator_r


def sample_data_models(cfg: DictConfig):
    x1_generator, x2_generator, y_generator_c, y_generator_r = get_generators(cfg)
    x1_params, x2_params, y_c_params, y_r_params = get_generator_params(cfg)

    X1 = x1_generator(**x1_params)
    X2 = x2_generator(**x2_params)

    covariates = np.hstack([X1[:, :10], X2])
    covariates = np.clip(covariates, -2.5, 2.5)

    covariates = np.hstack([X1[:, :10], X2])
    covariates = np.clip(covariates, -2.5, 2.5)

    model1c = set_y_c(covariates, model1, y_generator_c, y_c_params)
    model1r = set_y_r(covariates, model1, y_generator_r, y_r_params)

    model2c = set_y_c(covariates, model2, y_generator_c, y_c_params)
    model2r = set_y_r(covariates, model2, y_generator_r, y_r_params)

    model3c = set_y_c(covariates, model3, y_generator_c, y_c_params)
    model3r = set_y_r(covariates, model3, y_generator_r, y_r_params)

    model4c = set_y_c(covariates, model4, y_generator_c, y_c_params)
    model4r = set_y_r(covariates, model4, y_generator_r, y_r_params)

    return (model1c, model2c, model3c, model4c), (model1r, model2r, model3r, model4r)
