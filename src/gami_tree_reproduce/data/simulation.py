import numpy as np
from omegaconf import OmegaConf

from gami_tree_reproduce.data.simulation_models import (
    model1,
    model2,
    model3,
    model4,
    set_y,
)
from gami_tree_reproduce.utils import (
    ASSET,
    CONF_SIM,
    DATA,
    config_to_grid,
    save_hdf5,
)

DATA.mkdir(exist_ok=True, parents=True)
ASSET.mkdir(exist_ok=True, parents=True)


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


cfg = OmegaConf.load(CONF_SIM)

config_grid = config_to_grid(cfg)
for experiment_id, data_dict in enumerate(config_grid):
    cfg = OmegaConf.create(data_dict)
    rng = np.random.default_rng(cfg.SEED)

    generator1 = getattr(rng, cfg.x1_generator.name)
    X1 = generator1(**cfg.x1_generator.params)
    generator2 = getattr(rng, cfg.x2_generator.name)
    X2 = generator2(**cfg.x2_generator.params)

    covariates = np.hstack([X1[:, :10], X2])
    covariates = np.clip(covariates, -2.5, 2.5)

    y_generator = cfg.y_generator.name
    y_generator_params = cfg.y_generator.params

    basename = f"sim{experiment_id + 1}_mod"

    # Model 1
    data_model1_r = set_y(
        covariates, "regression", model1, y_generator, y_generator_params, rng
    )
    save_hdf5(basename + "1r", DATA, data_model1_r, data_dict)

    data_model1_c = set_y(
        covariates, "classification", model1, y_generator, y_generator_params, rng
    )
    save_hdf5(basename + "1c", DATA, data_model1_c, data_dict)

    # Model 2
    data_model2_r = set_y(
        covariates, "regression", model2, y_generator, y_generator_params, rng
    )
    save_hdf5(basename + "2r", DATA, data_model2_r, data_dict)

    data_model2_c = set_y(
        covariates, "classification", model2, y_generator, y_generator_params, rng
    )
    save_hdf5(basename + "2c", DATA, data_model2_c, data_dict)

    # Model 3
    data_model3_r = set_y(
        covariates, "regression", model3, y_generator, y_generator_params, rng
    )
    save_hdf5(basename + "3r", DATA, data_model3_r, data_dict)

    data_model3_c = set_y(
        covariates, "classification", model3, y_generator, y_generator_params, rng
    )
    save_hdf5(basename + "3c", DATA, data_model3_c, data_dict)

    # Model 4
    data_model4_r = set_y(
        covariates, "regression", model4, y_generator, y_generator_params, rng
    )
    save_hdf5(basename + "4r", DATA, data_model4_r, data_dict)

    data_model4_c = set_y(
        covariates, "classification", model4, y_generator, y_generator_params, rng
    )
    save_hdf5(basename + "4c", DATA, data_model4_c, data_dict)
