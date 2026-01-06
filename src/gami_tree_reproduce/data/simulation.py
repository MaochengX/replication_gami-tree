from pathlib import Path

import yaml

from gami_tree_reproduce.data.simulation_utils import (
    dict_to_omegaconf,
    sample_data_models,
    yaml_to_omegaconf,
)
from gami_tree_reproduce.utils import (
    ASSETS_SIM_CONF,
    CONF_SIM_YAML,
    DATA,
    config_to_grid,
)

cfg = yaml_to_omegaconf(CONF_SIM_YAML)

config_grid = config_to_grid(cfg)
for experiment_id, data_dict in enumerate(config_grid):
    cfg = dict_to_omegaconf(data_dict)

    model1r, model1c, model2r, model2c, model3r, model3c, model4r, model4c = (
        sample_data_models(cfg)
    )

    basename = f"sim{experiment_id + 1}_mod"

    # Save metadata
    with Path.open(
        ASSETS_SIM_CONF / Path(f"sim_{experiment_id + 1}").with_suffix(".yaml"), "w"
    ) as metafile:
        yaml.dump(data_dict, metafile)

    # Model 1
    model1r.to_parquet(DATA / Path(basename + "1r.pq"))

    model1c.to_parquet(DATA / Path(basename + "1c.pq"))

    # Model 2
    model2r.to_parquet(DATA / Path(basename + "2r.pq"))

    model2c.to_parquet(DATA / Path(basename + "2c.pq"))

    # Model 3
    model3r.to_parquet(DATA / Path(basename + "3r.pq"))

    model3c.to_parquet(DATA / Path(basename + "3c.pq"))

    # Model 4
    model4r.to_parquet(DATA / Path(basename + "4r.pq"))

    model4c.to_parquet(DATA / Path(basename + "4c.pq"))
