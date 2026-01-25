from pathlib import Path

import yaml

from gami_tree_reproduce.data.simulation_utils import (
    dict_to_omegaconf,
    sample_data_models,
    yaml_to_omegaconf,
)
from gami_tree_reproduce.utils import (
    config_to_grid,
    get_project_paths,
)

project_paths = get_project_paths()

cfg = yaml_to_omegaconf(project_paths["conf_data_simulation"])
data_path = project_paths["data_raw"]
sim_conf_path = project_paths["assets_conf_data"]

config_grid = config_to_grid(cfg)
for simulation_id, data_dict in enumerate(config_grid):
    cfg = dict_to_omegaconf(data_dict)

    (model1c, model2c, model3c, model4c), (model1r, model2r, model3r, model4r) = (
        sample_data_models(cfg)
    )

    basename = f"sim{simulation_id + 1}_mod"

    # Save metadata
    with Path.open(
        sim_conf_path / Path(f"sim{simulation_id + 1}").with_suffix(".yaml"),
        "w",
    ) as metafile:
        yaml.dump(data_dict, metafile)

    # Model 1
    model1r.to_parquet(data_path / Path(basename + "1r.pq"))

    model1c.to_parquet(data_path / Path(basename + "1c.pq"))

    # Model 2
    model2r.to_parquet(data_path / Path(basename + "2r.pq"))

    model2c.to_parquet(data_path / Path(basename + "2c.pq"))

    # Model 3
    model3r.to_parquet(data_path / Path(basename + "3r.pq"))

    model3c.to_parquet(data_path / Path(basename + "3c.pq"))

    # Model 4
    model4r.to_parquet(data_path / Path(basename + "4r.pq"))

    model4c.to_parquet(data_path / Path(basename + "4c.pq"))
