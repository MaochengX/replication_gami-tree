from pathlib import Path

from omegaconf import OmegaConf
from sklearn.datasets import fetch_openml

from gami_tree_reproduce.utils import get_project_paths

project_paths = get_project_paths()


def download_data_openml(
    id: int, filename: str, destination=project_paths["data"]
) -> Path:
    data = fetch_openml(data_id=id, as_frame=True, parser="auto")
    frame = data.frame

    full_path = destination / Path(filename).with_suffix(".pq")
    frame.to_parquet(full_path, index=False)


cfg = OmegaConf.load(project_paths["conf_data_openml"])
for dataset, dataset_cfg in cfg.items():
    download_data_openml(id=dataset_cfg["openml_id"], filename=dataset + ".csv")
