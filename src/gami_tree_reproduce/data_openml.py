from pathlib import Path

from omegaconf import OmegaConf
from sklearn.datasets import fetch_openml

ROOT = Path.cwd()
DATA = ROOT / "data"
CONF = ROOT / "conf" / "data" / "openml.yaml"


def download_data_openml(id: int, filename: str, destination=DATA) -> Path:
    data = fetch_openml(data_id=id, as_frame=True, parser="auto")
    frame = data.frame

    full_path = destination / filename
    frame.to_csv(full_path, index=False)


cfg = OmegaConf.load(CONF)
for dataset, dataset_cfg in cfg.items():
    download_data_openml(id=dataset_cfg["openml_id"], filename=dataset + ".csv")
