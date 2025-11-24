from pathlib import Path

import yaml
from sklearn.datasets import fetch_openml

root = Path.cwd()
path_config = root / "conf" / "config.yaml"
with Path.open(path_config) as f:
    cfg = yaml.safe_load(f)
path_data = Path(cfg["path_data"])


def download_data_openml(id: int, file_name: str) -> Path:
    destination = root / path_data
    if not Path.is_dir(destination):
        msg = f"{destination} is not an existing directory."
        raise NotADirectoryError(msg)

    data = fetch_openml(data_id=id, as_frame=True, parser="auto")
    frame = data.frame

    full_path = destination / Path(f"{file_name}.csv")
    frame.to_csv(full_path)
    return full_path


def get_data_from_conf(conf_data_folder: Path = root / "conf" / "datasets") -> None:
    for config_file in conf_data_folder.rglob("*.yaml"):
        with Path.open(config_file) as f:
            cfg = yaml.safe_load(f)
        if "openml_id" in cfg:
            id = cfg["openml_id"]
            file_name = cfg["name"]
            download_data_openml(id, file_name)


path_data.mkdir()
get_data_from_conf()
