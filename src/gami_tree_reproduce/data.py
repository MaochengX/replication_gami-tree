from pathlib import Path

import yaml
from sklearn.datasets import fetch_openml

root = Path.cwd()
path_config = root / "conf" / "data.yaml"
with Path.open(path_config) as f:
    cfg = yaml.safe_load(f)
path_data = cfg["path_data"]


def download_data_openml(id: int) -> Path:
    destination = root / path_data
    if not Path.is_dir(destination):
        msg = f"{destination} is not an existing directory."
        raise NotADirectoryError(msg)

    data = fetch_openml(data_id=id, as_frame=True, parser="auto")
    frame = data.frame

    data_name = data.details["name"]
    full_path = destination / Path(f"{data_name}.csv")
    frame.to_csv(full_path)
    return full_path
