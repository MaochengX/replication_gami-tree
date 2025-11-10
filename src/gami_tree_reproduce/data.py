from pathlib import Path

from omegaconf import OmegaConf
from sklearn.datasets import fetch_openml

root = Path(__file__).resolve().parents[2]
dir_raw = root / "data" / "raw"
dir_raw.mkdir(parents=True, exist_ok=True)
path_configuration = root / "conf" / "data.yaml"


def get_openml_data(name: str, destination: Path = dir_raw) -> Path:
    """Download the bike sharing data from UCI using openML.

    TODO:
        - check whether download necessary or already existing
        - Improve docstring: example code, error handling
        - is more generic than openml necessary/needed?

    Args:
        name (str): Name of the dataset as used as group in configuration file (eg. bike and not openML name)
        destination(Path, optional):

    Returns:
        Path: Path to the file itself.
    """
    config = OmegaConf.load(path_configuration)
    config = config.get(name)
    filename = destination / name

    if not Path.is_file(filename):
        openml_name = config.get("openml-name")
        data = fetch_openml(name=openml_name, as_frame=True)
        data = data.frame
        data.to_parquet(filename, index=False)

    return filename


get_openml_data("bike")
