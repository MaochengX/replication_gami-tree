from pathlib import Path

import hydra
from omegaconf import DictConfig

root = Path.cwd()

@hydra.main(config_name="main.yaml", config_path=str(root/ "conf"), version_base=None)
def train_all(cfg:DictConfig ) -> None:
    """Train all models.

    For each model and each dataset train the model on the dataset and document
    results in assets folder. Configurations for each model are expected to be in a
    configuration file next to `main.yaml`.

    Args:
        cfg (dict): confinguration file for main script
    """
    cfg_model = cfg["cfg_models"]
    cfg_data = cfg["cfg_data"]


if __name__ == "__main__":
    train_all()